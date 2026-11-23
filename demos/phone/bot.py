"""Phone demo: a physiotherapy clinic booking line over Twilio, in two modes with everything else identical.

  baseline   Pipecat's default turn-taking (Smart Turn v3.2, 3 s fallback)
  layer      the floor policy + still-talking v2, holding through blocks; reads numbers back in groups

  python demos/phone/bot.py            # serves on :8765; see demos/phone/README.md for Twilio + tunnel

Twilio's voice webhook -> POST https://<public host>/twiml   (mode: ?mode=, or switch locally with
  curl http://localhost:8765/mode/baseline  |  /mode/layer)
Each call logs its cut-off count (caller kept talking within 1.5 s of the agent starting) and, with
RECORD=1, saves a stereo wav (caller left, agent right) to demos/phone/recordings/ for the split screen.
"""
import asyncio
import base64
import dataclasses
import datetime
import hashlib
import hmac
import html
import re
import secrets
import urllib.parse
import os
import sys
import time
import wave

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, LLMFullResponseEndFrame, LLMRunFrame, LLMUpdateSettingsFrame, LLMTextFrame, TranscriptionFrame, TTSSpeakFrame,
                                   UserStartedSpeakingFrame, UserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.settings import LLMSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import (USER_TURN_COMPLETION_INSTRUCTIONS,
                                                      UserTurnCompletionConfig)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
from turnheed import FloorPolicy  # noqa: E402
from turnheed.plugins.pipecat import FloorTurnAnalyzer, FloorTurnStartStrategy, FloorTurnStopStrategy  # noqa: E402
from turnheed.still_talking import StillTalkingV2  # noqa: E402
from turnheed.wait import LearnedWait  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from clinic import SLOTS, Diary  # noqa: E402

_SECRET = re.compile(r"(Token |Bearer |sk-ant-[\w-]*|gsk_|AIza|AC[0-9a-f]{32})[\w.\-]*")


def _redacted_sink(message):  # every log line, including Pipecat's own (it prints auth headers on errors)
    sys.stderr.write(_SECRET.sub(lambda m: m.group(1)[:7] + "<redacted>", str(message)))


logger.remove()
logger.add(_redacted_sink, level=os.getenv("LOG_LEVEL", "INFO"))
if os.getenv("DEBUG_LOG"):  # full Pipecat detail for diagnosing a call (redacted; holds personal data; local only)
    _dbg = open(os.getenv("DEBUG_LOG"), "a", encoding="utf8")
    logger.add(lambda m: (_dbg.write(_SECRET.sub(lambda x: x.group(1)[:7] + "<redacted>", str(m))), _dbg.flush()),
               level="DEBUG")
TRANSCRIPTS = os.getenv("TRANSCRIPTS") == "1"  # caller/agent lines in the log; off by default (personal data)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(ROOT, ".env"))  # project-root .env works too
MODELS = os.path.join(ROOT, "data", "still_talking")
RECORDINGS = os.path.join(os.path.dirname(__file__), "recordings")

PROMPT = """You are the phone receptionist for Riverside Physiotherapy. You are an AI, and you say so.
Today is {today}. The diary holds this week's appointments, Monday to Friday.
Your only job is booking, moving or cancelling physiotherapy appointments; for anything else, say so politely.
Keep every reply to one or two short sentences: this is a phone call, and everything you write is spoken.
Never ask the caller to speak faster, slower or more clearly, and never finish their sentences for them.

The diary tools are the only truth about bookings. Never guess, remember or invent a booking or a free slot:
- Free slots: call open_slots.
- Anything about an existing booking (telling it, moving it, cancelling it): first get the caller's full name
  AND phone number, read the number back and get a yes, then call find_booking.
  If it finds nothing, say no booking matches that name and number, and offer to book a new one. Never tell
  or change a booking the caller has not verified, even if they say which day it is.
- To book: name, phone number (read back and confirmed) and a free slot, then call book.
- To read a phone number back, call read_back_phone and say exactly the words it returns; never write
  the digits yourself.
- Only say a booking was made, moved or cancelled after the tool returned ok; then say the slot once.
When you pass a phone number to a tool, pass it exactly as the caller said it (words or digits); the diary
converts it. Ask for one thing at a time. No lists, no markdown, no emoji."""

GREETING = ("Hello, you've reached Riverside Physiotherapy. I'm an AI assistant"
            + (", and this call is recorded for a demo" if os.getenv("RECORD") == "1" else "") + ". How can I help?")

# ponytail: keyword guess at what the agent just asked for; a real agent would set this from its dialogue state
EXPECT = [("spell", "spelling"), ("phone number", "digits"), ("number", "digits"), ("yes or no", "yesno")]


class CallerLog(FrameProcessor):
    """Logs what the caller said (final transcripts), so every call can be read back in full, and gives a
    caller who asks for time the maximum session patience."""

    def __init__(self, analyzer=None):
        super().__init__()
        self.analyzer, self.patient = analyzer, False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            if TRANSCRIPTS:
                logger.info(f"CALLER: {frame.text.strip()}")
            low = frame.text.lower()
            # letters one at a time ("y a h", "S u.") are spelling even if the agent didn't ask for it: never rush
            if self.analyzer is not None and all(len(w.strip(".,?!")) <= 1 for w in low.split()):
                self.analyzer.expected = "spelling"
            asked = any(k in low for k in ASK_FOR_TIME)
            if self.analyzer is not None and asked:
                pol = self.analyzer.policy
                if pol.patience < 1.5:
                    pol.max_patience = max(pol.max_patience, 1.5)
                    pol.patience = 1.5
                    logger.info(f"floor: caller asked for time -> patience {pol.patience:.2f} for this call")
            pauses = self.analyzer.policy.pauses if self.analyzer is not None else []
            long_pauser = len(pauses) >= 3 and sorted(pauses)[len(pauses) // 2] > 1.2
            if self.analyzer is not None and not self.patient and (asked or long_pauser):
                self.patient = self.analyzer.patient_checkins = True  # this call only
                await self.push_frame(LLMUpdateSettingsFrame(delta=LLMSettings(
                    filter_incomplete_user_turns=True, user_turn_completion_config=PATIENT_COMPLETION)))
                logger.info("floor: longer check-ins (8 s / 15 s) for the rest of this call "
                            f"({'caller asked for time' if asked else 'their pauses run long'})")
        await self.push_frame(frame, direction)


class CallWatch(FrameProcessor):
    """Counts cut-offs (caller resumes within 1.5 s of the agent starting to speak) and tells the floor
    analyzer what kind of answer the agent just asked for."""

    DEAD_AIR = 6.0  # s of silence after the caller's turn with no reply and no check-in due -> ask the model again
    AFTER_INTERRUPT = 2.0  # s: the caller stopped the agent mid-sentence and nothing followed -> carry on sooner

    def __init__(self, analyzer=None):
        super().__init__()
        self.analyzer, self.cutoffs, self._bot_start, self._text = analyzer, 0, None, ""
        self._guard, self._bot_speaking, self._interrupted = None, False, False

    async def _dead_air(self, delay):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._guard, self._bot_speaking, self._interrupted = None, False, False
        logger.info(f"floor: {delay:.0f}s of dead air after the caller's turn -> asking the model again")
        await self.push_frame(LLMRunFrame())  # the assistant aggregator turns this into a fresh reply

    def _arm(self, delay):
        if self._guard:
            self._guard.cancel()
        self._guard = asyncio.get_running_loop().create_task(self._dead_air(delay))

    def _disarm(self):
        if self._guard:
            self._guard.cancel()
            self._guard, self._bot_speaking, self._interrupted = None, False, False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._arm(self.AFTER_INTERRUPT if self._interrupted else self.DEAD_AIR)
            self._interrupted = False
        if isinstance(frame, BotStartedSpeakingFrame):
            self._disarm()
            self._bot_start, self._bot_speaking = time.monotonic(), True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._disarm()
            self._interrupted = self._bot_speaking
            if self._bot_start is not None and time.monotonic() - self._bot_start < 1.5:
                self.cutoffs += 1
                logger.info(f"cut-off #{self.cutoffs}")
            self._bot_start = None
        elif isinstance(frame, LLMTextFrame):
            if getattr(frame, "skip_tts", False):  # completion marker or a reply to an incomplete turn: not spoken
                if frame.text.strip() in ("○", "◐"):
                    logger.info(f"AGENT: (stays silent: caller is mid-sentence {frame.text.strip()})")
                    # a check-in is due after the completion timeout; only step in if even that never comes
                    cfg = PATIENT_COMPLETION if getattr(self.analyzer, "patient_checkins", False) else COMPLETION
                    t = cfg.incomplete_short_timeout if frame.text.strip() == "○" else cfg.incomplete_long_timeout
                    self._arm(t + self.DEAD_AIR)
            else:
                self._text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._text.strip() and TRANSCRIPTS:
                logger.info(f"AGENT: {self._text.strip()}")
            if self.analyzer is not None:
                low = self._text.lower()
                self.analyzer.expected = next((v for k, v in EXPECT if k in low), None)
            self._text = ""
        await self.push_frame(frame, direction)


class DeepgramSTT(DeepgramSTTService):
    """ponytail: pipecat 0.0.108 on Python 3.10 sends language=Language.EN (str() of its enum) and Deepgram
    answers 400; send the enum's value. Drop this once pipecat fixes it upstream."""

    def _build_connect_kwargs(self):
        kw = super()._build_connect_kwargs()
        if str(kw.get("language", "")).startswith("Language."):
            kw["language"] = Language[kw["language"].split(".", 1)[1]].value
        return kw


class GroqLLM(GroqLLMService):
    """Merges back-to-back messages of the same role before each request. A caller who spells their name
    arrives as several user turns; on that history Qwen sometimes returns an empty reply (seen on a live
    call), which leaves the caller in silence and stacks up even more user turns."""

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        merged = []
        for m in params.get("messages") or []:
            same = merged and m.get("role") == merged[-1].get("role") in ("user", "assistant")
            if same and isinstance(m.get("content"), str) and isinstance(merged[-1].get("content"), str):
                merged[-1] = {**merged[-1], "content": merged[-1]["content"] + " " + m["content"]}
            else:
                merged.append(m)
        if merged:
            params["messages"] = merged
        return params


DIARY = Diary()  # shared by all calls until restart
TOOLS = ToolsSchema(standard_tools=[
    FunctionSchema("open_slots", "List the free appointment slots this week.", {}, []),
    FunctionSchema("read_back_phone", "The exact words to read a phone number back to the caller.",
                   {"phone": {"type": "string", "description": "as the caller said it"}}, ["phone"]),
    FunctionSchema("find_booking", "Find the caller's booking. Needs the full name and the confirmed phone number.",
                   {"name": {"type": "string"}, "phone": {"type": "string", "description": "as the caller said it"}}, ["name", "phone"]),
    FunctionSchema("book", "Book a free slot, e.g. 'Thursday 14:00', for a new patient.",
                   {"name": {"type": "string"}, "phone": {"type": "string", "description": "as the caller said it"},
                    "slot": {"type": "string"}},
                   ["name", "phone", "slot"]),
    FunctionSchema("move", "Move the verified caller's booking to another free slot.",
                   {"name": {"type": "string"}, "phone": {"type": "string", "description": "as the caller said it"},
                    "new_slot": {"type": "string"}},
                   ["name", "phone", "new_slot"]),
    FunctionSchema("cancel", "Cancel the verified caller's booking.",
                   {"name": {"type": "string"}, "phone": {"type": "string", "description": "as the caller said it"}}, ["name", "phone"]),
])


async def diary_tool(params):
    fn = {"open_slots": DIARY.open_slots, "read_back_phone": DIARY.read_back_phone,
          "find_booking": DIARY.find_booking, "book": DIARY.book,
          "move": DIARY.move, "cancel": DIARY.cancel}[params.function_name]
    try:
        result = fn(**params.arguments)
    except TypeError as e:  # the model sent wrong arguments
        result = {"ok": False, "message": f"bad arguments: {e}"}
    logger.info(f"tool {params.function_name}({dict(params.arguments) if TRANSCRIPTS else '...'}) -> "
                f"{result if TRANSCRIPTS else {k: v for k, v in result.items() if k in ('ok', 'found')}}")
    await params.result_callback(result)


HAIKU = "claude-haiku-5-5"


def _haiku_cls():
    from pipecat.frames.frames import LLMThoughtEndFrame, LLMThoughtStartFrame, LLMThoughtTextFrame
    from pipecat.services.anthropic.llm import AnthropicLLMService

    class HaikuLLM(AnthropicLLMService):
        """ponytail: pipecat 0.0.108 stores each thinking block in the context and converts it back only if it
        has text AND a signature. Haiku 5.5 returns thinking with empty text by default, and an interrupted
        reply has no signature, so the next request crashed ("Unknown error occurred: 'role'") and the agent
        went silent on a live call. The receptionist doesn't need its thinking replayed: drop it."""

        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, (LLMThoughtStartFrame, LLMThoughtTextFrame, LLMThoughtEndFrame)):
                return
            await super().push_frame(frame, direction)

    return HaikuLLM


# The words decide too: Haiku marks each caller turn complete or not before replying (Pipecat's
# filter_incomplete_user_turns), and a reply to an incomplete turn is never spoken. No extra latency on complete
# turns; on "so I was thinking if" the agent keeps listening instead of saying "take your time".
COMPLETION = UserTurnCompletionConfig(
    instructions=USER_TURN_COMPLETION_INSTRUCTIONS + """
THIS CALLER MAY STUTTER. On a phone line speech arrives in pieces, so judge the words, not the pause.
First mentally remove repeated words and sounds ("I I I", "it it it", "post post post", "b- b-"); they say
nothing about whether the sentence is finished. The transcript may have no punctuation or capitals: a missing
full stop is not a sign the caller is unfinished. Then judge the LAST words:
- They break off mid-phrase: the turn ends on "and", "to", "the", "a", "if", "because", "is", "between", "my", or
  is only an opening ("so I was thinking", "if it is possible to", "my number is three two") -> `○`.
- They complete a sentence or question ("It it it is correct." -> "It is correct.", "do you know the address")
  -> COMPLETE: `✓`, and answer it. Earlier broken-off parts don't matter if the last thought is complete.
- A short reply or acknowledgement ("yes", "okay", "mhm", "sure thank you", their name, a full number) is
  COMPLETE: `✓` and answer at once.
- A longer answer that has made its point and trails off ("...so", "...yeah", "...so yeah", "...you know",
  "...or something", "...I don't know", "...that's it") is handing the turn back: COMPLETE: `✓`.
- Never finish the caller's sentence for them and never guess what they were about to say.
- If a COMPLETE turn needs a diary tool, start with `✓` and then call the tool.""",
    # 2.5 s, not 8: on real phone calls (SpokenWOZ) Haiku also says "mid-sentence" on some finished turns, and
    # every such veto made the caller wait the whole timeout. A short re-check that answers when it can caps it.
    incomplete_short_timeout=2.5,
    incomplete_long_timeout=8.0,
    incomplete_short_prompt="""The caller stopped a moment ago and has not continued.
You MUST start with ✓. If what the caller has said so far is enough to answer or act on, do that now, normally.
If the caller has spoken since (e.g. "That's all.", an answer), respond to what they said instead.
Only if it is not, say one short, calm sentence that invites them to carry on in their own time, without
guessing what they were going to say and without asking them to speak faster or more clearly.
Example of the second case: ✓ I'm still here, go on whenever you're ready.""",
    incomplete_long_prompt="""The caller has been quiet for a long time.
You MUST respond with ✓. If the caller has spoken since (e.g. "That's all.", an answer), respond to what they
said. Otherwise one short, warm sentence: you are still here and there is no rush.
Example: ✓ No rush at all, I'm still here when you're ready.""",
)


# Re-check time is targeted too: 2.5 s caps the cost of a wrong "mid-sentence" for most callers; a caller who asks
# for time, or whose own pauses run long, gets 8 s / 15 s for the rest of the call (a 2.5 s check-in on a block
# is exactly the "take your time" interjection that came across badly in a test call).
PATIENT_COMPLETION = dataclasses.replace(COMPLETION, incomplete_short_timeout=8.0, incomplete_long_timeout=15.0)


def make_llm(system: str):
    """LLM= anthropic | google | groq; default: the first one with a key, in that order."""
    keys = (("anthropic", "ANTHROPIC_API_KEY"), ("google", "GOOGLE_API_KEY"), ("groq", "GROQ_API_KEY"))
    which = os.getenv("LLM") or next((n for n, k in keys if os.getenv(k)), "groq")
    if which == "anthropic":
        HaikuLLM = _haiku_cls()
        # Haiku only, on purpose: cheapest current Claude model ($0.10 / $0.50 per MTok); no env override.
        # effort low keeps its (always-on, adaptive) reasoning short: cheaper and faster for 1-2 sentence replies.
        # max_tokens is a cap, not a charge; 1024 leaves room so replies are never cut off mid-sentence.
        llm = HaikuLLM(api_key=os.environ["ANTHROPIC_API_KEY"], settings=HaikuLLM.Settings(
            model=HAIKU, max_tokens=1024, extra={"output_config": {"effort": "low"}}, system_instruction=system))
        assert llm._settings.model == HAIKU
    elif which == "google":
        from pipecat.services.google.llm import GoogleLLMService
        llm = GoogleLLMService(api_key=os.environ["GOOGLE_API_KEY"], settings=GoogleLLMService.Settings(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash"), system_instruction=system))
    else:
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"], settings=GroqLLMService.Settings(
            model=model, extra={"reasoning_effort": "low"} if "gpt-oss" in model else {}, system_instruction=system))
    llm.register_function(None, diary_tool)  # every tool goes to the diary
    logger.info(f"LLM: {which}")
    return llm


class LoggedFloorTurnAnalyzer(FloorTurnAnalyzer):
    """Logs every turn end the floor policy makes: reason, signals and how long it waited."""

    def _step(self, is_speech, buffer=None):
        done_before = self._complete
        done = super()._step(is_speech, buffer)
        d = self.last_decision
        if done and not done_before and d is not None:
            logger.info(f"floor: respond ({d.reason}); p_still={d.p_still_talking} p_finished={d.p_finished} "
                        f"patience={self.policy.patience:.2f} expected={self.expected}")
        return done


# the caller asking for time is a request, not a label: max patience for the rest of this call only
ASK_FOR_TIME = ("stutter", "stop cutting", "let me finish", "hold on", "give me a moment", "give me a second",
                "wait a moment", "wait a second", "you keep cutting", "interrupting me")


def turn_stop(mode: str):
    if mode == "baseline":
        return None, None  # Pipecat defaults: Smart Turn v3.2
    still = StillTalkingV2(os.path.join(MODELS, "smart_turn_embed.onnx"),
                           os.path.join(MODELS, "still_talking_v2_head.onnx"), threads=2)
    # Learned wait (bench/recipes/fit_wait.py): fast when the evidence says finished, longer only where real
    # mid-turn pauses of that kind run long, and longer still for a caller whose own pauses are long (this call
    # only, never shorter than the population). No blanket patience growth: on the bench it made everyone late.
    wm_path = os.path.join(MODELS, "wait_model_phone.json")  # fitted on phone-line signals (8 kHz mu-law)
    wm_path = wm_path if os.path.exists(wm_path) else os.path.join(MODELS, "wait_model.json")
    wait_model = LearnedWait.load(wm_path, theta=0.5) if os.path.exists(wm_path) else None
    # Targeted fast path (the words veto below keeps mid-sentence turns silent): 0.5 s until the caller has had two
    # mid-turn pauses over 1 s. Replay: phone calls 1050 -> 500 ms median reply; stuttered cut-offs 5.0% -> 5.1%.
    policy = FloorPolicy(silence_timeout=0.8, max_hold=5.0, max_patience=1.0, wait_model=wait_model, fast_wait=0.5)
    analyzer = LoggedFloorTurnAnalyzer(policy=policy, still_talking=still)
    # noise, breath and "mm" never take the turn or stop the agent; words do (see FloorTurnStartStrategy)
    return UserTurnStrategies(start=[FloorTurnStartStrategy()],
                              stop=[FloorTurnStopStrategy(turn_analyzer=analyzer)]), analyzer


async def run_call(websocket: WebSocket, mode: str):
    _, call = await parse_telephony_websocket(websocket)
    sid, token = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    serializer = TwilioFrameSerializer(  # without credentials the caller hangs up, not the agent
        stream_sid=call["stream_id"], call_sid=call["call_id"], account_sid=sid, auth_token=token,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=bool(sid and token)))
    transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer))

    stt = DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = DeepgramTTSService(api_key=os.environ["DEEPGRAM_API_KEY"], settings=DeepgramTTSService.Settings(
        voice=os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en")))
    # The prompt is the LLM's system_instruction, not a context message: with the completion filter on, Pipecat
    # appends its rules to system_instruction and DISCARDS a context system message (a live call in a test call ran
    # with no clinic prompt at all because of this).
    llm = make_llm(PROMPT.format(today=datetime.date.today().strftime("%A")))

    strategies, analyzer = turn_stop(mode)
    context = LLMContext([], tools=TOOLS)
    pair = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(), user_turn_strategies=strategies,
        user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        filter_incomplete_user_turns=(mode == "layer"), user_turn_completion_config=COMPLETION))
    watch = CallWatch(analyzer)
    audio = AudioBufferProcessor(num_channels=2)

    pipeline = Pipeline([transport.input(), stt, CallerLog(analyzer), pair.user(), llm, watch, tts, transport.output(), audio,
                         pair.assistant()])
    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
                        enable_rtvi=False)  # RTVI is for browser clients; its error reporting crashed on Deepgram errors
    record = os.getenv("RECORD") == "1"
    chunks = []

    @audio.event_handler("on_audio_data")
    async def on_audio(_, pcm, sample_rate, channels):
        chunks.append((pcm, sample_rate, channels))

    @transport.event_handler("on_client_connected")
    async def on_connected(_t, _c):
        if record:
            await audio.start_recording()
        await task.queue_frames([TTSSpeakFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_t, _c):
        if record:
            await audio.stop_recording()
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)
    logger.info(f"diary changes so far: {DIARY.log}")
    logger.info(f"call ended: mode={mode} cut-offs={watch.cutoffs}"
                + (f" policy events={analyzer.policy.events}" if analyzer else ""))
    if record and chunks:
        os.makedirs(RECORDINGS, exist_ok=True)
        path = os.path.join(RECORDINGS, f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{mode}-cutoffs{watch.cutoffs}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(chunks[0][2]), w.setsampwidth(2), w.setframerate(chunks[0][1])
            w.writeframes(b"".join(c[0] for c in chunks))
        logger.info(f"saved {path}")


app = FastAPI()
MODE = {"current": os.getenv("DEMO_MODE", "layer")}
CALL_TOKENS: dict[str, tuple[str, float]] = {}   # one-time stream token -> (mode, expiry)
ACTIVE = {"calls": 0}
MAX_CALLS = int(os.getenv("MAX_CALLS", "2"))


FORWARDED = ("cf-connecting-ip", "x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded")


def local_only(request) -> bool:
    """True for requests from this machine itself. A tunnel (ngrok, cloudflared) also connects from localhost, so
    any proxy-forwarding header means the request came from the internet."""
    return request.client.host in ("127.0.0.1", "::1") and not any(h in request.headers for h in FORWARDED)


def twilio_signature_ok(url: str, params: dict, signature: str) -> bool:
    """Twilio's X-Twilio-Signature: base64(HMAC-SHA1(auth token, url + sorted key+value pairs))."""
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not token:
        return True  # not configured: allowed, with a warning at startup
    data = url + "".join(k + params[k] for k in sorted(params))
    want = base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(want, signature or "")


@app.get("/health")
async def health():
    return {"ok": True}  # no data: lets start.py check the public address is still up


@app.get("/mode/{mode}")
async def set_mode(mode: str, request: Request):
    """Switch the mode for the next call from this machine: curl http://localhost:8765/mode/baseline"""
    if not local_only(request):
        return Response(status_code=403)
    if mode in ("baseline", "layer"):
        MODE["current"] = mode
    return {"mode": MODE["current"]}


@app.get("/bookings")
async def bookings(request: Request):
    """Staff view of the clinic database and its audit trail (this machine only), refreshing every 3 s."""
    if not local_only(request):
        return Response(status_code=403)
    esc = html.escape
    rows = "".join(f"<tr><td>{esc(s)}</td><td>{esc(b['name'])}</td><td>{esc(b['phone'])}</td></tr>"
                   for s, b in sorted(DIARY.bookings.items(), key=lambda kv: SLOTS.index(kv[0])))
    audit = "".join(f"<tr><td>{datetime.datetime.fromtimestamp(t):%d %b %H:%M:%S}</td><td>{esc(a)}</td>"
                    f"<td>{esc(d)}</td></tr>" for t, a, d in DIARY.audit_rows())
    page = f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<title>Riverside Physiotherapy: bookings</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:56rem}}table{{border-collapse:collapse;width:100%;margin:.5rem 0 2rem}}
td,th{{border-bottom:1px solid #ddd;padding:.4rem .6rem;text-align:left}}th{{background:#f4f4f4}}</style>
<h1>Bookings</h1><p>Database: {esc(DIARY.path)} &middot; mode for the next call: {esc(MODE["current"])}</p>
<table><tr><th>Slot</th><th>Patient</th><th>Phone</th></tr>{rows}</table>
<h2>Audit trail</h2><table><tr><th>When</th><th>Action</th><th>Detail</th></tr>{audit}</table>"""
    return Response(content=page, media_type="text/html")


@app.post("/twiml")
async def twiml(request: Request):
    form = dict(urllib.parse.parse_qsl((await request.body()).decode(), keep_blank_values=True))  # Twilio posts a form
    host = os.getenv("PUBLIC_HOST") or request.headers.get("host", "")
    url = f"https://{host}{request.url.path}" + (f"?{request.url.query}" if request.url.query else "")
    if not twilio_signature_ok(url, form, request.headers.get("x-twilio-signature", "")):
        logger.warning("rejected /twiml: bad Twilio signature")
        return Response(status_code=403)
    mode = request.query_params.get("mode", MODE["current"])
    mode = mode if mode in ("baseline", "layer") else "layer"
    now = time.monotonic()
    for k in [k for k, (_, exp) in CALL_TOKENS.items() if exp < now]:
        del CALL_TOKENS[k]
    token = secrets.token_urlsafe(24)
    CALL_TOKENS[token] = (mode, now + 60)  # one use, one minute
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
           f'<Stream url="wss://{html.escape(host)}/ws/{token}"/></Connect></Response>')
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws/{token}")
async def ws(websocket: WebSocket, token: str):
    entry = CALL_TOKENS.pop(token, None)  # only a stream announced by a (signed) /twiml request may connect
    if entry is None or entry[1] < time.monotonic() or ACTIVE["calls"] >= MAX_CALLS:
        await websocket.close(code=4403)
        return
    await websocket.accept()
    ACTIVE["calls"] += 1
    try:
        await run_call(websocket, entry[0])
    finally:
        ACTIVE["calls"] -= 1


if __name__ == "__main__":
    if not os.getenv("TWILIO_AUTH_TOKEN"):
        logger.warning("TWILIO_AUTH_TOKEN not set: Twilio signatures are not checked (fine on a laptop demo, "
                       "not for anything public)")
    # localhost only: the tunnel connects locally; nothing else on the network can reach the agent
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8765")))
