import asyncio

import numpy as np
import pytest

SR = 16000
CHUNK = b"\0\0" * 320  # 20 ms pcm16


def test_livekit_detector_holds_and_scales():
    from turnheed.plugins.livekit import FloorTurnDetector

    class Inner:
        async def predict_end_of_turn(self, chat_ctx, *, timeout=None):
            return 0.8

    still = {"p": None}
    det = FloorTurnDetector(Inner(), still_talking=lambda: still["p"])
    run = lambda: asyncio.run(det.predict_end_of_turn(None))
    assert run() == pytest.approx(0.8)
    still["p"] = 0.9
    assert run() == pytest.approx(0.08)
    still["p"], det.expected = None, "digits"
    assert run() < 0.8


class Fixed:  # still-talking stand-in with the push/read protocol
    def __init__(self, p, kind): self.r, self.pushed = (p, kind), 0
    def push(self, pcm16, sample_rate=16000): self.pushed += 1
    def read(self): return self.r


def test_pipecat_analyzer_holds_then_ceiling():
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
    from turnheed.plugins.pipecat import FloorTurnAnalyzer
    from turnheed import FloorPolicy

    class Inner(BaseTurnAnalyzer):  # Smart Turn stand-in that always says "complete, p=0.9"
        speech_triggered = True
        params = None
        def append_audio(self, b, s): return EndOfTurnState.INCOMPLETE
        async def analyze_end_of_turn(self):
            class M: probability = 0.9
            return EndOfTurnState.COMPLETE, M()
        def clear(self): pass

    # no hold signal: Smart Turn's "complete" ends the turn at VAD stop
    a = FloorTurnAnalyzer(inner=Inner())
    a.set_sample_rate(SR)
    for _ in range(50): a.append_audio(CHUNK, True)
    for _ in range(15): a.append_audio(CHUNK, False)  # 300 ms silence, past min_silence
    assert asyncio.run(a.analyze_end_of_turn())[0] == EndOfTurnState.COMPLETE

    # "still talking" during a block: hold through Smart Turn's "complete", end at the ceiling
    a = FloorTurnAnalyzer(inner=Inner(), policy=FloorPolicy(max_hold=1.0), still_talking=Fixed(0.99, "block"))
    a.set_sample_rate(SR)
    for _ in range(50): a.append_audio(CHUNK, True)
    for _ in range(15): a.append_audio(CHUNK, False)
    assert asyncio.run(a.analyze_end_of_turn())[0] == EndOfTurnState.INCOMPLETE
    assert a.last_decision.reason == "mid_turn_block"
    for n in range(100):
        if a.append_audio(CHUNK, False) == EndOfTurnState.COMPLETE:
            break
    assert a.last_decision.reason == "hold_ceiling" and n < 100


def test_pipecat_real_smart_turn_smoke():
    pytest.importorskip("pipecat")
    from turnheed.plugins.pipecat import FloorTurnAnalyzer

    a = FloorTurnAnalyzer()
    a.set_sample_rate(SR)
    tone = (np.sin(np.arange(SR) * 2 * np.pi * 220 / SR) * 8000).astype(np.int16).tobytes()
    for i in range(0, len(tone), 640): a.append_audio(tone[i:i + 640], True)
    for _ in range(15): a.append_audio(CHUNK, False)
    state, metrics = asyncio.run(a.analyze_end_of_turn())
    assert metrics is not None and 0 <= metrics.probability <= 1


def test_pipecat_transcript_without_vad_restarts_the_wait():
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
    from turnheed.plugins.pipecat import FloorTurnAnalyzer

    class Inner(BaseTurnAnalyzer):
        speech_triggered = False
        params = None
        def append_audio(self, b, s): return EndOfTurnState.INCOMPLETE
        async def analyze_end_of_turn(self): return EndOfTurnState.INCOMPLETE, None
        def clear(self): pass

    a = FloorTurnAnalyzer(inner=Inner())
    a.set_sample_rate(SR)
    for _ in range(50): a.append_audio(CHUNK, True)      # earlier speech, then a long silence
    for _ in range(100): a.append_audio(CHUNK, False)    # 2 s: the earlier turn is long over
    a.heard_speech()                                     # transcript of a syllable the VAD missed
    states = [a.append_audio(CHUNK, False) for _ in range(100)]
    first = states.index(EndOfTurnState.COMPLETE) if EndOfTurnState.COMPLETE in states else None
    assert first is not None and first * 0.02 >= 0.5     # waits from the syllable, not from the old speech


def test_pipecat_transcript_without_vad_ignores_stale_sound():
    # live call: a sentence the VAD missed inherited an old "still talking" 0.9999 and waited the full 5 s ceiling
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
    from turnheed.plugins.pipecat import FloorTurnAnalyzer

    class Inner(BaseTurnAnalyzer):
        speech_triggered = False
        params = None
        def append_audio(self, b, s): return EndOfTurnState.INCOMPLETE
        async def analyze_end_of_turn(self): return EndOfTurnState.INCOMPLETE, None
        def clear(self): pass

    from turnheed import FloorPolicy

    class Sure:  # still-talking model that is (wrongly) certain
        def push(self, pcm, sr): pass
        def read(self): return 0.9999, "block"

    a = FloorTurnAnalyzer(inner=Inner(), policy=FloorPolicy(max_hold=5.0), still_talking=Sure())
    a.set_sample_rate(SR)
    for _ in range(50): a.append_audio(CHUNK, True)
    for _ in range(300): a.append_audio(CHUNK, False)
    a.heard_speech()
    states = [a.append_audio(CHUNK, False) for _ in range(300)]
    first = states.index(EndOfTurnState.COMPLETE)
    assert first * 0.02 <= 1.0                           # words decide: ~0.8 s, not the 5 s ceiling


def test_pipecat_late_transcript_keeps_the_pause_reading():
    # live call: Smart Turn said finished, the transcript landed 0.1 s later and wiped it -> 5 s hold ceiling
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
    from turnheed import FloorPolicy
    from turnheed.plugins.pipecat import FloorTurnAnalyzer

    class Inner(BaseTurnAnalyzer):
        speech_triggered = False
        params = None
        def append_audio(self, b, s): return EndOfTurnState.INCOMPLETE
        async def analyze_end_of_turn(self): return EndOfTurnState.COMPLETE, None
        def clear(self): pass

    a = FloorTurnAnalyzer(inner=Inner(), policy=FloorPolicy(max_hold=5.0))
    a.set_sample_rate(SR)
    for _ in range(50): a.append_audio(CHUNK, True)
    for _ in range(10): a.append_audio(CHUNK, False)
    a.policy.silence_timeout = 0.3  # make "finished" decisive without a wait model
    asyncio.run(a.analyze_end_of_turn())
    a._complete = False             # pretend the policy held, then the transcript arrives
    a.heard_speech()
    assert a._p_finished == 1.0     # the pause's sound verdict survives a normal STT lag


def test_floor_turn_start_ignores_noise_and_backchannels_while_agent_speaks():
    # live call: a half-second sound on the caller's line stopped the agent mid-sentence (11 s of dead air)
    pytest.importorskip("pipecat")
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterimTranscriptionFrame
    from pipecat.turns.types import ProcessFrameResult
    from turnheed.plugins.pipecat import FloorTurnStartStrategy

    def said(text):
        return InterimTranscriptionFrame(text=text, user_id="", timestamp="")

    async def run():
        s = FloorTurnStartStrategy()
        await s.process_frame(BotStartedSpeakingFrame())
        out = [await s.process_frame(said(t)) for t in ("", "Mm.", "Mhmm.", "mm-hmm", "okay", "yeah okay")]
        assert all(r == ProcessFrameResult.CONTINUE for r in out)          # agent keeps talking
        assert await s.process_frame(said("wait")) == ProcessFrameResult.STOP  # a real word yields
        await s.process_frame(BotStoppedSpeakingFrame())
        assert await s.process_frame(said("okay")) == ProcessFrameResult.STOP  # agent silent: "okay" is an answer
    asyncio.run(run())
