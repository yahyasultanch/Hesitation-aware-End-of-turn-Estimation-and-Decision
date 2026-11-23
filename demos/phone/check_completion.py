"""Live check (fractions of a cent): with the stutter-aware completion filter, does Haiku stay silent (for at least
3 s) on the fragments from a test call, and answer complete turns (including diary-tool turns) at once?

  python demos/phone/check_completion.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import bot  # noqa: E402
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame, LLMTextFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair, LLMUserAggregatorParams)
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402

# (what the agent last said, what the caller said, should the agent speak?)
CASES = [
    ("Is there anything else I can help you with?", "Yeah. Am", False),
    ("No problem, go ahead.", "wondering if", False),
    ("What would you like to ask?", "so I I I was I was I was thinking", False),
    ("Take your time.", "if it is possible to to", False),
    ("Take your time.", "post post post post post to", False),
    ("Which of those times would suit you best?", "Wednesday at nine please.", True),
    ("Could I start with your full name, please?", "It's Alex Morgan.", True),
    ("How can I help?", "Which slots are free this week?", True),
    ("How can I help?", "Please stop cutting me off, I stutter.", True),
    # complete sentences with stuttered repetitions must be answered (both were wrongly held in a test call)
    ("Is zero seven seven zero zero, nine zero zero, four five six correct?", "It it it is correct.", True),
    ("Let me read that back to you: zero seven seven zero zero, nine zero zero, four five six.", "As as what?", True),
    ("Which slot would you like?", "I I I think I like the the one on Tuesday.", True),
]


class Spoken(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.text, self.first, self.early, self.t0 = "", None, "", 0.0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and frame.text.strip() and not getattr(frame, "skip_tts", False):
            self.first = self.first or time.time()
            self.text += frame.text
            self.early += frame.text if time.time() - self.t0 <= 3.0 else ""  # 3 s: room for a diary lookup
        await self.push_frame(frame, direction)


async def one(agent_said, caller_said):
    os.environ["LLM"] = "anthropic"
    llm = bot.make_llm(bot.PROMPT.format(today="Sunday"))
    ctx = LLMContext([{"role": "assistant", "content": bot.GREETING},
                      {"role": "user", "content": "Hi."},
                      {"role": "assistant", "content": agent_said}], tools=bot.TOOLS)
    pair = LLMContextAggregatorPair(ctx, user_params=LLMUserAggregatorParams(
        filter_incomplete_user_turns=True, user_turn_completion_config=bot.COMPLETION))
    out = Spoken()
    task = PipelineTask(Pipeline([pair.user(), llm, out, pair.assistant()]))

    async def drive():
        await asyncio.sleep(0.3)
        t0 = out.t0 = time.time()
        await task.queue_frames([LLMMessagesAppendFrame([{"role": "user", "content": caller_said}], run_llm=True)])
        await asyncio.sleep(6)
        out.latency = (out.first - t0) if out.first else None
        await task.queue_frames([EndFrame()])

    await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), drive())
    si = llm._settings.system_instruction or ""  # what the model actually got, after Pipecat combined them
    assert "Riverside Physiotherapy" in si and "TURN COMPLETION" in si, "clinic prompt or completion rules missing"
    return out.text.strip(), out.latency, out.early.strip()


async def main():
    ok = 0
    for agent_said, caller_said, should_speak in CASES:
        text, lat, early = await one(agent_said, caller_said)
        good = (bool(early) == should_speak and not any(m in text for m in "✓○◐")
                and len(early) < 320)  # the clinic prompt asks for one or two short sentences
        ok += good
        later = f"; if they never go on, at {lat:.1f}s: {text[:55]!r}" if text and not early else ""
        print(f"{'PASS' if good else 'FAIL'}  caller: {caller_said!r:55} -> "
              f"{'says ' + repr(early[:60]) + f' ({lat:.1f}s)' if early else 'silent for 3 s' + later}")
    print(f"\n{ok}/{len(CASES)} as intended")


if __name__ == "__main__":
    asyncio.run(main())
