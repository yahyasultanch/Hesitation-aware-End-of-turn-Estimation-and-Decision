"""Live check (costs a fraction of a cent): HaikuLLM inside a real Pipecat pipeline keeps thinking out of the
context, survives an interrupted reply, and answers the turn after it. Guards the crash from a test call.

  python demos/phone/check_haiku_context.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import bot  # noqa: E402
from pipecat.frames.frames import EndFrame, InterruptionFrame, LLMMessagesAppendFrame, LLMTextFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair, LLMSpecificMessage)
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402


class Collect(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.text = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.text += frame.text
        await self.push_frame(frame, direction)


async def main():
    os.environ["LLM"] = "anthropic"
    llm = bot.make_llm(bot.PROMPT.format(today="Sunday"))
    assert type(llm).__name__ == "HaikuLLM" and llm._settings.model == bot.HAIKU
    ctx = LLMContext([{"role": "assistant", "content": bot.GREETING}], tools=bot.TOOLS)
    pair, out = LLMContextAggregatorPair(ctx), Collect()
    task = PipelineTask(Pipeline([pair.user(), llm, out, pair.assistant()]))
    say = lambda t: LLMMessagesAppendFrame([{"role": "user", "content": t}], run_llm=True)

    async def drive():
        await asyncio.sleep(0.5)
        await task.queue_frames([say("Hi, which slots are free this week?")])  # turn 1 (calls open_slots)...
        await asyncio.sleep(1.0)
        await task.queue_frames([InterruptionFrame()])            # ...interrupted mid-reply, like the call
        await asyncio.sleep(1.0)
        out.text = ""
        await task.queue_frames([say("Sorry, go on. Is Thursday free?")])     # turn 2 must still work
        await asyncio.sleep(8)
        await task.queue_frames([EndFrame()])

    await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), drive())
    specific = [m for m in ctx.get_messages() if isinstance(m, LLMSpecificMessage)]
    print("turn 2 reply:", out.text.strip())
    print("thinking messages in context:", len(specific))
    assert not specific, "thinking leaked into the context"
    assert out.text.strip(), "no reply after the interruption"
    print("ok")


if __name__ == "__main__":
    asyncio.run(main())
