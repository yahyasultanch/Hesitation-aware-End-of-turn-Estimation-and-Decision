"""Replay the caller's side of a recorded demo call through the layer, offline, and list where it would have
ended the caller's turn. Compares policy settings on the same real audio (VAD path only: turns that start
from a transcript need live STT).

  python demos/phone/replay_audio.py demos/phone/recordings/<call>.wav [start_s end_s]
"""
import asyncio
import os
import sys
import wave

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState  # noqa: E402
from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADState  # noqa: E402

from turnheed import FloorPolicy  # noqa: E402
from turnheed.plugins.pipecat import FloorTurnAnalyzer  # noqa: E402
from turnheed.still_talking import StillTalkingV2  # noqa: E402

MODELS = os.path.join(ROOT, "data", "still_talking")
SETTINGS = {
    "old (tau 0.3, patience <= 1)": dict(silence_timeout=0.8, tau=0.3, max_patience=1.0),
    "patient defaults, 3 s hold": dict(silence_timeout=0.8),
    "demo now (6 s hold)": dict(silence_timeout=0.8, max_hold=6.0),
}


async def replay(caller: np.ndarray, sr: int, kw: dict) -> list[float]:
    vad = SileroVADAnalyzer(sample_rate=sr)
    vad.set_sample_rate(sr)
    still = StillTalkingV2(os.path.join(MODELS, "smart_turn_embed.onnx"),
                           os.path.join(MODELS, "still_talking_v2_head.onnx"), threads=2)
    an = FloorTurnAnalyzer(policy=FloorPolicy(**kw), still_talking=still)
    an.set_sample_rate(sr)
    n = vad.num_frames_required()
    ends, speaking = [], False
    for i in range(0, len(caller) - n, n):
        chunk = caller[i:i + n].tobytes()
        state = await vad.analyze_audio(chunk) if asyncio.iscoroutinefunction(vad.analyze_audio) else vad.analyze_audio(chunk)
        now = state in (VADState.SPEAKING, VADState.STOPPING)
        st = an.append_audio(chunk, now)
        if speaking and not now:  # VAD stop: Pipecat asks the analyzer once
            st, _ = await an.analyze_end_of_turn()
        elif st == EndOfTurnState.COMPLETE:
            st, _ = await an.analyze_end_of_turn()
        if st == EndOfTurnState.COMPLETE:
            ends.append(round(i / sr, 2))
            an.clear()
        speaking = now
    return ends


def main(path, start=0.0, end=1e9):
    with wave.open(path) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    caller = x[0::ch].copy() if ch == 2 else x
    for name, kw in SETTINGS.items():
        ends = asyncio.run(replay(caller, sr, kw))
        inside = [t for t in ends if start <= t <= end]
        print(f"{name:30} turn ends in {start:.0f}-{end:.0f}s: {len(inside):2d}  {inside}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(float(v) for v in a[1:3]))
