"""Run Pipecat's bundled Smart Turn v3.2 on every FluencyBank gap, the way Pipecat does: once, on the turn
audio up to the moment VAD reports silence (STOP_SECS after speech ends), last 8 s.

  python bench/recipes/smart_turn_signals.py data/fluencybank     # needs the pipecat extra

Writes data/fluencybank/smart_turn_signals.json  {interview: {gap index: {p_finished}}} for the bench.
"""
import json
import sys
import wave

import numpy as np
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

STOP_SECS = 0.2  # Pipecat's default VAD stop_secs: Smart Turn sees this much trailing silence
SR = 16000


def main(root):
    st = LocalSmartTurnAnalyzerV3(cpu_count=4)
    out = {}
    for rec in json.load(open(f"{root}/bench.json", encoding="utf8")):
        with wave.open(f"{root}/wav16k/{rec['id']}.wav") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        for g, s in enumerate(rec["segments"]):
            end = int((s["end"] + STOP_SECS) * SR)
            p = st._predict_endpoint(x[max(0, end - 8 * SR):end])["probability"]
            out.setdefault(rec["id"], {})[str(g)] = {"p_finished": round(p, 4)}
        print(rec["id"], flush=True)
    json.dump(out, open(f"{root}/smart_turn_signals.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
