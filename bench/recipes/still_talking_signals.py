"""Score every gap of a bench set with a trained still-talking model (for sets it never trained on).

  python bench/recipes/still_talking_signals.py data/buckeye data/still_talking/still_talking.onnx

Expects <root>/bench.json and <root>/wav16k/<id>.wav. Writes <root>/still_talking_signals.json.
Same window as training: the 3 s of audio up to the moment the speech run ends.
"""
import json
import sys
import wave

import numpy as np

sys.path.insert(0, __file__.rsplit("bench", 1)[0])
from turnheed.still_talking import SR, StillTalking  # noqa: E402


def main(root, model):
    st, out = StillTalking(model, threads=4), {}
    for rec in json.load(open(f"{root}/bench.json", encoding="utf8")):
        with wave.open(f"{root}/wav16k/{rec['id']}.wav") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        for g, s in enumerate(rec["segments"]):
            p, kind = st.predict(x[:int(s["end"] * SR)])
            out.setdefault(rec["id"], {})[str(g)] = {"p_still_talking": round(p, 4), "still_talking_kind": kind}
    json.dump(out, open(f"{root}/still_talking_signals.json", "w"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
