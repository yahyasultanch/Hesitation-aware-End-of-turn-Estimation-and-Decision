"""Score a bench set the still-talking v2 head never trained on (e.g. SpokenWOZ), from its Smart Turn features.

  python bench/recipes/still_talking_v2_score.py data/spokenwoz     # after smart_turn_embed.py on that set

Writes <root>/still_talking_v2_signals.json with the head in data/still_talking/still_talking_v2_head.onnx.
"""
import json
import sys

import numpy as np
import onnxruntime as ort


def main(root, head="data/still_talking/still_talking_v2_head.onnx"):
    d = np.load(f"{root}/smart_turn_embed.npz")
    p = ort.InferenceSession(head).run(None, {"emb": d["emb"].astype(np.float32)})[0]
    out = {}
    for i, g, q in zip(d["id"], d["gap"], p):
        out.setdefault(str(i), {})[str(int(g))] = {"p_still_talking": round(float(q), 4)}
    json.dump(out, open(f"{root}/still_talking_v2_signals.json", "w"))
    print(f"scored {len(p)} gaps")


if __name__ == "__main__":
    main(*sys.argv[1:3])
