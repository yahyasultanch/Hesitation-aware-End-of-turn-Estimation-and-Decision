"""Cut-off / lateness / reply-time points for every system, from the cached signals and judgements (no API calls).

  python bench/recipes/tradeoff_curves.py      # writes data/tradeoff_<set>.csv

Series: fixed timeouts 0.4-2.4 s, Pipecat's default (Smart Turn), the layer with sound only, the layer with the
words check (Haiku v1, Haiku v2 where judged, Qwen3 8B v2 at a sweep of P(complete) cut-points).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
os.chdir(os.path.join(HERE, "..", ".."))
import completion_veto_eval as ev  # noqa: E402
from turnheed.wait import LearnedWait  # noqa: E402

RECHECK, LATE = 2.5, 2.0


def rates(reply, eot, gap):
    """-> cut-offs %, late % (reply > 2 s or the speaker resumed first), median and mean reply at turn ends (s)"""
    heard = eot & (gap > reply)
    return (100 * np.mean(gap[~eot] > reply[~eot]),
            100 * np.mean((reply[eot] > LATE) | (gap[eot] <= reply[eot])),
            float(np.median(reply[heard])), float(np.mean(reply[heard])))


def main():
    for name, root, n, wm in (("stuttered", "data/fluencybank", 2000, "wait_model.json"),
                              ("phone", "data/spokenwoz", None, "wait_model_phone.json")):
        texts, pauses, ends = ev.eval_keys(root, n)
        keys = pauses + ends
        st = json.load(open(f"{root}/smart_turn_signals.json", encoding="utf8"))
        v2 = json.load(open(f"{root}/still_talking_v2_signals.json", encoding="utf8"))
        eot = np.array([texts[k][2] for k in keys])
        gap = np.array([texts[k][3] for k in keys])
        pf = np.array([st[d][str(g)]["p_finished"] for d, g in keys])
        ps = np.array([v2[d][str(g)]["p_still_talking"] for d, g in keys])
        W = np.array([LearnedWait.load(f"data/still_talking/{wm}", theta=0.5)(a, b) for a, b in zip(pf, ps)])

        def done_from(path, cut=None):
            c = json.load(open(path, encoding="utf8"))
            v = [c.get(f"{d}|{g}") for d, g in keys]
            if cut is None:
                return np.array([(x[0] if isinstance(x, list) else x) not in ("○", "◐") for x in v])
            return np.array([(x[1] if isinstance(x, list) and x[1] is not None else 1.0) >= cut for x in v])

        rows = [("timeout", *rates(np.full(len(keys), t), eot, gap), f"{t:.1f} s")
                for t in np.arange(0.4, 2.41, 0.2)]
        rows.append(("pipecat", *rates(np.where(pf >= 0.5, 0.2, 3.0), eot, gap), "Pipecat default"))
        rows.append(("sound", *rates(W, eot, gap), "layer, sound only"))
        for tag, label in (("", "Haiku v1"), ("_v2", "Haiku v2")):
            path = f"{root}/completion_judgements{tag}.json"
            c = json.load(open(path, encoding="utf8")) if os.path.exists(path) else {}
            if all(f"{d}|{g}" in c for d, g in keys):
                rows.append(("haiku" + tag, *rates(np.where(done_from(path), W, W + RECHECK), eot, gap), label))
        qp = f"{root}/completion_judgements_ollama_qwen3_8b_v2.json"
        for cut in (1e-4, 1e-3, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7):
            rows.append(("qwen", *rates(np.where(done_from(qp, cut), W, W + RECHECK), eot, gap), f"P>={cut:g}"))
        with open(f"data/tradeoff_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["series", "late", "cut", "median", "mean", "label"])
            for s, cut, late, med, mean, label in rows:
                w.writerow([s, f"{late:.2f}", f"{cut:.2f}", f"{med:.3f}", f"{mean:.3f}", label])
        print(name, len(rows), "points")


if __name__ == "__main__":
    main()
