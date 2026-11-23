"""Hand-check the turn-end labels: are labelled turn ends real ends, or moments where the other person (agent,
interviewer) cut the speaker off? Those count against the words check today though the check is right.

  python bench/recipes/label_audit.py export     # data/label_audit/sheet.csv to label, key.csv kept apart
  python bench/recipes/label_audit.py score      # once sheet.csv's "label" column is filled in

Sample per set: 60 turn ends the words check (Haiku) called mid-sentence + 40 it called complete, shuffled, with
the model verdicts kept out of the sheet so they don't steer the labeller. Labels:
  finished (a real turn end)   cut off (still mid-turn: the label is wrong)   unclear (can't tell)
"""
import csv
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import completion_veto_eval as ev  # noqa: E402

OUT = "data/label_audit"
SETS = {"phone": "data/spokenwoz", "stuttered": "data/fluencybank"}
N_FLAGGED, N_OTHER = 60, 40


def next_lines_spokenwoz(root):
    """(call, caller turn end time) -> what the agent said next and when it started."""
    data = json.load(open(f"{root}/raw/data.json", encoding="utf8"))
    out = {}
    for did, d in data.items():
        agent = [(t["words"][0]["BeginTime"] / 1000, t["text"]) for t in d["log"] if t["tag"] != "user" and t["words"]]
        out[did] = agent
    return out


def next_lines_fluencybank(root):
    out = {}
    for f in os.listdir(f"{root}/Voices-AWS/interview"):
        iid, lines = f[:-4], []
        for line in open(f"{root}/Voices-AWS/interview/{f}", encoding="utf8"):
            m = re.search(r"\x15(\d+)_(\d+)\x15", line)
            if line.startswith("*INV:") and m:
                text = re.sub(r"\x15.*?\x15|\[.*?\]|&[-=+]?\S+|[+<>/()]|\bxxx\b", " ", line[5:])
                lines.append((int(m.group(1)) / 1000, " ".join(text.split())))
        out[iid] = lines
    return out


def sample(name, root):
    texts = ev.turn_texts_fluencybank(root) if name == "stuttered" else ev.turn_texts(root)
    recs = {r["id"]: r for r in json.load(open(f"{root}/bench.json", encoding="utf8"))}
    nxt = next_lines_fluencybank(root) if name == "stuttered" else next_lines_spokenwoz(root)
    hk = json.load(open(f"{root}/completion_judgements.json", encoding="utf8"))
    qw_path = f"{root}/completion_judgements_ollama_qwen3_8b.json"
    qw = json.load(open(qw_path, encoding="utf8")) if os.path.exists(qw_path) else {}
    ends = [k for k, v in texts.items() if v[2] and v[3] < 90 and f"{k[0]}|{k[1]}" in hk]
    flagged = [k for k in ends if hk[f"{k[0]}|{k[1]}"] in ("○", "◐")]
    other = [k for k in ends if k not in set(flagged)]
    rng = random.Random(1)
    rows = []
    for group, pool, n in (("flagged", flagged, N_FLAGGED), ("other", other, N_OTHER)):
        for d, g in rng.sample(pool, min(n, len(pool))):
            agent_before, caller, _, gap = texts[(d, g)]
            end = recs[d]["segments"][g]["end"]
            after = next(((t, x) for t, x in nxt.get(d, []) if t >= end - 1.0), (None, ""))
            q = qw.get(f"{d}|{g}")
            rows.append({"set": name, "id": f"{d}|{g}", "group": group, "haiku": hk[f"{d}|{g}"],
                         "qwen_p": q[1] if isinstance(q, list) else "",
                         "other_starts_s": "" if after[0] is None else round(after[0] - end, 2),
                         "caller_resumes_s": round(gap, 2),
                         "other_before": agent_before, "caller_turn": caller, "other_next": after[1]})
    return rows


def export():
    os.makedirs(OUT, exist_ok=True)
    rows = [r for name, root in SETS.items() for r in sample(name, root)]
    random.Random(2).shuffle(rows)
    for i, r in enumerate(rows):
        r["n"] = i + 1
    sheet_cols = ["n", "set", "other_starts_s", "other_before", "caller_turn", "other_next", "label"]
    with open(f"{OUT}/sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, sheet_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({**r, "label": ""} for r in rows)
    with open(f"{OUT}/key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["n", "set", "id", "group", "haiku", "qwen_p", "caller_resumes_s"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} turn ends -> {OUT}/sheet.csv (label column: finished / cut off / unclear); verdicts in {OUT}/key.csv")


def score():
    sheet = {r["n"]: r for r in csv.DictReader(open(f"{OUT}/sheet.csv", encoding="utf-8-sig"))}
    key = list(csv.DictReader(open(f"{OUT}/key.csv", encoding="utf-8-sig")))
    for name in SETS:
        for group in ("flagged", "other"):
            ls = [sheet[k["n"]]["label"].strip().lower()[:1] for k in key if k["set"] == name and k["group"] == group]
            ls = [x for x in ls if x]
            if ls:
                c = np.mean([x == "c" for x in ls])
                print(f"{name:9} {group:7} labelled {len(ls):3}: cut off / mid-turn {c*100:4.0f}%  "
                      f"finished {np.mean([x == 'f' for x in ls])*100:4.0f}%  unclear {np.mean([x == 'u' for x in ls])*100:4.0f}%")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
