"""Audio-track harness: replay annotated speech timelines through a system, score cut-offs and delay.

Annotation file (JSON list), one entry per recording:
  {"id": "f01", "condition": "stuttered" | "fluent",
   "segments": [{"start": 0.0, "end": 1.4, "pause_type": "block"},   # type of the gap AFTER this segment
                {"start": 2.6, "end": 3.9, "end_of_turn": true}]}
A gap after a segment without end_of_turn is a pause inside the turn; a cut-off is any response in it.
Recipes that build this format from real data live in bench/recipes.

Model outputs come in optional signal files, one reading per gap, computed by the recipes from audio:
  {"<id>": {"<segment index>": {"p_still_talking": 0.8, "still_talking_kind": "block", "p_finished": 0.1}}}
A gap's reading applies from SIGNAL_DELAY after speech stops (when the model would have run) to the next speech.

Usage: python -m turnheed.bench bench.json [more bench.json ...] [signals.json ...]
(bench files are JSON lists, signal files JSON objects; stuttered + fluent files together give the cut-off gap)
"""
from __future__ import annotations

import json
import statistics
import sys
from typing import Callable

from .policy import RESPOND, FloorPolicy, Signals

DT = 0.01          # 10 ms ticks
LATE_LIMIT = 2.0   # s after a true turn end with no reply = late/missed (late_rate = share of turn ends)
SIGNAL_DELAY = {"p_finished": 0.2}  # Smart Turn runs when VAD reports silence (Pipecat stop_secs 0.2)

# A system gets one Signals per tick and returns True to respond.
System = Callable[[Signals], bool]


def timeout_baseline(ms: int) -> Callable[[], System]:
    def make():
        state = {"silence_start": None, "fired": False}

        def step(s: Signals) -> bool:
            if s.user_speaking:
                state["silence_start"], state["fired"] = None, False
                return False
            if state["silence_start"] is None:
                state["silence_start"] = s.t
            if not state["fired"] and s.t - state["silence_start"] >= ms / 1000:
                state["fired"] = True
                return True
            return False
        return step
    return make


def smart_turn_pipecat(stop_secs: float = 3.0) -> Callable[[], System]:
    """Pipecat's default: answer when Smart Turn says complete, else after stop_secs of silence."""
    def make():
        state = {"silence_start": None, "fired": False}

        def step(s: Signals) -> bool:
            if s.user_speaking:
                state["silence_start"], state["fired"] = None, False
                return False
            if state["silence_start"] is None:
                state["silence_start"] = s.t
            done = (s.p_finished is not None and s.p_finished >= 0.5) or s.t - state["silence_start"] >= stop_secs
            if done and not state["fired"]:
                state["fired"] = True
                return True
            return False
        return step
    make.uses = ("p_finished",)
    return make


def policy_system(oracle_still_talking: bool = False, uses: tuple = (), **kw) -> Callable[[], System]:
    """FloorPolicy fed the signal-file fields named in `uses`. With oracle_still_talking the hold signal
    comes from the annotation itself: an upper bound for a perfect "still talking" model, not a result."""
    def make():
        pol = FloorPolicy(**kw)
        return lambda s: pol.step(s).decision == RESPOND
    make.oracle, make.uses = oracle_still_talking, uses
    return make


def _ticks(rec: dict, oracle: bool, sig: dict):
    segs = rec["segments"]
    end = segs[-1]["end"] + LATE_LIMIT + 0.5
    t, i = 0.0, 0
    while t < end:
        while i < len(segs) and t >= segs[i]["end"]:
            i += 1
        speaking = i < len(segs) and segs[i]["start"] <= t < segs[i]["end"]
        prev = segs[i - 1] if i > 0 else None
        in_pause = not speaking and prev is not None and not prev.get("end_of_turn") and i < len(segs)
        kind = prev.get("pause_type") if in_pause else None
        f = {}
        if not speaking and prev is not None:
            f = {k: v for k, v in sig.get(str(i - 1), {}).items() if t - prev["end"] >= SIGNAL_DELAY.get(k, 0)}
        if oracle:
            f["p_still_talking"] = 0.99 if kind and kind != "natural" else 0.01
            f["still_talking_kind"] = kind
        yield Signals(t=round(t, 3), user_speaking=speaking, **f)
        t += DT


def score(rec: dict, make: Callable[[], System], signals: dict | None = None) -> dict:
    """Responses per gap -> cut-offs, delays, late replies, split turns."""
    uses = getattr(make, "uses", ())
    sig = {g: {k: v for k, v in f.items() if k in uses}
           for g, f in (signals or {}).get(rec["id"], {}).items()}
    sys_ = make()
    responses = [s.t for s in _ticks(rec, getattr(make, "oracle", False), sig) if sys_(s)]
    segs = rec["segments"]
    pauses = cutoffs = late = 0
    delays, split_turns, turn_cut = [], 0, False
    for k, seg in enumerate(segs):
        gap_end = segs[k + 1]["start"] if k + 1 < len(segs) else seg["end"] + LATE_LIMIT
        hit = [r for r in responses if seg["end"] <= r < gap_end]
        if seg.get("end_of_turn"):
            if hit and hit[0] - seg["end"] <= LATE_LIMIT:
                delays.append(hit[0] - seg["end"])
            else:
                late += 1
            split_turns += turn_cut
            turn_cut = False
        else:
            pauses += 1
            if hit:
                cutoffs += 1
                turn_cut = True
    return dict(pauses=pauses, cutoffs=cutoffs, delays=delays, late=late, split_turns=split_turns)


def run(records: list[dict], systems: dict[str, Callable[[], System]], signals: dict | None = None) -> list[dict]:
    rows = []
    for name, make in systems.items():
        by_cond: dict[str, dict] = {}
        for rec in records:
            r = score(rec, make, signals)
            agg = by_cond.setdefault(rec["condition"], dict(pauses=0, cutoffs=0, delays=[], late=0, split_turns=0))
            for k, v in r.items():
                agg[k] += v
        row = {"system": name}
        for cond, a in by_cond.items():
            row[f"cutoff_rate_{cond}"] = a["cutoffs"] / a["pauses"] if a["pauses"] else None
        if row.get("cutoff_rate_fluent") and row.get("cutoff_rate_stuttered") is not None:
            row["cutoff_gap"] = row["cutoff_rate_stuttered"] / row["cutoff_rate_fluent"]
        for cond, a in by_cond.items():
            row[f"delay_median_ms_{cond}"] = 1000 * statistics.median(a["delays"]) if a["delays"] else None
            ends = len(a["delays"]) + a["late"]
            row[f"late_rate_{cond}"] = a["late"] / ends if ends else None
        row["split_turns"] = sum(a["split_turns"] for a in by_cond.values())
        rows.append(row)
    return rows


DEFAULT_SYSTEMS = {
    "timeout_300ms": timeout_baseline(300),
    "timeout_500ms": timeout_baseline(500),
    "timeout_800ms": timeout_baseline(800),
    "timeout_1200ms": timeout_baseline(1200),
    "policy_audio_only": policy_system(False),
    "policy_oracle_hold (ceiling)": policy_system(True),
}

# need signal files; dropped when a field they use is missing
MODEL_SYSTEMS = {
    "smart_turn (pipecat default)": smart_turn_pipecat(),
    "policy + still_talking": policy_system(uses=("p_still_talking", "still_talking_kind")),
    "policy + smart_turn": policy_system(uses=("p_finished",)),
    "policy + smart_turn + still_talking": policy_system(uses=("p_finished", "p_still_talking", "still_talking_kind")),
}


def merge_signals(paths: list[str]) -> dict:
    out: dict = {}
    for p in paths:
        for rid, gaps in json.load(open(p, encoding="utf8")).items():
            for g, f in gaps.items():
                out.setdefault(rid, {}).setdefault(g, {}).update(f)
    return out


def _fmt(v):
    return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def main(argv: list[str]) -> None:
    files = [json.load(open(p, encoding="utf8")) for p in argv]
    records = [r for f in files if isinstance(f, list) for r in f]   # bench files are lists
    signals = merge_signals([p for p, f in zip(argv, files) if isinstance(f, dict)])
    have = {k for gaps in signals.values() for f in gaps.values() for k in f}
    systems = dict(DEFAULT_SYSTEMS)
    systems.update({n: m for n, m in MODEL_SYSTEMS.items() if all(u in have for u in m.uses if u.startswith("p_"))})
    rows = run(records, systems, signals)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(_fmt(r.get(c)) for c in cols))


if __name__ == "__main__":
    main(sys.argv[1:])
