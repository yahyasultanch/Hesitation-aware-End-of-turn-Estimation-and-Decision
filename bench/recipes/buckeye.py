"""Buckeye Corpus (fluent interviews) -> 16 kHz audio and a bench file, built like the FluencyBank one.

Expects data/buckeye/zips/s01.zip .. s40.zip from buckeyecorpus.osu.edu (licence: register, research use only).

  python bench/recipes/buckeye.py data/buckeye

Writes
  wav16k/<session>.wav   e.g. s0101a.wav (already 16 kHz mono in the corpus)
  bench.json             one record per session: talker speech runs from the hand-corrected word labels,
                         end_of_turn where the interviewer (<IVER>) takes a turn (>= MIN_TURN) next. Same 0.2 s merge gap and
                         same turn-end rule as bench/recipes/fluencybank.py, so the two are comparable.

.words lines: "<end time> 122 <label>; <citation phones>; <actual phones>; <POS>". A label spans from the
previous line's time to its own.
"""
import io
import json
import os
import sys
import zipfile

MERGE_GAP = 0.2
MIN_TURN = 1.0    # interviewer speech shorter than this ("mhm", "yeah") is a backchannel, not a turn
MERGE_INV = 0.5   # interviewer stretches closer than this are one turn
NONSPEECH = {"<SIL>", "<NOISE>", "<VOCNOISE>", "<LAUGH>", "<UNKNOWN>"}


def is_speech(label: str) -> bool:
    if label.startswith(("{", "<IVER", "<VOICE", "<CONF", "<SIL")) or label in NONSPEECH:
        return False
    return True  # words, plus <CUTOFF-..>, <HES-..>, <EXT-..>, <ERROR-..>, <LAUGH-word>, <EXCLUDE-name>


def parse_words(text: str):
    """-> list of (start, end, label)"""
    out, prev, body = [], 0.0, False
    for line in text.splitlines():
        if not body:
            body = line.strip() == "#"
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        end, label = float(parts[0]), parts[2].split(";")[0].strip()
        out.append((prev, end, label))
        prev = end
    return out


def merge_spans(spans):
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] < MERGE_INV:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def pause_type(before: str, after: str) -> str:
    if after.startswith("<CUTOFF") or before.startswith("<CUTOFF"):
        return "repetition"
    if after.startswith(("<EXT", "<HES")):
        return "prolongation"
    if after.lower() in ("um", "uh") or before.lower() in ("um", "uh"):
        return "word_search"
    return "pause"


def session_record(sid: str, words) -> dict | None:
    speech = [(a, b, l) for a, b, l in words if is_speech(l) and b > a]
    inv = [a for a, b in merge_spans((a, b) for a, b, l in words
                                     if l.startswith("<IVER") and not l.startswith("<IVER_overlap"))
           if b - a >= MIN_TURN]
    if not speech:
        return None
    runs = [{"start": speech[0][0], "end": speech[0][1], "_last": speech[0][2]}]
    for a, b, l in speech[1:]:
        if a - runs[-1]["end"] < MERGE_GAP:
            runs[-1]["end"], runs[-1]["_last"] = max(runs[-1]["end"], b), l
        else:
            runs[-1]["pause_type"] = pause_type(runs[-1]["_last"], l)
            runs.append({"start": a, "end": b, "_last": l})
    for k, run in enumerate(runs):
        nxt = runs[k + 1]["start"] if k + 1 < len(runs) else float("inf")
        if any(run["end"] - 0.5 <= s < nxt for s in inv):
            run["end_of_turn"] = True
            run.pop("pause_type", None)
    runs[-1]["end_of_turn"] = True
    runs[-1].pop("pause_type", None)
    return {"id": sid, "condition": "fluent",
            "segments": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                          if not k.startswith("_")} for r in runs]}


def main(root: str) -> None:
    os.makedirs(f"{root}/wav16k", exist_ok=True)
    records = []
    for n in range(1, 41):
        outer = zipfile.ZipFile(f"{root}/zips/s{n:02d}.zip")
        for name in sorted(outer.namelist()):
            if not name.endswith(".zip"):
                continue
            inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
            files = {os.path.basename(f): f for f in inner.namelist()}
            sid = os.path.basename(name)[:-4]
            if f"{sid}.words" not in files or f"{sid}.wav" not in files:
                print("skip", sid, sorted(files))
                continue
            wav = f"{root}/wav16k/{sid}.wav"
            if not os.path.exists(wav):
                open(wav, "wb").write(inner.read(files[f"{sid}.wav"]))
            rec = session_record(sid, parse_words(inner.read(files[f"{sid}.words"]).decode("latin-1")))
            if rec:
                records.append(rec)
    json.dump(records, open(f"{root}/bench.json", "w"), indent=0)
    n_eot = sum(s.get("end_of_turn", False) for r in records for s in r["segments"])
    n_pause = sum(not s.get("end_of_turn") for r in records for s in r["segments"])
    print(f"{len(records)} sessions, {n_pause} in-turn pauses, {n_eot} turn ends")


if __name__ == "__main__":
    main(sys.argv[1])
