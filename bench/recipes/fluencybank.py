"""FluencyBank Timestamped + Voices-AWS transcripts -> 16 kHz audio and a bench file.

Expects (all under data/fluencybank, never committed):
  TimeStamped/metadata.csv, TimeStamped/csvs/*.csv   Romana et al. 2024 (TalkBank derived corpora)
  Voices-AWS/interview/*.cha                         Voices-AWS transcripts (INV/PAR turns, ms bullets)
  media/*.mp4                                        Voices-AWS interview media

  python bench/recipes/fluencybank.py data/fluencybank

Writes:
  wav16k/<id>.wav            full interview, 16 kHz mono
  TimeStamped/wav16k/<seg>.wav  segment audio, 50 ms silence each side so csv word times line up (per README)
  bench.json                 one record per interview: participant speech runs from word timings,
                             end_of_turn where the interviewer (INV) takes a turn (>= MIN_TURN) next.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np

SR = 16000
MERGE_GAP = 0.2   # word gaps shorter than this are not pauses a system could act on
PAD = 0.05        # README: 50 ms silence added to each segment
MIN_TURN = 1.0    # interviewer speech shorter than this ("mhm", "yeah") is a backchannel, not a turn
MERGE_INV = 0.5   # interviewer utterances closer than this are one turn


def to_wav(mp4: str, out: str) -> None:
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", mp4, "-vn",
                        "-ac", "1", "-ar", str(SR), out + ".tmp.wav"], check=True)
        os.replace(out + ".tmp.wav", out)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(x.astype(np.int16).tobytes())


def inv_turns(cha: str) -> list[float]:
    """Start times (s) of interviewer turns; CHAT bullets are start_end in ms.
    Utterances < MERGE_INV apart are one turn; turns shorter than MIN_TURN are backchannels, not turns."""
    spans = []
    for line in open(cha, encoding="utf8"):
        if line.startswith("*INV:"):
            m = re.search(r"(\d+)_(\d+)", line)
            if m:
                spans.append((int(m.group(1)) / 1000, int(m.group(2)) / 1000))
    return [a for a, b in merge_spans(spans) if b - a >= MIN_TURN]


def merge_spans(spans):
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] < MERGE_INV:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def pause_type(before: dict, after: dict) -> str:
    if after["rp"] == "1" or after["pw"] == "1" or before["pw"] == "1":
        return "repetition"
    if after["fp"] == "1" or before["fp"] == "1":
        return "word_search"
    return "pause"  # unlabeled: may hide a silent block; Timestamped has no block labels


def main(root: str) -> None:
    ts = f"{root}/TimeStamped"
    segs: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(f"{ts}/metadata.csv", encoding="utf8")):
        segs.setdefault(r["segid"].rsplit("_", 1)[0], []).append(r)

    os.makedirs(f"{root}/wav16k", exist_ok=True)
    os.makedirs(f"{ts}/wav16k", exist_ok=True)
    records, skipped = [], []
    for iid, rows in sorted(segs.items()):
        mp4, cha = f"{root}/media/{iid}.mp4", f"{root}/Voices-AWS/interview/{iid}.cha"
        if not (os.path.exists(mp4) and os.path.exists(cha)):
            skipped.append(iid)
            continue
        full = f"{root}/wav16k/{iid}.wav"
        to_wav(mp4, full)
        audio = read_wav(full)
        pad = np.zeros(int(PAD * SR), dtype=np.int16)

        words = []  # absolute times
        for r in rows:
            a, b = int(float(r["segstart"]) * SR), int(float(r["segend"]) * SR)
            out = f"{ts}/wav16k/{r['segid']}.wav"
            if not os.path.exists(out):
                write_wav(out, np.concatenate([pad, audio[a:b], pad]))
            off = float(r["segstart"]) - PAD
            for w in csv.DictReader(open(f"{ts}/csvs/{r['segid']}.csv", encoding="utf8")):
                w["start"], w["end"] = off + float(w["wordstart"]), off + float(w["wordend"])
                words.append(w)
        words.sort(key=lambda w: w["start"])

        # merge words into speech runs; remember the words either side of each gap
        runs = [{"start": words[0]["start"], "end": words[0]["end"], "_last": words[0]}]
        for w in words[1:]:
            if w["start"] - runs[-1]["end"] < MERGE_GAP:
                runs[-1]["end"], runs[-1]["_last"] = max(runs[-1]["end"], w["end"]), w
            else:
                runs[-1]["pause_type"] = pause_type(runs[-1]["_last"], w)
                runs.append({"start": w["start"], "end": w["end"], "_last": w})

        # a run is a turn end if the interviewer starts speaking before the next run does
        inv = inv_turns(cha)
        for k, run in enumerate(runs):
            nxt = runs[k + 1]["start"] if k + 1 < len(runs) else float("inf")
            if any(run["end"] - 0.5 <= s < nxt for s in inv):
                run["end_of_turn"] = True
                run.pop("pause_type", None)
        runs[-1]["end_of_turn"] = True
        records.append({"id": iid, "condition": "stuttered",
                        "segments": [{k: (round(v, 3) if isinstance(v, float) else v)
                                      for k, v in r.items() if not k.startswith("_")} for r in runs]})

    json.dump(records, open(f"{root}/bench.json", "w"), indent=0)
    n_eot = sum(s.get("end_of_turn", False) for r in records for s in r["segments"])
    n_pause = sum(not s.get("end_of_turn") for r in records for s in r["segments"])
    print(f"{len(records)} interviews, {n_pause} in-turn pauses, {n_eot} turn ends; skipped {skipped}")


if __name__ == "__main__":
    main(sys.argv[1])
