"""SpokenWOZ (real 8 kHz phone calls: people booking hotels, restaurants, taxis) -> bench file + caller audio.

Expects (data/spokenwoz, never committed; CC BY-NC 4.0, huggingface.co/datasets/ssz1111/SpokenWOZ-*):
  raw/data.json                   annotations: per turn "tag" user|system and per-word BeginTime/EndTime (ms)
  audio/audio_5700_test/<id>.wav  stereo 8 kHz: channel 0 = caller, channel 1 = agent

  python bench/recipes/spokenwoz.py data/spokenwoz

Writes
  wav16k/<id>.wav   the caller channel only (what a deployed agent hears), 8 -> 16 kHz with soxr as Pipecat does
  bench.json        caller speech runs from word times; end_of_turn where the agent takes a turn of >= 1 s next.
                    Same 0.2 s merge gap and turn rule as the FluencyBank and Buckeye recipes.
"""
import glob
import json
import os
import sys
import wave

import numpy as np
import soundfile as sf
import soxr

MERGE_GAP, MIN_TURN, MERGE_AGENT = 0.2, 1.0, 0.5


def merge(spans, gap):
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def record(did, log):
    user = [(w["BeginTime"] / 1000, w["EndTime"] / 1000) for t in log if t["tag"] == "user" for w in t["words"]
            if w["EndTime"] > w["BeginTime"]]
    agent = [(w["BeginTime"] / 1000, w["EndTime"] / 1000) for t in log if t["tag"] != "user" for w in t["words"]
             if w["EndTime"] > w["BeginTime"]]
    if not user:
        return None
    runs = [{"start": a, "end": b} for a, b in merge(user, MERGE_GAP)]
    agent_turns = [a for a, b in merge(agent, MERGE_AGENT) if b - a >= MIN_TURN]
    for k, run in enumerate(runs):
        nxt = runs[k + 1]["start"] if k + 1 < len(runs) else float("inf")
        if any(run["end"] - 0.5 <= s < nxt for s in agent_turns):
            run["end_of_turn"] = True
    runs[-1]["end_of_turn"] = True
    return {"id": did, "condition": "fluent_phone",
            "segments": [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in runs]}


def main(root):
    data = json.load(open(f"{root}/raw/data.json", encoding="utf8"))
    os.makedirs(f"{root}/wav16k", exist_ok=True)
    wavs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{root}/audio/**/*.wav", recursive=True)
            if "__MACOSX" not in p}  # the zip carries macOS metadata files named like the audio
    records = []
    for did, d in sorted(data.items()):
        if did not in wavs:
            continue
        out = f"{root}/wav16k/{did}.wav"
        if not os.path.exists(out):
            x, sr = sf.read(wavs[did], dtype="float32", always_2d=True)  # some files are 32-bit float wavs
            caller = soxr.resample(x[:, 0], sr, 16000)
            with wave.open(out, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes(np.clip(caller * 32768, -32768, 32767).astype(np.int16).tobytes())
        rec = record(did, d["log"])
        if rec:
            records.append(rec)
    json.dump(records, open(f"{root}/bench.json", "w"), indent=0)
    n_eot = sum(s.get("end_of_turn", False) for r in records for s in r["segments"])
    n_pause = sum(not s.get("end_of_turn") for r in records for s in r["segments"])
    print(f"{len(records)} calls, {n_pause} in-turn pauses, {n_eot} turn ends")


if __name__ == "__main__":
    main(sys.argv[1])
