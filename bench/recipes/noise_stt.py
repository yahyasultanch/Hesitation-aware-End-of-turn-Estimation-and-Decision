"""What does background noise do to the words? Transcribe the caller's turn (up to the pause) with Deepgram, clean
and with ESC-50 noise at 5 dB, both through a simulated phone line, and judge both transcripts with the agent's
own words check (Haiku, the live instructions).

  python bench/recipes/noise_stt.py data/fluencybank data/spokenwoz [n per set, default 250]
  (~$0.005 Deepgram + ~$0.0003 Haiku per gap and condition; everything cached in data/noise_stt/)

Reports how much the transcript changes (word error rate of noisy vs clean transcript), how often noise empties
it, and how often the words check's verdict flips, split by real mid-turn pauses and real turn ends.
"""
import asyncio
import json
import os
import random
import sys
import urllib.request
import wave
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import completion_veto_eval as ev  # noqa: E402
from noise_bench import active_rms, load_noise, noise_bed  # noqa: E402
from smart_turn_embed import SR, phone  # noqa: E402

LEVEL = 5  # dB: the loudest level of noise_bench
OUT = "data/noise_stt"


def deepgram(x):
    """Prerecorded Deepgram nova-3 (the live agent streams nova-3-general) on 8 kHz mu-law-like phone audio."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    req = urllib.request.Request("https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en",
                                 data=buf.getvalue(), method="POST",
                                 headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                                          "Content-Type": "audio/wav"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    alts = (r["results"]["channels"] or [{}])[0].get("alternatives") or [{}]
    return alts[0].get("transcript", "")  # no alternatives: nothing heard


def wer(ref, hyp):
    a, b = ref.lower().split(), hyp.lower().split()
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(a))


def turn_audio(root, recs, d, g):
    """The caller's turn so far: from the first speech after their previous turn end to 0.2 s into this pause."""
    segs = recs[d]["segments"]
    s0 = g
    while s0 > 0 and not segs[s0 - 1].get("end_of_turn"):
        s0 -= 1
    with wave.open(f"{root}/wav16k/{d}.wav") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    lo, hi = int(segs[s0]["start"] * SR), int((segs[g]["end"] + 0.2) * SR)
    return x[max(lo, hi - 20 * SR):hi]


async def judge_all(items):
    import anthropic
    client, sem = anthropic.AsyncAnthropic(), asyncio.Semaphore(8)
    return await asyncio.gather(*(ev.judge(client, sem, a, c) for a, c in items))


def main(roots, n=250):
    load = __import__("dotenv").load_dotenv
    load(".env")
    os.makedirs(OUT, exist_ok=True)
    clips = load_noise()
    for root in roots:
        name = os.path.basename(root)
        path = f"{OUT}/{name}.json"
        cache = json.load(open(path, encoding="utf8")) if os.path.exists(path) else {}
        texts, pauses, ends = ev.eval_keys(root, 2000 if name == "fluencybank" else None)
        rng = random.Random(3)
        keys = rng.sample(pauses, n // 2) + rng.sample(ends, n - n // 2)
        recs = {r["id"]: r for r in json.load(open(f"{root}/bench.json", encoding="utf8"))}
        secs = 0.0
        for i, (d, g) in enumerate(keys):
            k = f"{d}|{g}"
            if k in cache and "snr" in cache[k]:
                continue
            x = turn_audio(root, recs, d, g)
            bed = noise_bed(clips, len(x), np.random.default_rng(zlib.crc32(k.encode())))
            noisy = x + bed * active_rms(x) / 10 ** (LEVEL / 20)
            cache[k] = {"eot": texts[(d, g)][2], "agent": texts[(d, g)][0],
                        "clean": deepgram(phone(x)), "snr": deepgram(phone(noisy))}
            secs += 2 * len(x) / SR
            if i % 25 == 0:
                json.dump(cache, open(path, "w", encoding="utf8"))
                print(f"{name}: {i + 1}/{len(keys)} transcribed ({secs / 60:.0f} min of audio so far)", flush=True)
        todo = [(k, c) for k in (f"{d}|{g}" for d, g in keys) for c in ("clean", "snr")
                if f"v_{c}" not in cache[k]]
        res = asyncio.run(judge_all([(cache[k]["agent"], cache[k][c]) for k, c in todo]))
        for (k, c), (m, _, _) in zip(todo, res):
            cache[k][f"v_{c}"] = m
        json.dump(cache, open(path, "w", encoding="utf8"))

        rows = [cache[f"{d}|{g}"] for d, g in keys]
        done = lambda m: m not in ("○", "◐")  # noqa: E731
        print(f"\n{name} ({len(rows)} gaps, noise {LEVEL} dB, phone line):")
        print(f"  transcript change, noisy vs clean: word error rate {np.mean([wer(r['clean'], r['snr']) for r in rows]) * 100:.1f}%;"
              f" emptied by noise {np.mean([bool(r['clean'].strip()) and not r['snr'].strip() for r in rows]) * 100:.1f}%")
        for label, sel in (("mid-turn pauses", [r for r in rows if not r["eot"]]), ("turn ends", [r for r in rows if r["eot"]])):
            cc = np.mean([done(r["v_clean"]) for r in sel]) * 100
            cn = np.mean([done(r["v_snr"]) for r in sel]) * 100
            flip = np.mean([done(r["v_clean"]) != done(r["v_snr"]) for r in sel]) * 100
            print(f"  {label:16} ({len(sel)}): words check says complete {cc:5.1f}% clean, {cn:5.1f}% noisy; "
                  f"verdict flips {flip:5.1f}%")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(args, *nums)
