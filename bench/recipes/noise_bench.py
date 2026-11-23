"""How much does background noise hurt the sound side of turn-taking? Mix ESC-50 environmental noise into every
gap the words-check eval scores, at several loudness levels, and recompute what the live agent hears:

- Smart Turn v3.2 P(finished) (8 s up to 0.2 s into the pause, as Pipecat runs it)
- still-talking v2 P(mid-turn) (8 s up to the end of speech). FluencyBank speakers are scored by the
  cross-validation head that never saw them, as in the published numbers; SpokenWOZ by the full head (unseen set)
- Silero VAD (Pipecat's analyzer and defaults) on 1.5 s of the noise alone at that level: does it hear "speech"
  in an otherwise silent pause?
  The agent's silence clock restarts whenever it does, so this is what delays replies on a noisy line.

  python bench/recipes/noise_bench.py data/fluencybank 2000     # same gaps as completion_veto_eval.py
  python bench/recipes/noise_bench.py data/spokenwoz
  SIG=_snr10 python bench/recipes/completion_veto_eval.py data/fluencybank 2000    # turn-taking with that noise

Noise: ESC-50 (K. Piczak, CC BY-NC 3.0; unpack to data/ESC-50-master or set ESC50=<path>), all classes except crying baby and laughing (voice-like). Each gap gets
its own random clips (seeded, the same at every level), looped over the window and the pause; level = speech RMS /
noise RMS over their active 20 ms frames. Writes <root>/{smart_turn,still_talking_v2}_signals_snr<N>.json and
<root>/vad_pause_signals.json.
"""
import asyncio
import csv
import json
import os
import sys
import wave
import zlib

import numpy as np
import onnxruntime as ort
import soxr
import torch
from transformers import WhisperFeatureExtractor

sys.path.insert(0, os.path.dirname(__file__))
import completion_veto_eval as ev  # noqa: E402
from smart_turn_embed import SR, model_with_embedding  # noqa: E402

ESC = os.getenv("ESC50", "data/ESC-50-master")  # github.com/karolpiczak/ESC-50, unpacked
LEVELS = (None, 20, 10, 5)  # dB speech-to-noise; None = clean (checks the pipeline reproduces the published signals)
VOICE_LIKE = {"crying_baby", "laughing"}
PAUSE_VAD = 1.5  # s of pause the VAD check listens to


def active_rms(x):
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = (f ** 2).mean(1)
    return float(np.sqrt(e[e > 0.1 * e.max()].mean())) if len(e) and e.max() > 0 else 0.0


def load_noise():
    clips = []
    for r in csv.DictReader(open(f"{ESC}/meta/esc50.csv", encoding="utf8")):
        if r["category"] in VOICE_LIKE:
            continue
        with wave.open(f"{ESC}/audio/{r['filename']}") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            x = soxr.resample(x, w.getframerate(), SR).astype(np.float32)
        if active_rms(x) > 1e-4:
            clips.append(x / active_rms(x))
    return clips


def noise_bed(clips, n, rng):
    out = np.concatenate([clips[i] for i in rng.integers(0, len(clips), n // (5 * SR) + 2)])
    return out[:n]


async def vad_hears_speech(vad, pause):
    """Pipecat's Silero analyzer on pause audio: True if it ever reaches 'speaking' (what restarts the clock)."""
    from pipecat.audio.vad.vad_analyzer import VADState
    vad._vad_buffer, vad._vad_state, vad._vad_starting_count, vad._vad_stopping_count = b"", VADState.QUIET, 0, 0
    vad._prev_volume = 0
    vad._model.reset_states()
    pcm = (np.clip(pause, -1, 1) * 32767).astype(np.int16).tobytes()
    for i in range(0, len(pcm), 640):  # 20 ms chunks, as the transport delivers them
        if await vad.analyze_audio(pcm[i:i + 640]) == VADState.SPEAKING:
            return True
    return False


def main(root, n_pauses=None):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    texts, pauses, ends = ev.eval_keys(root, n_pauses)
    want = {}
    for d, g in pauses + ends:
        want.setdefault(d, []).append(g)
    stuttered = "fluencybank" in root
    if stuttered:
        import still_talking_v2_train as tr
        nets, fold_of = tr.fold_nets(os.path.dirname(root))
        head = lambda d, e: nets[fold_of[("fluencybank", d)]](torch.from_numpy(e)).detach().numpy()  # noqa: E731
    else:
        sess_h = ort.InferenceSession("data/still_talking/still_talking_v2_head.onnx")
        head = lambda d, e: sess_h.run(None, {"emb": e})[0]  # noqa: E731
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_with_embedding("data/still_talking/smart_turn_embed.onnx"), so)
    fe = WhisperFeatureExtractor(chunk_length=8)
    vad = SileroVADAnalyzer(sample_rate=SR)
    vad.set_sample_rate(SR)
    clips = load_noise()
    recs = {r["id"]: r for r in json.load(open(f"{root}/bench.json", encoding="utf8"))}

    def run(a):
        a = a[-8 * SR:]
        a = np.pad(a, (8 * SR - len(a), 0))
        f = fe(a, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=8 * SR,
               truncation=True, do_normalize=True).input_features.astype(np.float32)
        return sess.run(None, {"input_features": f})

    st = {lv: {} for lv in LEVELS}
    v2 = {lv: {} for lv in LEVELS}
    vad_out = {lv: {} for lv in LEVELS if lv is not None}
    for n, (d, gs) in enumerate(want.items()):
        with wave.open(f"{root}/wav16k/{d}.wav") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        segs = recs[d]["segments"]
        for g in gs:
            end = int(segs[g]["end"] * SR)
            nxt = int(segs[g + 1]["start"] * SR) if g + 1 < len(segs) else len(x)
            lo, hi = max(0, end - 8 * SR), min(len(x), end + int(PAUSE_VAD * SR), max(nxt, end + int(0.2 * SR)))
            clean = x[lo:hi]
            bed = noise_bed(clips, len(clean), np.random.default_rng(zlib.crc32(f"{d}|{g}".encode())))
            speech = active_rms(x[lo:end])
            for lv in LEVELS:
                y = clean if lv is None else clean + bed * speech / 10 ** (lv / 20)
                p_fin, _ = run(y[: end + int(0.2 * SR) - lo])
                _, emb = run(y[: end - lo])
                key = str(g)
                st[lv].setdefault(d, {})[key] = {"p_finished": round(float(p_fin[0][0]), 4)}
                v2[lv].setdefault(d, {})[key] = {"p_still_talking": round(float(head(d, emb.astype(np.float32))[0]), 4)}
                if lv is not None:  # the noise alone, as a truly silent pause would carry it (the recordings'
                    # pauses can hold the other party's voice, which would muddle this)
                    quiet = bed[end - lo:][: int(PAUSE_VAD * SR)] * speech / 10 ** (lv / 20)
                    vad_out[lv].setdefault(d, {})[key] = asyncio.run(vad_hears_speech(vad, quiet))
        print(f"{n + 1}/{len(want)} recordings", flush=True)
    for lv in LEVELS:
        tag = "_clean_check" if lv is None else f"_snr{lv}"
        json.dump(st[lv], open(f"{root}/smart_turn_signals{tag}.json", "w"))
        json.dump(v2[lv], open(f"{root}/still_talking_v2_signals{tag}.json", "w"))
    json.dump({f"snr{lv}": v for lv, v in vad_out.items()},
              open(f"{root}/vad_pause_signals.json", "w"))
    for lv, out in vad_out.items():
        hits = [v for gg in out.values() for v in gg.values()]
        print(f"{lv:>3} dB noise: the VAD hears speech in {np.mean(hits) * 100:5.1f}% of 1.5 s silent pauses")


if __name__ == "__main__":
    main(*sys.argv[1:3])
