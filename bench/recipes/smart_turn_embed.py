"""Smart Turn v3.2's pooled encoder output (384-d) for every gap of a bench set: features for still-talking v2.

  python bench/recipes/smart_turn_embed.py data/fluencybank data/buckeye [--phone]   # needs the pipecat extra
  --phone: the same audio through a simulated phone line (8 kHz mu-law), written to smart_turn_embed_phone.npz

Writes data/still_talking/smart_turn_embed.onnx once (the bundled BSD-2 model with its pooled embedding
exposed as a second output), and <root>/smart_turn_embed.npz per set: emb (N,384) f16, id (N,), gap (N,),
p_finished (N,). Window: the 8 s up to the moment the speech run ends (same as still-talking v1, so the
model never hears the interviewer start).
"""
import importlib.resources
import json
import os
import sys
import wave

import numpy as np
import onnx
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

SR = 16000
EMB = "sum_1_DequantizeLinear_Output"


def model_with_embedding(out):
    if not os.path.exists(out):
        src = importlib.resources.files("pipecat.audio.turn.smart_turn.data").joinpath("smart-turn-v3.2-cpu.onnx")
        m = onnx.load(str(src))
        m.graph.output.append(onnx.helper.make_tensor_value_info(EMB, onnx.TensorProto.FLOAT, None))
        onnx.save(m, out)
    return out


def phone(x: np.ndarray) -> np.ndarray:
    """What a Twilio call does to the audio: 16 kHz -> 8 kHz -> mu-law -> back to 16 kHz (soxr both ways, as in
    the live pipeline)."""
    import audioop
    import soxr
    x8 = np.clip(soxr.resample(x, SR, 8000) * 32768, -32768, 32767).astype(np.int16)
    x8 = np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(x8.tobytes(), 2), 2), dtype=np.int16).astype(np.float32) / 32768
    return soxr.resample(x8, 8000, SR).astype(np.float32)


def main(roots, tel=False):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_with_embedding("data/still_talking/smart_turn_embed.onnx"), so)
    fe = WhisperFeatureExtractor(chunk_length=8)
    for root in roots:
        emb, ids, gaps, pf = [], [], [], []
        for rec in json.load(open(f"{root}/bench.json", encoding="utf8")):
            with wave.open(f"{root}/wav16k/{rec['id']}.wav") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            if tel:
                x = phone(x)
            for g, s in enumerate(rec["segments"]):
                end = int(s["end"] * SR)
                a = x[max(0, end - 8 * SR):end]
                a = np.pad(a, (8 * SR - len(a), 0))
                f = fe(a, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=8 * SR,
                       truncation=True, do_normalize=True).input_features.astype(np.float32)
                p, e = sess.run(None, {"input_features": f})
                emb.append(e[0].astype(np.float16)), ids.append(rec["id"]), gaps.append(g), pf.append(float(p[0][0]))
            print(root, rec["id"], flush=True)
        np.savez(f"{root}/smart_turn_embed{'_phone' if tel else ''}.npz", emb=np.stack(emb), id=np.array(ids), gap=np.array(gaps),
                 p_finished=np.array(pf, np.float32))


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if a != "--phone"], tel="--phone" in args)
