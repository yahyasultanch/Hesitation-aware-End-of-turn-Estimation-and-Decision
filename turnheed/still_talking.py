"""'Still talking' audio model: is this silence a pause inside the turn (block, repetition, word search)
or the end of it? Looks at the 3 s of audio up to the moment speech stopped. numpy + onnxruntime only.

    st = StillTalking("still_talking.onnx")                 # v1: own small CNN+GRU, 3 s, ~4 ms
    st = StillTalkingV2("smart_turn_embed.onnx", "still_talking_v2_head.onnx")   # v2: Smart Turn encoder + head
    p_still, kind = st.predict(audio_float32_16k)
    FloorTurnAnalyzer(still_talking=st)    # Pipecat plugin: push(pcm16) every chunk, read() once when speech stops

v1 is trained by bench/recipes/still_talking_train.py, v2 by still_talking_v2_train.py (better: AUC 0.82 vs
~0.67 on FluencyBank, 0.72 vs 0.58 on unseen-speaker Buckeye; ~25 ms per read). p_still is balanced
(prior-free): 0.5 means the audio looks equally like a mid-turn pause and a turn end.
"""
from __future__ import annotations

import numpy as np

SR = 16000
WIN = 3 * SR
N_FFT, HOP, N_MELS = 400, 160, 64
KINDS = ["prolongation", "block", "repetition", "repetition", "word_search"]  # model heads, in order
HEADS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]       # SEP-28k columns


def _mel_fb() -> np.ndarray:
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = hz(np.linspace(mel(0), mel(SR / 2), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, 1 / SR)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid))).astype(np.float32)


_FB = _mel_fb()
_HANN = np.hanning(N_FFT).astype(np.float32)


def log_mel(x: np.ndarray) -> np.ndarray:
    """3 s float32 audio (left-padded with zeros if shorter) -> (298, 64) normalised log-mel."""
    x = np.asarray(x, dtype=np.float32)[-WIN:]
    x = np.pad(x, (WIN - len(x), 0))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * _HANN
    m = np.log(np.abs(np.fft.rfft(frames)) ** 2 @ _FB.T + 1e-6)
    return ((m - m.mean()) / 4).astype(np.float32)  # loudness-free: podcasts and interviews differ in gain


def _session(path: str, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    return ort.InferenceSession(path, sess_options=so)


class _Buffered:
    """push() every 16 kHz pcm16 chunk (cheap), read() the model on the latest audio when speech stops."""
    window = WIN

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, pcm16: bytes, sample_rate: int = SR) -> None:
        chunk = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768
        if sample_rate != SR:  # phone lines are 8 kHz; the models run at 16 kHz
            chunk = self._resample(chunk, sample_rate)
        self._buf = np.concatenate([self._buf, chunk])[-self.window:]

    def read(self) -> tuple[float, str | None]:
        return self.predict(self._buf)

    def _resample(self, chunk: np.ndarray, rate: int) -> np.ndarray:
        """Streaming soxr (what Pipecat's Smart Turn uses, and what training's phone simulation uses);
        linear interpolation if soxr is not installed."""
        try:
            import soxr
        except ImportError:
            n = int(round(len(chunk) * SR / rate))
            return np.interp(np.arange(n) * rate / SR, np.arange(len(chunk)), chunk).astype(np.float32)
        if getattr(self, "_rs_rate", None) != rate:
            self._rs, self._rs_rate = soxr.ResampleStream(rate, SR, 1, dtype="float32"), rate
        return self._rs.resample_chunk(chunk)


class StillTalking(_Buffered):
    def __init__(self, model_path: str, threads: int = 1):
        super().__init__()
        self._sess = _session(model_path, threads)

    def predict(self, audio: np.ndarray) -> tuple[float, str]:
        cont, kinds = self._sess.run(None, {"mel": log_mel(audio)[None]})
        return float(cont[0]), KINDS[int(np.argmax(kinds[0]))]


class StillTalkingV2(_Buffered):
    """Smart Turn v3.2's encoder (8 s, Whisper features) + a small head. No disfluency kind."""
    window = 8 * SR

    def __init__(self, embed_model: str, head_model: str, threads: int = 1):
        super().__init__()
        from transformers import WhisperFeatureExtractor  # ships with the pipecat extra
        self._fe = WhisperFeatureExtractor(chunk_length=8)
        self._enc, self._head = _session(embed_model, threads), _session(head_model, threads)

    def predict(self, audio: np.ndarray) -> tuple[float, None]:
        a = np.asarray(audio, dtype=np.float32)[-self.window:]
        a = np.pad(a, (self.window - len(a), 0))
        f = self._fe(a, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=self.window,
                     truncation=True, do_normalize=True).input_features.astype(np.float32)
        emb = self._enc.run(None, {"input_features": f})[1]
        return float(self._head.run(None, {"emb": emb})[0][0]), None
