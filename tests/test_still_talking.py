import os

import numpy as np
import pytest

from turnheed.still_talking import KINDS, log_mel

MODEL = "data/still_talking/still_talking.onnx"


def test_log_mel_shape_and_gain_invariance():
    x = np.random.default_rng(0).standard_normal(48000).astype(np.float32) * 0.1
    m = log_mel(x)
    assert m.shape == (298, 64)
    assert np.allclose(m, log_mel(x * 3), atol=1e-2)  # loudness drops out (full window; zero-padded starts are not gain-free)


@pytest.mark.skipif(not os.path.exists(MODEL), reason="train with bench/recipes/still_talking_train.py")
def test_model_runs_on_pcm_chunks():
    from turnheed.still_talking import StillTalking
    st = StillTalking(MODEL)
    for _ in range(10):
        st.push(np.zeros(1600, dtype=np.int16).tobytes())
    p, kind = st.read()
    assert 0.0 <= p <= 1.0 and kind in KINDS


V2 = ("data/still_talking/smart_turn_embed.onnx", "data/still_talking/still_talking_v2_head.onnx")


@pytest.mark.skipif(not all(map(os.path.exists, V2)), reason="run smart_turn_embed.py + still_talking_v2_train.py")
def test_v2_runs_on_pcm_chunks():
    pytest.importorskip("transformers")
    from turnheed.still_talking import StillTalkingV2
    st = StillTalkingV2(*V2)
    noise = (np.random.default_rng(0).standard_normal(16000) * 3000).astype(np.int16)
    st.push(noise.tobytes())
    p, kind = st.read()
    assert 0.0 <= p <= 1.0 and kind is None


def test_push_resamples_phone_audio():
    from turnheed.still_talking import _Buffered
    b = _Buffered()
    for _ in range(10):
        b.push(np.zeros(800, dtype=np.int16).tobytes(), sample_rate=8000)  # 1 s at 8 kHz, in 100 ms chunks
    assert abs(len(b._buf) - 16000) < 1200  # streaming soxr holds back ~66 ms; read only after >= 0.2 s silence
