"""Has the caller finished their thought? Pipecat's Smart Turn v3 model on the phrase's audio.

A fixed silence timer either cuts people off when they pause to think or makes
every answer wait. The model listens to how the phrase ends instead.
`finished(pcm)` returns the probability that a 16 kHz PCM16 phrase is complete,
or None when the model is not available (then the caller falls back to the timer).
"""

import logging
import time

import numpy as np

log = logging.getLogger("jarvis")
REPO, FILE = "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"
_session = _features = None
_failed = False


def load():
    """Load the model (a download of a few MB the first time). Safe to call again."""
    global _session, _features, _failed
    if _session or _failed:
        return _session is not None
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        so = ort.SessionOptions()
        so.inter_op_num_threads = so.intra_op_num_threads = 1
        _session = ort.InferenceSession(hf_hub_download(REPO, FILE), sess_options=so)
        _features = WhisperFeatureExtractor(chunk_length=8)
        finished(bytes(32000))     # first run is slow
    except Exception as e:
        _failed = True
        log.warning("turn model not available, using the silence timer: %r", e)
    return _session is not None


def finished(pcm):
    if not load():
        return None
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
    x = x[-8 * 16000:]                                   # the model hears the last 8 seconds,
    x = np.pad(x, (8 * 16000 - len(x), 0))               # padded with silence in front
    f = _features(x, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000,
                  truncation=True, do_normalize=True).input_features.astype(np.float32)
    return float(_session.run(None, {"input_features": f})[0].reshape(-1)[0])
