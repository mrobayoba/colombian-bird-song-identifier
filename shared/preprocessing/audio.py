"""Canonical audio preprocessing — the SINGLE source of train/serve parity.

These constants and functions are lifted verbatim from the inference helper in
train.ipynb (Step 11) and the tensor export in pre-processing.ipynb (Step 8) so
that the serving path produces byte-identical tensors to what the model trained
on. If you change a value in the notebook, change it HERE and nowhere else; the
Audio Recorder Agent and any local re-derivation import from this module.

NOTE on the bandpass low cut: train.ipynb's inference helper uses 1000 Hz,
while PREPROCESSING_CONTEXT.md documents 500 Hz. The notebook's actual value
wins for serving parity, so BUTTER_LOW_HZ = 1000 here. Reconcile the doc when
convenient, but do not silently diverge the serving code from the trained model.
"""

from __future__ import annotations

import numpy as np

try:  # librosa/scipy are only needed where real preprocessing runs
    import librosa
    from scipy.signal import butter, sosfilt

    _HAVE_AUDIO_LIBS = True
except Exception:  # pragma: no cover - allows import in stub-only environments
    _HAVE_AUDIO_LIBS = False


# --------------------------------------------------------------------------- #
# Constants (must match the notebooks)
# --------------------------------------------------------------------------- #
SR: int = 32_000
CHUNK_LEN: float = 3.0
CHUNK_SAMPLES: int = int(SR * CHUNK_LEN)          # 96_000
STRIDE_SAMPLES: int = CHUNK_SAMPLES // 2          # 50% overlap

BUTTER_LOW_HZ: int = 1_000                         # see module docstring
BUTTER_HIGH_HZ: int = 15_000
BUTTER_ORDER: int = 5

N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 128
TARGET_FRAMES: int = 188                            # (128, 188) tensors

DB_MIN: float = -80.0
DB_MAX: float = 0.0

INPUT_HEIGHT: int = N_MELS                           # 128
INPUT_WIDTH: int = TARGET_FRAMES                     # 188


# --------------------------------------------------------------------------- #
# Core functions (mirror train.ipynb Step 11)
# --------------------------------------------------------------------------- #
def make_bandpass_sos(
    low_hz: int = BUTTER_LOW_HZ,
    high_hz: int = BUTTER_HIGH_HZ,
    sr: int = SR,
    order: int = BUTTER_ORDER,
):
    nyq = sr / 2.0
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def audio_to_chunks(audio: np.ndarray) -> list[np.ndarray]:
    """Bandpass-filter then slice into overlapping 3 s chunks."""
    sos = make_bandpass_sos()
    filtered = sosfilt(sos, audio).astype(np.float32)
    return [
        filtered[s : s + CHUNK_SAMPLES].copy()
        for s in range(0, len(filtered) - CHUNK_SAMPLES + 1, STRIDE_SAMPLES)
    ]


def chunk_to_tensor(chunk: np.ndarray) -> np.ndarray:
    """Mel-spectrogram → log dB → fixed-range [0,1], padded/cropped to 188 frames."""
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db = librosa.power_to_db(mel, ref=np.max)
    tensor = (np.clip(db, DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)
    if tensor.shape[1] < TARGET_FRAMES:
        tensor = np.pad(tensor, ((0, 0), (0, TARGET_FRAMES - tensor.shape[1])))
    else:
        tensor = tensor[:, :TARGET_FRAMES]
    return tensor.astype(np.float32)


def audio_bytes_to_tensors(audio: np.ndarray) -> np.ndarray:
    """Full path: raw mono 32 kHz audio array -> stacked (K, 128, 188) tensors.

    Returns an empty array with the right trailing shape if no chunk survives.
    """
    chunks = audio_to_chunks(audio)
    if not chunks:
        return np.empty((0, N_MELS, TARGET_FRAMES), dtype=np.float32)
    return np.stack([chunk_to_tensor(c) for c in chunks], axis=0)


def libs_available() -> bool:
    return _HAVE_AUDIO_LIBS
