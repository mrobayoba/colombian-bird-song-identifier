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

NOTE on fmin/fmax and BUTTER_HIGH_HZ: train.ipynb Step 11 does NOT specify
fmin/fmax in melspectrogram (librosa defaults apply: 0–16 000 Hz) and uses
BUTTER_HIGH_HZ = 15 000. pre_processing.ipynb uses different values (fmin=300,
fmax=10 000, F_MAX=10 000) but those reflect a later notebook revision that
diverges from the trained model. Do not change these constants here.
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

# RMS normalisation (mirrors pre_processing.ipynb Block 4)
TARGET_RMS: float = 0.1      # −20 dBFS target; matches training corpus normalisation
PEAK_CEILING: float = 0.99   # anti-clip safety margin after RMS scaling
MIN_RMS: float = 1e-6        # below this the recording is treated as pure silence

# Per-chunk energy gate (mirrors pre_processing.ipynb Block 3 SILENCE_DB)
SILENCE_DB: float = -40.0    # chunks whose RMS falls below this dBFS are discarded


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


def rms_normalise(audio: np.ndarray) -> np.ndarray | None:
    """Scale audio to TARGET_RMS with a PEAK_CEILING anti-clip pass.

    Returns None when the recording is too quiet to normalise (pure silence or
    a dead microphone), which the caller should treat as zero surviving chunks.
    Mirrors pre_processing.ipynb Block 4 two-step normalisation.
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        return None
    y = audio * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(y)))
    if peak > PEAK_CEILING:
        y = y * (PEAK_CEILING / peak)
    return y.astype(np.float32)


def audio_to_chunks(audio: np.ndarray) -> list[np.ndarray]:
    """Bandpass-filter, slice into overlapping 3 s chunks, drop silent ones.

    Chunks whose RMS energy falls below SILENCE_DB are discarded before mel
    computation, matching the silence-rejection logic in pre_processing.ipynb
    Block 3 and avoiding wasted inference on noise-only windows.
    """
    sos = make_bandpass_sos()
    filtered = sosfilt(sos, audio).astype(np.float32)
    chunks = []
    for s in range(0, len(filtered) - CHUNK_SAMPLES + 1, STRIDE_SAMPLES):
        chunk = filtered[s : s + CHUNK_SAMPLES].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_db = 20.0 * np.log10(rms) if rms > 0.0 else -200.0
        if rms_db >= SILENCE_DB:
            chunks.append(chunk)
    return chunks


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

    Returns an empty array with the right trailing shape if no chunk survives
    (pure silence, dead mic, or all chunks below the energy gate).  The
    orchestrator detects n_chunks == 0 and raises IdentificationError cleanly.
    """
    audio = rms_normalise(audio)
    if audio is None:
        return np.empty((0, N_MELS, TARGET_FRAMES), dtype=np.float32)
    chunks = audio_to_chunks(audio)
    if not chunks:
        return np.empty((0, N_MELS, TARGET_FRAMES), dtype=np.float32)
    return np.stack([chunk_to_tensor(c) for c in chunks], axis=0)


def libs_available() -> bool:
    return _HAVE_AUDIO_LIBS
