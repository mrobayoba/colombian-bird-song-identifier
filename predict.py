import argparse
import numpy as np
import pandas as pd
import librosa
from pathlib import Path

# Import our custom classifier
from model import BirdSongClassifier

# ── Pre-processing constants (must match notebook) ────────────────────────
SR             = 22_050
CHUNK_LEN      = 3.0
MIN_CHUNK_LEN  = 1.0
CHUNK_SAMPLES  = int(SR * CHUNK_LEN)
STRIDE_SAMPLES = CHUNK_SAMPLES // 2
N_FFT          = 1024
HOP_LENGTH     = 256
N_MELS         = 128
F_MIN          = 300.0
F_MAX          = 10_000.0
TOP_DB         = 80
TARGET_FRAMES  = 431

def load_norm_stats(norm_stats_path):
    norm_stats_path = Path(norm_stats_path)
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found at {norm_stats_path}")

    norm_df = pd.read_parquet(norm_stats_path)
    required_columns = {"mel_bin", "mean", "std"}
    missing_columns = required_columns - set(norm_df.columns)
    if missing_columns:
        raise ValueError(
            f"norm_stats.parquet is missing required columns: {sorted(missing_columns)}"
        )

    norm_df = norm_df.sort_values("mel_bin")
    norm_mean = norm_df["mean"].to_numpy(dtype=np.float32)
    norm_std = np.maximum(norm_df["std"].to_numpy(dtype=np.float32), 1e-6)
    if len(norm_mean) != N_MELS or len(norm_std) != N_MELS:
        raise ValueError(
            f"Expected {N_MELS} norm-stat rows, got mean={len(norm_mean)}, std={len(norm_std)}"
        )
    return norm_mean, norm_std

def audio_to_chunks(audio):
    if len(audio) < int(MIN_CHUNK_LEN * SR):
        return []
    if len(audio) < CHUNK_SAMPLES:
        padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        return [padded]

    chunks = []
    for start in range(0, len(audio), STRIDE_SAMPLES):
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) == CHUNK_SAMPLES:
            chunks.append(chunk.astype(np.float32))
        elif len(chunk) >= int(MIN_CHUNK_LEN * SR):
            padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunks.append(padded)
            break
        else:
            break
    return chunks

def chunk_to_tensor(chunk, norm_mean, norm_std):
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)

    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])),
            mode="constant",
            constant_values=log_mel.min(),
        )
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]

    tensor = (log_mel - norm_mean[:, None]) / norm_std[:, None]
    return tensor[:, :, np.newaxis].astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Predict bird species from audio")
    parser.add_argument("audio_path", type=str, help="Path to the raw audio file (.mp3, .wav, etc.)")
    parser.add_argument(
        "--weights",
        type=str,
        default="resnet50v2_index_split_best.weights.h5",
        help="Path to weights file (default: resnet50v2_index_split_best.weights.h5)",
    )
    parser.add_argument(
        "--dataset-index",
        type=str,
        default="dataset_index.parquet",
        help="Path to dataset index parquet file (default: dataset_index.parquet)",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="norm_stats.parquet",
        help="Path to normalization stats parquet file (default: norm_stats.parquet)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return

    print(f"Loading audio from {audio_path.name}...")
    try:
        audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    try:
        norm_mean, norm_std = load_norm_stats(args.norm_stats)
    except Exception as e:
        print(f"Error loading normalization stats: {e}")
        return
        
    chunks = audio_to_chunks(audio)
    if not chunks:
        print("Error: Audio is too short to extract any valid chunks (needs >=1 second).")
        return
        
    tensors = np.stack([chunk_to_tensor(c, norm_mean, norm_std) for c in chunks], axis=0)
    print(f"Extracted {len(chunks)} chunks of 3.0s each.")
    print(f"Tensor shape: {tensors.shape}")

    print(f"\nLoading model... (this may take a few seconds)")
    try:
        classifier = BirdSongClassifier(
            weights_path=args.weights,
            dataset_index_path=args.dataset_index
        )
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return
    
    print("Running inference...")
    result_json = classifier.predict_top3(tensors)
    
    print("\n--- Top-3 Predicted Species ---")
    print(result_json)

if __name__ == "__main__":
    main()
