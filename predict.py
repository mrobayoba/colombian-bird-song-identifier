import argparse
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import butter, sosfilt

# Import our custom classifier
from model import BirdSongClassifier

# ── Pre-processing constants (must match notebook) ────────────────────────
SR             = 32_000
CHUNK_LEN      = 3.0
CHUNK_SAMPLES  = int(SR * CHUNK_LEN)
STRIDE_SAMPLES = CHUNK_SAMPLES // 2
BUTTER_LOW_HZ  = 1_000
BUTTER_HIGH_HZ = 15_000
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 128
TARGET_FRAMES  = 188
DB_MIN         = -80.0
DB_MAX         = 0.0

def make_bandpass_sos(low_hz, high_hz, sr, order=5):
    nyq = sr / 2.0
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")

def audio_to_chunks(audio):
    sos      = make_bandpass_sos(BUTTER_LOW_HZ, BUTTER_HIGH_HZ, SR)
    filtered = sosfilt(sos, audio).astype(np.float32)
    return [
        filtered[s : s + CHUNK_SAMPLES].copy()
        for s in range(0, len(filtered) - CHUNK_SAMPLES + 1, STRIDE_SAMPLES)
    ]

def chunk_to_tensor(chunk):
    mel    = librosa.feature.melspectrogram(
        y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db     = librosa.power_to_db(mel, ref=np.max)
    tensor = (np.clip(db, DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)
    if tensor.shape[1] < TARGET_FRAMES:
        tensor = np.pad(tensor, ((0, 0), (0, TARGET_FRAMES - tensor.shape[1])))
    else:
        tensor = tensor[:, :TARGET_FRAMES]
    return tensor.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Predict bird species from audio")
    parser.add_argument("audio_path", type=str, help="Path to the raw audio file (.mp3, .wav, etc.)")
    parser.add_argument("--weights", type=str, default="fold1_best.weights.h5", help="Path to weights file (default: fold1_best.weights.h5)")
    parser.add_argument("--label-map", type=str, default="label_map.json", help="Path to label map JSON file (default: label_map.json)")
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
        
    chunks = audio_to_chunks(audio)
    if not chunks:
        print("Error: Audio is too short to extract any valid chunks (needs >3 seconds).")
        return
        
    tensors = np.stack([chunk_to_tensor(c) for c in chunks], axis=0)
    print(f"Extracted {len(chunks)} chunks of 3.0s each.")

    print(f"\nLoading model... (this may take a few seconds)")
    try:
        classifier = BirdSongClassifier(
            weights_path=args.weights,
            label_map_path=args.label_map
        )
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return
    
    print("Running inference...")
    result_json = classifier.predict_top3(tensors)
    
    print("\n── Top-3 Predicted Species ──────────────────")
    print(result_json)

if __name__ == "__main__":
    main()
