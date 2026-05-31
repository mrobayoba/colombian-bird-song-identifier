"""
Bird song audio recorder and pre-processor.

Records audio from the microphone, applies the same pre-processing pipeline
used in pre-processing.ipynb and train.ipynb Step 11, and emits the resulting
spectrogram tensor as JSON — ready to pipe directly into model.py.

Pre-processing pipeline
-----------------------
1. Resample to SR=32 000 Hz (recorded natively at that rate)
2. Butterworth bandpass filter 1 000–15 000 Hz
3. Slice into 3-second chunks with 50% stride
4. Log-Mel spectrogram (n_fft=2048, hop=512, n_mels=128)
5. Convert to dB, clip to [-80, 0], normalise to [0, 1]
6. Pad / crop to TARGET_FRAMES=188

Output JSON
-----------
A 3-D array of shape (K, 128, 188) where K is the number of 3-second chunks
extracted from the recording. Values are float32 in [0, 1].

Usage examples
--------------
  # Record 10 s, print tensor JSON to stdout
  python recorder.py

  # Record up to 20 s, save tensor to file
  python recorder.py --duration 20 --output tensor.json

  # Full pipeline (record -> infer)
  python recorder.py --output tensor.json && python model.py --tensor-file tensor.json

  # List audio devices
  python recorder.py --list-devices

  # Keep the raw audio for debugging
  python recorder.py --save-audio recording.wav --output tensor.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt


# ── Pre-processing constants (must match pre-processing.ipynb) ────────────
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

MAX_DURATION   = 20


def _error(message: str) -> None:
    print(json.dumps({"error": message}))
    sys.exit(1)


def _make_bandpass_sos(low_hz: float, high_hz: float, sr: int, order: int = 5):
    nyq = sr / 2.0
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def _audio_to_chunks(audio: np.ndarray) -> list[np.ndarray]:
    sos      = _make_bandpass_sos(BUTTER_LOW_HZ, BUTTER_HIGH_HZ, SR)
    filtered = sosfilt(sos, audio).astype(np.float32)
    return [
        filtered[s : s + CHUNK_SAMPLES].copy()
        for s in range(0, len(filtered) - CHUNK_SAMPLES + 1, STRIDE_SAMPLES)
    ]


def _chunk_to_tensor(chunk: np.ndarray) -> np.ndarray:
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


def record_audio(duration: int, device: int | None) -> np.ndarray:
    """
    Record `duration` seconds of mono audio at SR from the given device.
    Prints a live countdown to stderr so stdout stays clean for the JSON tensor.
    """
    print(f"Recording for {duration}s ... (press Ctrl+C to stop early)",
          file=sys.stderr)

    frames       = duration * SR
    audio_buffer = np.empty(frames, dtype=np.float32)
    recorded     = [0]

    def _callback(indata, frame_count, time_info, status):
        if status:
            print(f"  [audio warning] {status}", file=sys.stderr)
        start = recorded[0]
        end   = min(start + frame_count, frames)
        n     = end - start
        audio_buffer[start:end] = indata[:n, 0]
        recorded[0] = end

    start_ts = time.time()
    try:
        with sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            device=device,
            callback=_callback,
        ):
            while recorded[0] < frames:
                elapsed   = time.time() - start_ts
                remaining = duration - int(elapsed)
                print(f"\r  {remaining:2d}s remaining ...  ", end="", file=sys.stderr)
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n  Recording stopped early.", file=sys.stderr)

    print(file=sys.stderr)   # newline after countdown

    actual_samples = recorded[0]
    if actual_samples == 0:
        _error("No audio was captured.")

    return audio_buffer[:actual_samples]


def preprocess(audio: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D float32 audio array to a (K, 128, 188) tensor batch.
    Exits with an error if the recording is too short for one 3-second chunk.
    """
    chunks = _audio_to_chunks(audio)

    if not chunks:
        _error(
            f"Recording is too short to extract any chunks. "
            f"Minimum required: {CHUNK_LEN:.0f}s of audio "
            f"({CHUNK_SAMPLES} samples at {SR} Hz)."
        )

    tensors = np.stack([_chunk_to_tensor(c) for c in chunks], axis=0)
    print(f"  Chunks extracted : {len(chunks)}", file=sys.stderr)
    print(f"  Tensor shape     : {tensors.shape}", file=sys.stderr)
    return tensors


def list_devices() -> None:
    print(sd.query_devices())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record bird song and emit a pre-processed tensor as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        metavar="SECONDS",
        help=f"Recording duration in seconds (default: 10, max: {MAX_DURATION}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write tensor JSON to this file. Defaults to stdout.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Audio input device index (default: system default). "
             "Use --list-devices to find the index.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument(
        "--save-audio",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also save the raw recording as a WAV file (useful for debugging).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not (1 <= args.duration <= MAX_DURATION):
        _error(f"--duration must be between 1 and {MAX_DURATION} seconds.")

    # ── Record ────────────────────────────────────────────────────────────
    audio = record_audio(args.duration, args.device)

    # ── Optionally save raw WAV ───────────────────────────────────────────
    if args.save_audio is not None:
        sf.write(str(args.save_audio), audio, SR)
        print(f"  Raw audio saved  : {args.save_audio}", file=sys.stderr)

    # ── Pre-process ───────────────────────────────────────────────────────
    tensors = preprocess(audio)

    # ── Serialise — round to 5 dp to keep JSON files manageable ──────────
    tensor_list = np.round(tensors, 5).tolist()
    json_str    = json.dumps(tensor_list, separators=(",", ":"))

    if args.output is not None:
        args.output.write_text(json_str, encoding="utf-8")
        print(f"  Tensor JSON saved: {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
