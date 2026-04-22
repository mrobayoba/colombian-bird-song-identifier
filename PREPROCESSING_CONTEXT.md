
# 🦜 Audio Preprocessing & Tensor Extraction Plan

This document outlines the concrete technical plan for translating raw `.mp3` audio files into machine-learning-ready tensors. It maps directly to **Steps 6, 7 and 8** of the `pre-processing.ipynb` Colab notebook.

---

## 🎯 Approach & Objectives

Bird recordings from Xeno-Canto vary wildly in length, quality, sample rate, and noise profiles. Our goal is to extract uniform, high-quality feature representations (Mel-Spectrograms) while optimizing for Google Drive's I/O limits (avoiding millions of individual tiny image/array files).

### 1. Library Choices
- **Audio Loading & Processing**: `librosa`
- **Filtering**: `scipy.signal`
- **Tensor Storage**: `numpy` (uncompressed `.npz` grouped by species)

---

## 🛠️ Step 6: Automated Chunking & Silence Rejection

Because MP3 files can range from a few seconds to a full minute, we must chop them into uniform 3-second segments. Doing this manually for ~7,000 files would take hundreds of hours. Instead, we use an automated sliding-window approach that mathematically rejects empty or silent portions of the audio.

### A. Load & Standardize
- **Resample**: Force all audio to `sr = 32000` Hz (bird vocalizations span high frequencies; 32kHz captures up to 16kHz, which is sufficient).
- **Channels**: Force Mono (`mono=True`).

### B. Noise Reduction & Filtering
- **Band-pass Filter**: Apply a Butterworth filter to keep frequencies between `500 Hz` and `15000 Hz`. This eliminates low-frequency wind/traffic noise and extreme high-frequency mic hiss.

### C. Automated Segmentation (Chunking)
- **Window Length**: `3.0` seconds. (At 32kHz, this is exactly 96,000 samples).
- **Overlap**: `50%` (1.5 seconds stride). This artificially augments our dataset by giving multiple overlapping views of the same call, ensuring we don't accidentally split a bird call precisely down the middle without having a whole chunk containing it nearby.
- **Silence / Static Rejection**: Calculate the Root Mean Square (RMS) energy of each 3-second chunk. If the chunk's energy is below a dynamic threshold (relative to the file's peak energy), it is assumed to be silence or low-level background noise and is automatically discarded.

### D. Amplitude Normalization
- **Peak Normalization**: Scale each surviving chunk so its maximum absolute amplitude is `1.0`. Normalizing *after* segmentation ensures quiet bird calls in otherwise loud recordings are boosted to a standard level. 
*(If you want to perform manual spot-checking, this is the stage where the normalized `.wav` arrays could be saved/auditioned temporarily before tensor conversion)*.

---

## 🎵 Step 7: Convert Chunks to Mel-Spectrograms

This phase transforms the surviving 3-second audio chunks into images (spectrograms) for the neural network.

### A. Feature Extraction (Mel-Spectrogram)
- **Transform**: Short-Time Fourier Transform (STFT) mapped to the Mel scale.
- **Parameters**: 
  - `n_fft = 2048`
  - `hop_length = 512`
  - `n_mels = 128`
- **Output Shape**: `(128, 188)` (128 mel frequency bands x 188 time frames for a 3-second clip at 32kHz).
- **Scale**: Convert power spectrogram to Decibels (log scale) using `librosa.power_to_db`.

---

## 📦 Step 8: Export Tensors for Training

Saving raw `.wav` chunks or individual `.png`/`.npy` files for every 3-second clip will bottleneck Google Drive, making training incredibly slow. Instead, we use packed archives.

### A. Grouping Strategy
- **Per-Species Archives**: For each species, all valid `(128, 188)` spectrogram chunks from all its recordings are combined into a single 3D numpy array of shape `(num_chunks, 128, 188)`.

### B. Serialization
- Save this array as a `.npz` file (e.g., `Amazilia_amazonica.npz`) in a dedicated `bird_tensors/` directory on Drive.
- `np.savez_compressed` or `np.savez` offers rapid read speeds during the PyTorch/TensorFlow DataLoader phase.

### C. Metadata Tracking (Index)
- Alongside the tensors, generate an index file (`tensor_index.json` or `.csv`) mapping chunk counts per species and pre-calculating the Train/Validation/Test split boundaries (e.g., 70/15/15) so data leakage is prevented during training.

---

## 🚀 Summary of the Data Flow
```text
[Drive: bird_songs/Species_X/XC12345.mp3]
       │
       ▼  (librosa load)
[1] 32kHz Mono Audio Array
       │
       ▼  (scipy.signal)
[2] Band-pass Filtered Array (500Hz - 15kHz)
       │
       ▼  (numpy array slicing)
[3] Segmented into N Chunks (3 sec, 50% overlap)
       │
       ▼  (RMS thresholding)
[4] Drop Silent Chunks
       │
       ▼  (peak normalization)
[5] Normalized Chunks [-1.0, 1.0]
       │
       ▼  (librosa.feature.melspectrogram)
[6] Mel-Spectrogram DB Tensors (shape: 128 x 188)
       │
       ▼  (numpy.savez)
[Drive: bird_tensors/Species_X.npz]
```