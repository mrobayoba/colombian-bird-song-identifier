
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

### E. Export 3-Second Audio Chunks
- **Format & Structure**: The raw files downloaded from Xeno-Canto are `.mp3`. During loading (`librosa.load`), they are decoded into raw audio arrays. We must then save the processed, 3-second chunks as uncompressed `.wav` files. 
- **Directory Mirroring**: Save the output `.wav` chunks in a new directory (e.g., `processed_audio/`) mirroring the original folder structure (separated by species folders: `Species_X/XC12345_chunk1.wav`). This makes it easy to validate the dataset, audition specific calls, and track data lineage before spectrogram conversion.

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

### B. Diagnostic Visualization for Normalization Selection
- **Goal**: Before locking in the final tensor normalization, generate summary plots from a representative sample of spectrogram chunks across many species and recording conditions.
- **Spectrogram Panels**: Plot side-by-side examples of raw log-Mel tensors for quiet, medium, and loud recordings to verify whether bird calls remain visible against the background.
- **Value Distribution Plots**: Build histograms of all log-Mel dB values and mark key percentiles (`1`, `5`, `50`, `95`, `99`). This helps choose a stable clipping range such as `[-80, 0]` dB instead of guessing.
- **Per-Band Statistics**: Plot mean and standard deviation per Mel bin across the sampled training chunks. This reveals whether per-frequency standardization is justified.
- **Normalization Comparisons**: Visualize the same chunk under multiple candidate normalizations: fixed-range min-max on clipped dB values, dataset-level z-score, and optionally PCEN. Compare which one preserves bird structure while avoiding background amplification.
- **Sampling Rule**: Use only the future training split for these statistics and plots so the chosen normalization does not leak information from validation or test data.

### C. Final Tensor Normalization
- **Default Choice**: After converting to log-Mel dB, clip the tensor to a fixed range such as `[-80, 0]` dB and scale it to `[0, 1]`. This is a strong baseline for bird audio because it keeps values bounded and comparable across files.
- **Alternative Choice**: If the diagnostic plots show strong frequency-dependent variance, compute training-set mean and standard deviation per Mel bin and apply z-score normalization using only training data.
- **Avoid Per-Chunk Scaling**: Do not apply per-spectrogram min-max normalization as the final export format, because it can over-amplify noisy background-only chunks and erase useful loudness contrast between examples.
- **Metadata**: Save the selected normalization method and its parameters (for example, dB clip range or per-band mean/std arrays) alongside the tensor index so inference uses the exact same transformation.

---

## 📦 Step 8: Export Tensors for Training

Saving raw `.wav` chunks or individual `.png`/`.npy` files for every 3-second clip will bottleneck Google Drive, making training incredibly slow. Instead, we use packed archives.

### A. Grouping Strategy
- **Per-Species Archives**: For each species, all valid `(128, 188)` spectrogram chunks from all its recordings are combined into a single 3D numpy array of shape `(num_chunks, 128, 188)`.

### B. Serialization
- Save this array as a `.npz` file (e.g., `Amazilia_amazonica.npz`) in a dedicated `bird_tensors/` directory on Drive.
- `np.savez_compressed` or `np.savez` offers rapid read speeds during the PyTorch/TensorFlow DataLoader phase.

### C. Metadata Tracking (Index)
- Alongside the tensors, generate an index file (`tensor_index.json`) that records chunk counts per species and pre-calculates **5-fold cross-validation** split boundaries so data leakage is prevented during training.

- **Fold Strategy**: Assign each chunk to one of 5 equally-sized folds (each fold ≈ 20% of the species' chunks). Assignment is done by sequential index after a deterministic shuffle (`random_seed = 42`) so results are reproducible.
- **Usage**: For each of the 5 possible experiments, one fold serves as the **test set** (20%) and the remaining 4 folds serve as the **training set** (80%). At minimum, 3 distinct train/test combinations should be run per model.
- **Index structure**: For each species, the index stores the fold assignment for every chunk as a list of integers (`0`–`4`), rather than fixed index ranges, so any combination of folds can be selected at training time without rewriting the index.
- **No separate validation split at export time**: Validation can be carved out of the training folds at training time (e.g., hold out one training fold), keeping the preprocessing step decoupled from specific training configurations.

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
       ├──> (soundfile.write)
       │    [Drive: processed_audio/Species_X/XC12345_chunk01.wav]
       │
       ▼  (librosa.feature.melspectrogram)
[6] Mel-Spectrogram DB Tensors (shape: 128 x 188)
       │
       ▼  (numpy.savez)
[Drive: bird_tensors/Species_X.npz]
```

**5-Fold Cross-Validation Layout (per species)**
```text
Chunks (shuffled, seed=42):  [ 0  1  2  3  4  5  6  7  8  9  ... ]
                               └─fold 0─┘└─fold 1─┘└─fold 2─┘ ...

Fold used as TEST →  Folds used as TRAIN        Experiment
       0             1 + 2 + 3 + 4  (80%)           A
       1             0 + 2 + 3 + 4  (80%)           B
       2             0 + 1 + 3 + 4  (80%)           C
      (3)            0 + 1 + 2 + 4  (80%)          (D)
      (4)            0 + 1 + 2 + 3  (80%)          (E)

tensor_index.json per species:
  { "chunk_count": N,
    "npz_file": "Species_X.npz",
    "fold_assignments": [0, 2, 1, 4, 3, 0, ...]  ← one int per chunk }
```