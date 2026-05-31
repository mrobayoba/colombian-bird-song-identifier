"""
Bird song inference script.

Loads a trained ResNet50V2 .keras model and predicts the top-3 most likely
Colombian bird species from a pre-processed log-Mel spectrogram tensor.

Input tensor JSON format
------------------------
A 2-D array  (single spectrogram):  [[row0], [row1], ...]  — shape (128, 188)
A 3-D array  (batch of spectrograms): [[[...]], ...]       — shape (K, 128, 188)

Values must be float32-compatible, normalised to [0, 1] using the same
pipeline as pre-processing.ipynb (DB_MIN=-80, DB_MAX=0).

Output JSON format
------------------
{
  "predictions": [
    {"rank": 1, "species": "Turdus fuscater",    "score": 0.812},
    {"rank": 2, "species": "Zonotrichia capensis","score": 0.103},
    {"rank": 3, "species": "Mimus gilvus",        "score": 0.041}
  ]
}

On error the script exits with code 1 and writes a JSON error object to stdout:
  {"error": "<message>"}

Usage
-----
  python model.py --tensor '<json>'
  python model.py --tensor-file spectrogram.json
  python model.py --tensor '<json>' --model path/to/model.keras --label-map path/to/label_map.json
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf


# ── Defaults ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_MODEL     = _SCRIPT_DIR / "resnet50v2_fold2.keras"
_DEFAULT_LABEL_MAP = _SCRIPT_DIR / "data" / "label_map.json"
_FALLBACK_FILTERED = _SCRIPT_DIR / "data" / "colombia_birds_filtered.json"


def _error(message: str) -> None:
    """Print a JSON error object and exit with code 1."""
    print(json.dumps({"error": message}))
    sys.exit(1)


def _load_label_map(label_map_path: Path) -> dict[int, str]:
    """
    Load label_map.json (int-key → species name).

    Falls back to reconstructing the map alphabetically from
    colombia_birds_filtered.json when label_map.json is absent.
    """
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {int(k): v for k, v in raw.items()}

    if _FALLBACK_FILTERED.exists():
        with open(_FALLBACK_FILTERED, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        species_keys = sorted(
            name.replace(" ", "_") for name in data.get("species", {}).keys()
        )
        return {i: name for i, name in enumerate(species_keys)}

    _error(
        f"label_map.json not found at '{label_map_path}' and no fallback "
        f"available at '{_FALLBACK_FILTERED}'."
    )


def _parse_tensor(raw_json: str) -> np.ndarray:
    """
    Parse a JSON string into a float32 numpy array.

    Accepts:
      - 2-D list  (128, 188)    → returned as (1, 128, 188)
      - 3-D list  (K, 128, 188) → returned as-is

    Shape validation: inner dimensions must be (128, 188).
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        _error(f"Invalid JSON for tensor: {exc}")

    arr = np.array(data, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]   # (128, 188) → (1, 128, 188)

    if arr.ndim != 3:
        _error(
            f"Tensor must be 2-D (128×188) or 3-D (K×128×188), "
            f"got shape {arr.shape}."
        )

    if arr.shape[1] != 128 or arr.shape[2] != 188:
        _error(
            f"Tensor inner dimensions must be (128, 188), "
            f"got {arr.shape[1]}×{arr.shape[2]}."
        )

    return arr


# ── Model architecture constants (must match train.ipynb Step 6) ─────────
_INPUT_HEIGHT = 128
_INPUT_WIDTH  = 188
_DROPOUT_RATE = 0.4
_DENSE_UNITS  = 256


def _build_architecture(n_classes: int) -> tf.keras.Model:
    """
    Rebuild the BirdSongClassifier architecture with an explicit output_shape
    on the Lambda layer (required by Keras 3).
    """
    inputs = tf.keras.Input(
        shape=(_INPUT_HEIGHT, _INPUT_WIDTH, 1), name="spectrogram"
    )
    x = tf.keras.layers.Lambda(
        lambda t: tf.repeat(t, 3, axis=-1),
        name="to_rgb",
        output_shape=(_INPUT_HEIGHT, _INPUT_WIDTH, 3),
    )(inputs)
    base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_shape=(_INPUT_HEIGHT, _INPUT_WIDTH, 3),
    )
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(_DENSE_UNITS, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(_DROPOUT_RATE, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="predictions"
    )(x)
    return tf.keras.Model(inputs, outputs, name="BirdSongClassifier")


def _load_weights_from_keras_archive(
    model: tf.keras.Model, model_path: Path
) -> None:
    """
    Extract model.weights.h5 from the .keras ZIP archive and load it
    into the already-built model.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(model_path), "r") as zf:
            zf.extract("model.weights.h5", tmp)
        model.load_weights(str(Path(tmp) / "model.weights.h5"))


def _load_model(model_path: Path, n_classes: int) -> tf.keras.Model:
    """
    Load the .keras model from disk.

    Primary path: tf.keras.models.load_model with safe_mode=False.
    Fallback (Keras 3 Lambda output_shape issue): rebuilds the known
    architecture and loads the weights directly from the .keras archive.
    """
    if not model_path.exists():
        _error(f"Model file not found: '{model_path}'")

    try:
        return tf.keras.models.load_model(str(model_path), safe_mode=False)
    except (NotImplementedError, Exception) as exc:
        if "output_shape" not in str(exc) and "Lambda" not in str(exc):
            _error(f"Failed to load model: {exc}")

    # Keras 3 fallback: rebuild architecture + load weights from the archive
    print(
        "[model.py] Direct load failed (Keras 3 Lambda compat); "
        "rebuilding architecture and loading weights from archive.",
        file=sys.stderr,
    )
    model = _build_architecture(n_classes)
    _load_weights_from_keras_archive(model, model_path)
    return model


def infer(
    tensors: np.ndarray,
    model: tf.keras.Model,
    label_map: dict[int, str],
    top_n: int = 3,
) -> dict:
    """
    Run inference on a batch of spectrograms.

    Parameters
    ----------
    tensors   : float32 array, shape (K, 128, 188)
    model     : loaded Keras model
    label_map : {int: species_name}
    top_n     : number of top predictions to return

    Returns
    -------
    dict with a "predictions" list, each entry being
    {"rank": int, "species": str, "score": float}.
    """
    tensors_in  = np.expand_dims(tensors, axis=-1)      # (K, 128, 188, 1)
    chunk_probs = model.predict(tensors_in, verbose=0)  # (K, N_CLASSES)
    avg_probs   = chunk_probs.mean(axis=0)              # (N_CLASSES,)

    top_indices = np.argsort(avg_probs)[::-1][:top_n]

    predictions = [
        {
            "rank":    int(rank),
            "species": label_map[int(idx)].replace("_", " "),
            "score":   round(float(avg_probs[idx]), 6),
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

    return {"predictions": predictions}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer bird species from a log-Mel spectrogram tensor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    tensor_group = parser.add_mutually_exclusive_group(required=True)
    tensor_group.add_argument(
        "--tensor",
        metavar="JSON",
        help="JSON-encoded 2-D or 3-D tensor array (inline string).",
    )
    tensor_group.add_argument(
        "--tensor-file",
        metavar="PATH",
        type=Path,
        help="Path to a JSON file containing the tensor array.",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        type=Path,
        default=_DEFAULT_MODEL,
        help=f"Path to the .keras model file (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--label-map",
        metavar="PATH",
        type=Path,
        default=_DEFAULT_LABEL_MAP,
        help=f"Path to label_map.json (default: {_DEFAULT_LABEL_MAP}).",
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=int,
        default=3,
        help="Number of top predictions to return (default: 3).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Load tensor ───────────────────────────────────────────────────────
    if args.tensor_file is not None:
        if not args.tensor_file.exists():
            _error(f"Tensor file not found: '{args.tensor_file}'")
        raw_json = args.tensor_file.read_text(encoding="utf-8")
    else:
        raw_json = args.tensor

    tensors = _parse_tensor(raw_json)

    # ── Load label map (needed for n_classes before model load) ──────────
    label_map = _load_label_map(args.label_map)
    n_classes = len(label_map)

    # ── Load model ────────────────────────────────────────────────────────
    model = _load_model(args.model, n_classes)

    # ── Run inference ─────────────────────────────────────────────────────
    result = infer(tensors, model, label_map, top_n=args.top)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
