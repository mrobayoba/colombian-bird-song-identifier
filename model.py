import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

class BirdSongClassifier:
    def __init__(
        self,
        weights_path: str,
        dataset_index_path: str,
        input_height=128,
        input_width=431,
        input_channels=1,
        dropout_rate=0.3,
        dense_units=256,
    ):
        """
        Rebuilds the training architecture and restores a .weights.h5 checkpoint.
        Labels are derived from the preprocessing dataset index.
        """
        self.weights_path = Path(weights_path)
        self.dataset_index_path = Path(dataset_index_path)
        self.input_shape = (input_height, input_width, input_channels)
        
        if not self.dataset_index_path.exists():
            raise FileNotFoundError(f"Dataset index not found at {self.dataset_index_path}")

        self.label_map = self._load_label_map(self.dataset_index_path)
            
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {self.weights_path}")
            
        n_classes = len(self.label_map)
        
        self.model = self._build_model(
            n_classes,
            input_height,
            input_width,
            input_channels,
            dropout_rate,
            dense_units,
        )
        self.model.load_weights(str(self.weights_path))

    def _load_label_map(self, dataset_index_path: Path) -> dict[int, str]:
        df_index = pd.read_parquet(dataset_index_path)
        required_columns = {"label", "scientific_name"}
        missing_columns = required_columns - set(df_index.columns)
        if missing_columns:
            raise ValueError(
                f"dataset_index.parquet is missing required columns: {sorted(missing_columns)}"
            )

        df_index = df_index.copy()
        df_index["label"] = df_index["label"].astype(int)
        label_table = (
            df_index[["label", "scientific_name"]]
            .drop_duplicates(subset=["label"])
            .sort_values("label")
        )
        actual_labels = label_table["label"].astype(int).tolist()
        expected_labels = list(range(len(label_table)))
        if actual_labels != expected_labels:
            raise ValueError(
                "Labels in dataset_index.parquet must be contiguous starting at 0. "
                f"Expected {expected_labels[:5]}... got {actual_labels[:5]}..."
            )

        return {
            int(row.label): row.scientific_name
            for row in label_table.itertuples(index=False)
        }

    def _build_model(
        self,
        n_classes,
        input_height,
        input_width,
        input_channels,
        dropout_rate,
        dense_units,
    ):
        inputs = tf.keras.Input(
            shape=(input_height, input_width, input_channels),
            name="spectrogram",
        )
        
        if input_channels == 1:
            x = tf.keras.layers.Concatenate(axis=-1, name="to_rgb")([inputs, inputs, inputs])
        elif input_channels == 3:
            x = inputs
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {input_channels}")
        
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=None,
            input_shape=(input_height, input_width, 3),
        )
        base_model.trainable = False
        x = base_model(x, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="predictions")(x)

        return tf.keras.Model(inputs, outputs, name="BirdSongClassifier")

    def predict_top3(self, tensors: np.ndarray) -> str:
        """
        Receives a tensor (or multiple tensors) of spectrograms,
        runs inference, averages the predictions across chunks,
        and returns a JSON string with the top 3 predicted species.
        
        Expected shape of tensors: (K, 128, 431, 1), or without the channel
        dimension for callers that still pass (K, 128, 431).
        """
        tensors = np.asarray(tensors, dtype=np.float32)
        if tensors.ndim == 2:
            tensors_in = tensors[np.newaxis, :, :, np.newaxis]
        elif tensors.ndim == 3:
            if tensors.shape[-1] == self.input_shape[-1] and tensors.shape[:2] == self.input_shape[:2]:
                tensors_in = tensors[np.newaxis, :, :, :]
            else:
                tensors_in = tensors[:, :, :, np.newaxis]
        elif tensors.ndim == 4:
            tensors_in = tensors
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D tensors, got shape={tensors.shape}")

        if tuple(tensors_in.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected tensor shape (*, {self.input_shape}), got {tensors_in.shape}"
            )
        
        chunk_probs = self.model.predict(tensors_in, verbose=0)  # (K, N_CLASSES)
        avg_probs = chunk_probs.mean(axis=0)  # average across chunks

        top3_indices = np.argsort(avg_probs)[::-1][:3]
        
        results = []
        for rank, idx in enumerate(top3_indices, 1):
            species = self.label_map[int(idx)].replace("_", " ")
            confidence = float(avg_probs[idx])
            results.append({
                "rank": rank,
                "species": species,
                "confidence": confidence
            })
            
        return json.dumps(results, indent=2)
