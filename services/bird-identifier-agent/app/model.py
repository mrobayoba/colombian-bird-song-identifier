import json
import numpy as np
import tensorflow as tf
from pathlib import Path

class BirdSongClassifier:
    def __init__(self, weights_path: str, label_map_path: str, 
                 input_height=128, input_width=188, dropout_rate=0.4, dense_units=256):
        """
        Initializes the classifier by explicitly rebuilding the ResNet50V2 architecture
        and restoring the state from a .weights.h5 checkpoint file.
        """
        self.weights_path = Path(weights_path)
        self.label_map_path = Path(label_map_path)
        
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found at {self.label_map_path}")
            
        with open(self.label_map_path, "r", encoding="utf-8") as fh:
            self.label_map = {int(k): v for k, v in json.load(fh).items()}
            
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {self.weights_path}")
            
        n_classes = len(self.label_map)
        
        # 1. Build architecture natively
        self.model = self._build_model(n_classes, input_height, input_width, dropout_rate, dense_units)
        # 2. Load the best epoch checkpoint weights
        self.model.load_weights(str(self.weights_path))

    def _build_model(self, n_classes, input_height, input_width, dropout_rate, dense_units):
        inputs = tf.keras.Input(shape=(input_height, input_width, 1), name="spectrogram")
        
        # Grayscale → 3-channel
        x = tf.keras.layers.Concatenate(axis=-1, name="to_rgb")([inputs, inputs, inputs])
        
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=None, # Weights are provided by our checkpoint
            input_shape=(input_height, input_width, 3),
        )
        x = base_model(x, training=False)

        # Classification head
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
        
        Expected shape of tensors: (K, 128, 188) where K is number of chunks.
        """
        # Ensure correct dimensionality
        if len(tensors.shape) == 2:
            tensors = np.expand_dims(tensors, axis=0)  # (1, 128, 188)

        if tensors.shape[0] == 0:
            raise ValueError("no spectrogram chunks to classify")

        tensors_in = np.expand_dims(tensors, axis=-1)  # (K, 128, 188, 1)

        # Run inference
        chunk_probs = self.model.predict(tensors_in, verbose=0)  # (K, N_CLASSES)
        avg_probs = chunk_probs.mean(axis=0)  # average across chunks

        if avg_probs.size == 0 or not np.isfinite(avg_probs).any():
            raise ValueError("model produced no valid probabilities")

        # Get top 3 (or fewer if fewer classes exist)
        top_k = min(3, avg_probs.size)
        top3_indices = np.argsort(avg_probs)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top3_indices, 1):
            species = self.label_map[idx].replace("_", " ")
            confidence = float(avg_probs[idx])
            results.append({
                "rank": rank,
                "species": species,
                "confidence": confidence
            })
            
        return json.dumps(results, indent=2)
