"""Tests for BirdSongClassifier.predict_top3 contract."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.model import BirdSongClassifier


def _make_classifier(n_classes: int = 125) -> BirdSongClassifier:
    """Classifier stub with label map and mocked Keras model (no weights on disk)."""
    clf = BirdSongClassifier.__new__(BirdSongClassifier)
    clf.label_map = {i: f"Species_{i}" for i in range(n_classes)}
    clf.model = MagicMock()
    return clf


def test_predict_top3_returns_three_candidates() -> None:
    clf = _make_classifier()
    probs = np.zeros(125, dtype=np.float32)
    probs[0], probs[1], probs[2] = 0.5, 0.3, 0.2
    clf.model.predict.return_value = np.tile(probs, (2, 1))

    result = json.loads(
        clf.predict_top3(np.zeros((2, 128, 188), dtype=np.float32))
    )

    assert len(result) == 3
    assert result[0]["species"] == "Species 0"
    assert result[1]["species"] == "Species 1"
    assert result[2]["species"] == "Species 2"
    assert all(np.isfinite(r["confidence"]) for r in result)
    assert result[0]["confidence"] == pytest.approx(0.5)
    assert result[0]["rank"] == 1


def test_predict_top3_single_chunk_still_returns_three() -> None:
    clf = _make_classifier()
    probs = np.linspace(0, 1, 125, dtype=np.float32)
    probs /= probs.sum()
    clf.model.predict.return_value = probs.reshape(1, -1)

    result = json.loads(
        clf.predict_top3(np.zeros((128, 188), dtype=np.float32))
    )

    assert len(result) == 3


def test_predict_top3_zero_chunks_raises() -> None:
    clf = _make_classifier()

    with pytest.raises(ValueError, match="no spectrogram chunks"):
        clf.predict_top3(np.zeros((0, 128, 188), dtype=np.float32))


def test_predict_top3_invalid_probabilities_raises() -> None:
    clf = _make_classifier()
    clf.model.predict.return_value = np.full((1, 125), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="no valid probabilities"):
        clf.predict_top3(np.zeros((1, 128, 188), dtype=np.float32))
