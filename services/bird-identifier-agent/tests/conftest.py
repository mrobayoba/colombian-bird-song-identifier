"""Test bootstrap — mock TensorFlow so unit tests run without the full runtime."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

if "tensorflow" not in sys.modules:
    tf_mock = MagicMock()
    tf_mock.keras = MagicMock()
    sys.modules["tensorflow"] = tf_mock
