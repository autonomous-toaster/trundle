"""Pure numeric helpers and the batching contract."""

from __future__ import annotations

import numpy as np
import pytest

from trundle.embedder import EmbedderError, batches, normalize, pool, truncate


def test_batches_covers_everything() -> None:
    assert [list(chunk) for chunk in batches(["a", "b", "c", "d", "e"], 2)] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_mean_pooling_ignores_padding() -> None:
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    assert np.allclose(pool(hidden, mask, "mean"), [[2.0, 2.0]])


def test_cls_pooling_takes_the_first_token() -> None:
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    mask = np.array([[1, 1]], dtype=np.int64)
    assert np.allclose(pool(hidden, mask, "cls"), [[1.0, 2.0]])


def test_normalize_gives_unit_vectors() -> None:
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(matrix[0]), 1.0)
    assert np.allclose(matrix[1], 0.0)


def test_truncate_keeps_the_head_and_renormalises() -> None:
    matrix = np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    truncated = truncate(matrix, 2)
    assert truncated.shape == (1, 2)
    assert np.allclose(np.linalg.norm(truncated), 1.0)


def test_truncate_rejects_impossible_dimensions() -> None:
    matrix = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(EmbedderError):
        truncate(matrix, 8)
    with pytest.raises(EmbedderError):
        truncate(matrix, 0)
