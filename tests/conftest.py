"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260423)


def exact_ground_state(H_dense: np.ndarray) -> tuple[float, np.ndarray]:
    """Smallest eigenpair of a Hermitian matrix, used as an oracle in tests."""
    H_sym = 0.5 * (H_dense + H_dense.conj().T)
    evals, evecs = np.linalg.eigh(H_sym)
    return float(evals[0]), evecs[:, 0]
