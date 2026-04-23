"""Tests for low-level tensor primitives."""

from __future__ import annotations

import numpy as np
import pytest

from dmrg_pp.tensors.linalg import left_qr, right_qr, truncated_svd


def test_truncated_svd_recovers_matrix(rng):
    A = rng.standard_normal((20, 30))
    U, s, Vh, info = truncated_svd(A, max_bond=20)
    rec = U @ np.diag(s) @ Vh
    np.testing.assert_allclose(rec, A, atol=1e-10)
    assert info.bond_dim == 20
    assert info.truncation_error == pytest.approx(0.0, abs=1e-20)


def test_truncated_svd_caps_bond(rng):
    A = rng.standard_normal((40, 40))
    U, s, Vh, info = truncated_svd(A, max_bond=10)
    assert U.shape == (40, 10)
    assert s.shape == (10,)
    assert Vh.shape == (10, 40)
    # discarded weight equals sum of squared discarded singular values
    full = np.linalg.svd(A, compute_uv=False)
    expected_err = float(np.sum(full[10:] ** 2))
    assert info.truncation_error == pytest.approx(expected_err, rel=1e-10)


def test_truncated_svd_cutoff(rng):
    s_true = np.array([10.0, 5.0, 1e-3, 1e-6, 1e-12])
    U_true, _ = np.linalg.qr(rng.standard_normal((6, 5)))
    V_true, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    A = U_true @ np.diag(s_true) @ V_true.T

    # Zero cutoff: every singular value retained.
    _, s, _, info = truncated_svd(A, cutoff=0.0)
    assert info.bond_dim == 5
    np.testing.assert_allclose(s, s_true, atol=1e-10)

    # Looser cutoff allows dropping every singular value below ≈1e-3:
    _, s, _, info = truncated_svd(A, cutoff=1e-6)
    assert info.bond_dim == 2
    np.testing.assert_allclose(s, s_true[:2], atol=1e-10)


def test_truncated_svd_no_cutoff_keeps_all(rng):
    A = rng.standard_normal((10, 8))
    _, _, _, info = truncated_svd(A, cutoff=0.0)
    assert info.bond_dim == 8
    assert info.truncation_error == 0.0


def test_truncated_svd_renormalize(rng):
    A = rng.standard_normal((30, 30))
    A /= np.linalg.norm(A)
    _U, s, _Vh, _info = truncated_svd(A, max_bond=5, renormalize=True)
    assert np.linalg.norm(s) == pytest.approx(1.0, abs=1e-12)


def test_left_qr_isometric(rng):
    A = rng.standard_normal((25, 10))
    Q, R = left_qr(A)
    assert Q.shape == (25, 10)
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(10), atol=1e-10)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)
    assert np.all(np.diag(R) >= 0)


def test_right_qr_isometric(rng):
    A = rng.standard_normal((10, 25))
    L, Q = right_qr(A)
    assert Q.shape == (10, 25)
    np.testing.assert_allclose(Q @ Q.conj().T, np.eye(10), atol=1e-10)
    np.testing.assert_allclose(L @ Q, A, atol=1e-10)
