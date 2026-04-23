"""Tests for the MPS class."""

from __future__ import annotations

import numpy as np
import pytest

from dmrg_pp.states.mps import MPS
from dmrg_pp.states.random import random_mps


def test_random_mps_shape():
    psi = random_mps([2] * 8, bond_dim=10, seed=0)
    assert psi.L == 8
    assert psi.physical_dims == (2,) * 8
    bonds = psi.bond_dims
    assert bonds[0] == 1 and bonds[-1] == 1
    assert max(bonds) <= 10


def test_random_mps_normalized():
    psi = random_mps([2] * 6, bond_dim=8, seed=1)
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)


def test_orthogonalize_preserves_state():
    psi = random_mps([2] * 6, bond_dim=8, seed=2)
    v0 = psi.to_state_vector()
    for c in [0, 2, 5]:
        psi.orthogonalize(c)
        assert psi.centre == c
        v = psi.to_state_vector()
        np.testing.assert_allclose(v, v0, atol=1e-10)


def test_overlap_with_self_equals_norm_sq():
    psi = random_mps([2] * 5, bond_dim=4, seed=3)
    o = psi.overlap(psi)
    assert o.real == pytest.approx(1.0, abs=1e-12)
    assert abs(o.imag) < 1e-12


def test_truncate_does_not_blow_up():
    psi = random_mps([2] * 8, bond_dim=16, seed=4)
    v0 = psi.to_state_vector()
    psi.truncate(max_bond=4)
    assert psi.max_bond <= 4
    # truncated state still close to a normalized state
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    # overlap with original is meaningful (not zero)
    v = psi.to_state_vector()
    assert abs(np.vdot(v, v0)) > 0.0


def test_product_state_constructor():
    psi = MPS.from_product_state([0, 1, 0, 1])
    v = psi.to_state_vector()
    # bit pattern 0101 at positions (sigma0, sigma1, sigma2, sigma3) → index 5
    expected = np.zeros(16, dtype=complex)
    expected[0b0101] = 1.0
    np.testing.assert_allclose(v, expected, atol=1e-12)
