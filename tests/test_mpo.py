"""Tests for the MPO class and the symbolic builder."""

from __future__ import annotations

import numpy as np
import pytest

from dmrg_pp.models import HeisenbergXXZ, TransverseFieldIsing
from dmrg_pp.operators.builder import MPOBuilder
from dmrg_pp.operators.local_ops import spin_half_ops


def _kron_n(ops):
    out = np.array([[1.0]])
    for o in ops:
        out = np.kron(out, o)
    return out


def test_builder_single_site_term():
    ops = spin_half_ops()
    L = 4
    b = MPOBuilder(L=L, local_ops=ops)
    b.add(2.5, (1, "Sz"))
    mpo = b.build()
    Id, Sz = ops["Id"], ops["Sz"]
    expected = 2.5 * _kron_n([Id, Sz, Id, Id])
    np.testing.assert_allclose(mpo._to_dense_explicit(), expected, atol=1e-12)


def test_builder_nn_term():
    ops = spin_half_ops()
    L = 4
    b = MPOBuilder(L=L, local_ops=ops)
    b.add(0.7, (1, "Sx"), (2, "Sx"))
    mpo = b.build()
    Id, Sx = ops["Id"], ops["Sx"]
    expected = 0.7 * _kron_n([Id, Sx, Sx, Id])
    np.testing.assert_allclose(mpo._to_dense_explicit(), expected, atol=1e-12)


def test_builder_long_range_term():
    ops = spin_half_ops()
    L = 5
    b = MPOBuilder(L=L, local_ops=ops)
    b.add(1.3, (0, "Sz"), (3, "Sz"))
    mpo = b.build()
    Id, Sz = ops["Id"], ops["Sz"]
    expected = 1.3 * _kron_n([Sz, Id, Id, Sz, Id])
    np.testing.assert_allclose(mpo._to_dense_explicit(), expected, atol=1e-12)


def test_heisenberg_dense_matches_handbuilt():
    L = 4
    Jxy, Jz = 1.0, 0.7
    h = 0.3
    model = HeisenbergXXZ(L=L, Jxy=Jxy, Jz=Jz, h=h)
    mpo = model.mpo()
    H_mpo = mpo._to_dense_explicit()

    ops = spin_half_ops()
    Id, _Sx, _Sy, Sz = ops["Id"], ops["Sx"], ops["Sy"], ops["Sz"]
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    H_ref = np.zeros((2**L, 2**L), dtype=complex)
    for i in range(L - 1):
        for op_pair, coeff in [
            ((Sp, Sm), 0.5 * Jxy),
            ((Sm, Sp), 0.5 * Jxy),
            ((Sz, Sz), Jz),
        ]:
            mats = [Id] * L
            mats[i], mats[i + 1] = op_pair
            H_ref += coeff * _kron_n(mats)
    for i in range(L):
        mats = [Id] * L
        mats[i] = Sz
        H_ref += h * _kron_n(mats)
    np.testing.assert_allclose(H_mpo, H_ref, atol=1e-10)
    # Hermitian (real-eigenvalue check)
    np.testing.assert_allclose(H_mpo, H_mpo.conj().T, atol=1e-10)


def test_tfim_optimal_bond_dim():
    """Nearest-neighbour 2-body Hamiltonian should give D ≤ 5."""
    model = TransverseFieldIsing(L=20, J=1.0, g=0.5)
    mpo = model.mpo()
    assert mpo.max_bond <= 5


def test_mpo_expectation_matches_dense():
    L = 4
    model = HeisenbergXXZ(L=L, Jxy=1.0, Jz=1.0)
    mpo = model.mpo()
    H = mpo._to_dense_explicit()
    rng = np.random.default_rng(7)
    psi_vec = rng.standard_normal(2**L) + 1j * rng.standard_normal(2**L)
    psi_vec /= np.linalg.norm(psi_vec)
    # Build a random product MPS by reshaping (won't have low bond dim, but works for testing)
    from dmrg_pp.states.random import random_mps

    psi = random_mps([2] * L, bond_dim=8, seed=11, dtype=np.complex128)
    psi_vec = psi.to_state_vector()
    psi_vec /= np.linalg.norm(psi_vec)

    e_dense = float(np.real(np.vdot(psi_vec, H @ psi_vec)))
    e_mpo = mpo.expectation(psi).real
    assert e_mpo == pytest.approx(e_dense, abs=1e-10)
