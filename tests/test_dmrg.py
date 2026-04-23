"""End-to-end DMRG tests: convergence to exact diagonalization on small systems."""

from __future__ import annotations

import pytest

from dmrg_pp import DMRGConfig, run_dmrg
from dmrg_pp.measurements import bond_entropies, local_expectations
from dmrg_pp.models import HeisenbergXXZ, HubbardChain, TransverseFieldIsing
from dmrg_pp.operators.local_ops import spin_half_ops
from tests.conftest import exact_ground_state


@pytest.mark.integration
def test_dmrg_heisenberg_matches_exact_L8():
    L = 8
    model = HeisenbergXXZ(L=L, Jxy=1.0, Jz=1.0)
    mpo = model.mpo()
    H = mpo._to_dense_explicit()
    e_exact, _ = exact_ground_state(H)

    cfg = DMRGConfig(n_sweeps=8, max_bond=64, cutoff=1e-12, seed=0)
    res = run_dmrg(mpo, config=cfg)
    assert res.energy == pytest.approx(e_exact, abs=1e-8)


@pytest.mark.integration
def test_dmrg_xxz_anisotropic_matches_exact_L6():
    L = 6
    model = HeisenbergXXZ(L=L, Jxy=0.7, Jz=1.5, h=0.2)
    mpo = model.mpo()
    H = mpo._to_dense_explicit()
    e_exact, _ = exact_ground_state(H)

    cfg = DMRGConfig(n_sweeps=10, max_bond=64, cutoff=1e-12, seed=1)
    res = run_dmrg(mpo, config=cfg)
    assert res.energy == pytest.approx(e_exact, abs=1e-8)


@pytest.mark.integration
def test_dmrg_tfim_off_critical_matches_exact_L8():
    L = 8
    model = TransverseFieldIsing(L=L, J=1.0, g=0.5)
    mpo = model.mpo()
    H = mpo._to_dense_explicit()
    e_exact, _ = exact_ground_state(H)

    cfg = DMRGConfig(n_sweeps=8, max_bond=32, cutoff=1e-12, seed=2)
    res = run_dmrg(mpo, config=cfg)
    assert res.energy == pytest.approx(e_exact, abs=1e-8)


@pytest.mark.integration
def test_dmrg_tfim_critical_matches_exact_L6():
    L = 6
    model = TransverseFieldIsing(L=L, J=1.0, g=1.0)
    mpo = model.mpo()
    H = mpo._to_dense_explicit()
    e_exact, _ = exact_ground_state(H)

    cfg = DMRGConfig(n_sweeps=10, max_bond=32, cutoff=1e-12, seed=3)
    res = run_dmrg(mpo, config=cfg)
    assert res.energy == pytest.approx(e_exact, abs=1e-8)


@pytest.mark.integration
def test_dmrg_hubbard_small_matches_exact():
    L = 4
    model = HubbardChain(L=L, t=1.0, U=2.0, mu=0.0)
    mpo = model.mpo()
    H = mpo._to_dense_explicit()
    e_exact, _ = exact_ground_state(H)

    cfg = DMRGConfig(n_sweeps=10, max_bond=64, cutoff=1e-12, seed=4)
    res = run_dmrg(mpo, config=cfg)
    assert res.energy == pytest.approx(e_exact, abs=1e-7)


@pytest.mark.integration
def test_dmrg_returns_normalized_state():
    model = HeisenbergXXZ(L=6, Jxy=1.0, Jz=1.0)
    res = run_dmrg(model.mpo(), config=DMRGConfig(n_sweeps=4, max_bond=32, seed=5))
    assert res.mps.norm() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.integration
def test_local_observables_sum_to_zero_for_total_Sz_singlet():
    """The Heisenberg AFM ground state on even L is a singlet (Σ Sᶻ = 0)."""
    L = 8
    res = run_dmrg(
        HeisenbergXXZ(L=L, Jxy=1.0, Jz=1.0).mpo(),
        config=DMRGConfig(n_sweeps=8, max_bond=64, cutoff=1e-12, seed=6),
    )
    sz = local_expectations(res.mps, spin_half_ops()["Sz"])
    assert abs(sz.sum().real) < 1e-6


@pytest.mark.integration
def test_bond_entropies_nonzero_for_entangled_state():
    L = 8
    res = run_dmrg(
        HeisenbergXXZ(L=L, Jxy=1.0, Jz=1.0).mpo(),
        config=DMRGConfig(n_sweeps=6, max_bond=32, seed=7),
    )
    S = bond_entropies(res.mps)
    assert S[L // 2] > 0.1
