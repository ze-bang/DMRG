"""Tests for HDF5 persistence and YAML config validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmrg_pp import DMRGConfig, run_dmrg
from dmrg_pp.io.config import load_config
from dmrg_pp.io.hdf5 import load_mps, load_result, save_mps, save_result
from dmrg_pp.models import HeisenbergXXZ
from dmrg_pp.states.random import random_mps


def test_save_load_mps_round_trip(tmp_path: Path):
    psi = random_mps([2] * 6, bond_dim=8, seed=99)
    p = tmp_path / "state.h5"
    save_mps(p, psi)
    loaded = load_mps(p)
    assert loaded.L == psi.L
    np.testing.assert_allclose(loaded.to_state_vector(), psi.to_state_vector(), atol=1e-12)


def test_save_load_result_round_trip(tmp_path: Path):
    model = HeisenbergXXZ(L=6, Jxy=1.0, Jz=1.0)
    res = run_dmrg(model.mpo(), config=DMRGConfig(n_sweeps=3, max_bond=16, seed=1))
    p = tmp_path / "res.h5"
    save_result(p, res)
    loaded = load_result(p)
    assert loaded.energy == pytest.approx(res.energy, abs=1e-12)
    assert len(loaded.sweeps) == len(res.sweeps)
    np.testing.assert_allclose(loaded.mps.to_state_vector(), res.mps.to_state_vector(), atol=1e-10)


def test_load_config_from_yaml(tmp_path: Path):
    cfg_text = """
name: test_run
model:
  kind: heisenberg
  L: 8
  Jxy: 1.0
  Jz: 0.5
runtime:
  n_sweeps: 3
  max_bond: 16
  seed: 0
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert cfg.name == "test_run"
    assert cfg.model.kind == "heisenberg"
    assert cfg.model.L == 8
    assert cfg.runtime.n_sweeps == 3
    model = cfg.model.build()
    assert model.L == 8
