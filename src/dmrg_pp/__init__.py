"""DMRG++ — production-grade Density Matrix Renormalization Group.

A modular tensor-network library for ground-state and low-lying excited-state
calculations of one-dimensional and quasi-one-dimensional quantum lattice
models, built on Matrix Product States (MPS) and Matrix Product Operators (MPO).

Key entry points
----------------
- :class:`dmrg_pp.MPS`           — Matrix Product State container
- :class:`dmrg_pp.MPO`           — Matrix Product Operator container
- :class:`dmrg_pp.MPOBuilder`    — Symbolic finite-state-machine MPO builder
- :func:`dmrg_pp.run_dmrg`       — High-level two-site DMRG driver
- :class:`dmrg_pp.DMRGConfig`    — Convergence / sweep configuration

Models
------
- :class:`dmrg_pp.models.HeisenbergXXZ`
- :class:`dmrg_pp.models.TransverseFieldIsing`
- :class:`dmrg_pp.models.HubbardChain`

Examples
--------
>>> from dmrg_pp import run_dmrg, DMRGConfig
>>> from dmrg_pp.models import HeisenbergXXZ
>>> model = HeisenbergXXZ(L=20, Jxy=1.0, Jz=1.0)
>>> result = run_dmrg(model.mpo(), config=DMRGConfig(max_bond=64, n_sweeps=6))
>>> result.energy        # doctest: +SKIP
"""

from __future__ import annotations

from dmrg_pp._version import __version__
from dmrg_pp.algorithms.dmrg import DMRGConfig, DMRGResult, run_dmrg
from dmrg_pp.operators.builder import MPOBuilder, OpTerm
from dmrg_pp.operators.mpo import MPO
from dmrg_pp.states.mps import MPS

__all__ = [
    "MPO",
    "MPS",
    "DMRGConfig",
    "DMRGResult",
    "MPOBuilder",
    "OpTerm",
    "__version__",
    "run_dmrg",
]
