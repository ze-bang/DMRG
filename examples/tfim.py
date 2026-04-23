"""Quantum-critical TFIM: ground state at the Ising critical point ``g = J``.

We compute the bond entropies and verify the central-charge prediction from
conformal field theory: at the critical point, the bipartite entropy of an
open chain follows ``S(b) = c/6 · log[(2L/π) sin(π b / L)] + const`` with
``c = 1/2`` for the Ising universality class.
"""

from __future__ import annotations

import numpy as np

from dmrg_pp import DMRGConfig, run_dmrg
from dmrg_pp.measurements import bond_entropies
from dmrg_pp.models import TransverseFieldIsing
from dmrg_pp.utils.logging import setup_logging


def main() -> None:
    setup_logging("INFO")
    L = 32
    model = TransverseFieldIsing(L=L, J=1.0, g=1.0)
    cfg = DMRGConfig(n_sweeps=10, max_bond=[16, 32, 64, 128, 128], seed=42)
    res = run_dmrg(model.mpo(), config=cfg)
    print(f"E = {res.energy:.12f}, E/L = {res.energy / L:.12f}")

    S = bond_entropies(res.mps)
    bs = np.arange(1, L)
    chord = (2.0 * L / np.pi) * np.sin(np.pi * bs / L)
    # Linear fit S(b) = (c/6) log(chord) + const
    A = np.vstack([np.log(chord), np.ones_like(chord)]).T
    slope, _ = np.linalg.lstsq(A, S[1:L], rcond=None)[0]
    print(f"Central-charge fit: c = 6 · slope = {6 * slope:.4f}  (expected 0.5)")


if __name__ == "__main__":
    main()
