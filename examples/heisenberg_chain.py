"""Ground state of the spin-½ Heisenberg chain via DMRG.

Reproduces, in a few seconds, the famous Bethe-ansatz energy density
``e_∞ = 1/4 - ln 2 ≈ -0.4431…`` for an L = 40 antiferromagnetic chain with
isotropic couplings.
"""

from __future__ import annotations

import math

from dmrg_pp import DMRGConfig, run_dmrg
from dmrg_pp.measurements import bond_entropies, local_expectations
from dmrg_pp.models import HeisenbergXXZ
from dmrg_pp.operators.local_ops import spin_half_ops
from dmrg_pp.utils.logging import setup_logging


def main() -> None:
    setup_logging("INFO")
    L = 32
    model = HeisenbergXXZ(L=L, Jxy=1.0, Jz=1.0)
    mpo = model.mpo()
    config = DMRGConfig(
        n_sweeps=8,
        max_bond=[16, 32, 64, 64, 96],
        cutoff=1e-10,
        e_tol=1e-8,
        seed=2026,
    )
    result = run_dmrg(mpo, config=config)

    bethe_e_density = 0.25 - math.log(2.0)
    print(f"\nFinal energy:        E       = {result.energy:.12f}")
    print(f"Per site:            E / L   = {result.energy / L:.12f}")
    print(f"Bethe ansatz e_inf:           = {bethe_e_density:.12f}")

    sz = local_expectations(result.mps, spin_half_ops()["Sz"])
    print(f"Total magnetisation: <Sz_total> = {sz.sum().real:+.3e}")

    entropies = bond_entropies(result.mps)
    print(f"Mid-chain entropy:   S(L/2)     = {entropies[L // 2]:.4f}")


if __name__ == "__main__":
    main()
