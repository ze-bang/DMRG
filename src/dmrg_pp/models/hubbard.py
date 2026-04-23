"""One-band Hubbard chain (open boundary).

Hamiltonian::

    H = -t Σ_{i, σ} (c†_{i,σ} c_{i+1,σ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
        - μ Σ_i (n_{i↑} + n_{i↓})

The hopping is implemented with the Jordan–Wigner string ``F`` between the
creation and annihilation operators to enforce fermionic anticommutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dmrg_pp.models.base import LatticeModel
from dmrg_pp.operators.builder import MPOBuilder
from dmrg_pp.operators.local_ops import fermion_ops
from dmrg_pp.operators.mpo import MPO

__all__ = ["HubbardChain"]


@dataclass
class HubbardChain(LatticeModel):
    L: int
    t: float = 1.0
    U: float = 4.0
    mu: float = 0.0
    physical_dims: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_chain(self.L)
        self.physical_dims = self._uniform_dims(self.L, 4)

    def mpo(self) -> MPO:
        # Build Jordan–Wigner-decorated nearest-neighbour hops.  We define
        #   c_iσ → (∏_{k<i} F_k) c_iσ
        # so the bilinear c†_{i σ} c_{j σ} (j = i+1) equals
        #   c†_{i σ} F_i c_{j σ}      (with the leftover F absorbed as below)
        ops = fermion_ops()
        # Pre-build F-decorated hopping operators so the MPOBuilder sees a
        # *single* operator name per site, keeping the bond dimension minimal.
        ext = dict(ops)
        ext["Cup_dag_F"] = ops["Cup_dag"] @ ops["F"]
        ext["Cdn_dag_F"] = ops["Cdn_dag"] @ ops["F"]
        ext["F_Cup"] = ops["F"] @ ops["Cup"]
        ext["F_Cdn"] = ops["F"] @ ops["Cdn"]

        b = MPOBuilder(L=self.L, local_ops=ext)

        for i in range(self.L - 1):
            # -t (c†_iσ c_jσ + h.c.) with JW string absorbed into the *left* site:
            # c†_iσ c_{i+1}σ = (c̃†_iσ F_i) ⊗ c̃_{i+1}σ
            # c†_{i+1}σ c_iσ = (F_i c̃_iσ) ⊗ c̃†_{i+1}σ
            # Spin-up
            b.add(-self.t, (i, "Cup_dag_F"), (i + 1, "Cup"))
            b.add(-self.t, (i, "F_Cup"), (i + 1, "Cup_dag"))
            # Spin-down
            b.add(-self.t, (i, "Cdn_dag_F"), (i + 1, "Cdn"))
            b.add(-self.t, (i, "F_Cdn"), (i + 1, "Cdn_dag"))

        for i in range(self.L):
            if self.U != 0.0:
                b.add(self.U, (i, "NupNdn"))
            if self.mu != 0.0:
                b.add(-self.mu, (i, "N"))

        return b.build()

    def __repr__(self) -> str:
        return f"HubbardChain(L={self.L}, t={self.t}, U={self.U}, mu={self.mu})"
