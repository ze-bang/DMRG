"""Transverse-field Ising chain.

Hamiltonian::

    H = -J Σ_i σ^z_i σ^z_{i+1} - g Σ_i σ^x_i

with σ^α = 2 S^α (Pauli matrices).  The quantum critical point sits at
``g = J``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dmrg_pp.models.base import LatticeModel
from dmrg_pp.operators.builder import MPOBuilder
from dmrg_pp.operators.local_ops import spin_half_ops
from dmrg_pp.operators.mpo import MPO

__all__ = ["TransverseFieldIsing"]


@dataclass
class TransverseFieldIsing(LatticeModel):
    L: int
    J: float = 1.0
    g: float = 1.0
    physical_dims: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_chain(self.L)
        self.physical_dims = self._uniform_dims(self.L, 2)

    def mpo(self) -> MPO:
        ops = spin_half_ops()
        b = MPOBuilder(L=self.L, local_ops=ops)
        # Use Pauli matrices: σ^α = 2 S^α
        for i in range(self.L - 1):
            b.add(-self.J * 4.0, (i, "Sz"), (i + 1, "Sz"))
        for i in range(self.L):
            b.add(-self.g * 2.0, (i, "Sx"))
        return b.build()

    def __repr__(self) -> str:
        return f"TransverseFieldIsing(L={self.L}, J={self.J}, g={self.g})"
