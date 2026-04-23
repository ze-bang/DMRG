"""Spin-½ XXZ Heisenberg chain.

Hamiltonian (with optional staggered / external field ``h``)::

    H = Σ_i [ Jxy/2 (S+_i S-_{i+1} + h.c.) + Jz S^z_i S^z_{i+1} ] + Σ_i h_i S^z_i
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from dmrg_pp.models.base import LatticeModel
from dmrg_pp.operators.builder import MPOBuilder
from dmrg_pp.operators.local_ops import spin_half_ops
from dmrg_pp.operators.mpo import MPO

__all__ = ["HeisenbergXXZ"]


@dataclass
class HeisenbergXXZ(LatticeModel):
    """Spin-½ XXZ chain with arbitrary couplings and a longitudinal field."""

    L: int
    Jxy: float | Sequence[float] = 1.0
    Jz: float | Sequence[float] = 1.0
    h: float | Sequence[float] = 0.0
    periodic: bool = False
    physical_dims: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_chain(self.L)
        self.physical_dims = self._uniform_dims(self.L, 2)

    def mpo(self) -> MPO:
        ops = spin_half_ops()
        n_bonds = self.L if self.periodic else self.L - 1
        if self.periodic:
            raise NotImplementedError(
                "Periodic boundary conditions inflate the MPO bond dimension and are "
                "not supported by the FSM builder; use OBC and consider Lieb–Mattis-style "
                "wrapping if you need PBC."
            )
        Jxy = self._per_bond(self.Jxy, n_bonds, "Jxy")
        Jz = self._per_bond(self.Jz, n_bonds, "Jz")
        if isinstance(self.h, (int, float)):
            h = [float(self.h)] * self.L
        else:
            h = [float(x) for x in self.h]
            if len(h) != self.L:
                raise ValueError(f"h expected {self.L} sites, got {len(h)}")

        b = MPOBuilder(L=self.L, local_ops=ops)

        for i in range(n_bonds):
            j = (i + 1) % self.L
            if Jxy[i] != 0.0:
                b.add(0.5 * Jxy[i], (i, "Sp"), (j, "Sm"))
                b.add(0.5 * Jxy[i], (i, "Sm"), (j, "Sp"))
            if Jz[i] != 0.0:
                b.add(Jz[i], (i, "Sz"), (j, "Sz"))

        for i, hi in enumerate(h):
            if hi != 0.0:
                b.add(hi, (i, "Sz"))

        return b.build()

    def __repr__(self) -> str:
        return (
            f"HeisenbergXXZ(L={self.L}, Jxy={self.Jxy}, Jz={self.Jz}, "
            f"h={self.h}, periodic={self.periodic})"
        )
