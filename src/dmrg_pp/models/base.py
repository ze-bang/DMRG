"""Common base class for physical models shipped with DMRG++."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from dmrg_pp.operators.mpo import MPO
from dmrg_pp.states.mps import MPS

__all__ = ["LatticeModel"]


class LatticeModel(ABC):
    """Minimal protocol every shipped model must satisfy."""

    L: int
    physical_dims: tuple[int, ...]

    @abstractmethod
    def mpo(self) -> MPO:
        """Return the Hamiltonian MPO."""

    def initial_state(self, *, seed: int | None = None) -> MPS:
        """Default starting state for DMRG (override for symmetry-targeted seeds)."""
        from dmrg_pp.states.random import random_mps

        return random_mps(self.physical_dims, bond_dim=4, seed=seed)

    @abstractmethod
    def __repr__(self) -> str: ...

    # convenience alias for sub-classes that want a simple uniform lattice
    @staticmethod
    def _uniform_dims(L: int, d: int) -> tuple[int, ...]:
        return tuple([d] * L)

    @staticmethod
    def _validate_chain(L: int) -> None:
        if L < 2:
            raise ValueError(f"chain length must be >= 2, got {L}")

    @staticmethod
    def _per_bond(value: float | Sequence[float], n_bonds: int, name: str) -> list[float]:
        """Promote a scalar coupling to a per-bond list, or validate a sequence."""
        if isinstance(value, (int, float, complex)):
            return [float(value)] * n_bonds
        out = [float(x) for x in value]
        if len(out) != n_bonds:
            raise ValueError(f"{name} expected {n_bonds} bonds, got {len(out)}")
        return out
