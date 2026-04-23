"""Left/right MPO–MPS environments with O(L) caching for DMRG sweeps.

Definitions
-----------
For an MPS ``|ψ⟩`` and an MPO ``Ĥ``, the *left environment* at bond ``i`` is

    L_i  = ⟨ψ|_{<i} Ĥ_{<i} |ψ⟩_{<i}

a tensor of shape ``(D_a, D_o, D_a)`` indexed by ``(bra MPS bond, MPO bond,
ket MPS bond)``.  ``L_0`` is the unit scalar ``[[[1]]]``.  Right environments
``R_i`` are the mirror image, indexed in the same order.  These environments
allow the local effective Hamiltonian for sites ``i, i+1`` to be applied in
``O(D³ d² D_o + D² d² D_o²)`` time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.operators.mpo import MPO
from dmrg_pp.states.mps import MPS

__all__ = ["EnvironmentCache", "build_left_env", "build_right_env"]


def build_left_env(L_prev: NDArray, A: NDArray, W: NDArray) -> NDArray:
    """Extend a left environment by one site.

    Parameters
    ----------
    L_prev:
        Existing environment with shape ``(D_a, D_o, D_a)``.
    A:
        MPS site tensor, shape ``(D_a, d, D_a')``.
    W:
        MPO site tensor, shape ``(D_o, d, d, D_o')``.
    """
    # contract: (D_a, D_o, D_a) · (D_a, d, D_a') → (D_o, D_a, d, D_a')
    tmp = np.tensordot(L_prev, A, axes=([2], [0]))  # (D_a, D_o, d, D_a')
    # contract MPO: (D_a, D_o, d, D_a') · (D_o, d, d, D_o') over (D_o, d)
    tmp = np.tensordot(tmp, W, axes=([1, 2], [0, 2]))  # (D_a, D_a', d, D_o')
    # close on bra: (D_a, D_a', d, D_o') · (D_a, d, D_a'')* over (D_a, d)
    tmp = np.tensordot(tmp, A.conj(), axes=([0, 2], [0, 1]))  # (D_a', D_o', D_a'')
    return tmp.transpose(2, 1, 0)  # (D_a'', D_o', D_a')


def build_right_env(R_prev: NDArray, A: NDArray, W: NDArray) -> NDArray:
    """Extend a right environment by one site (toward the left).

    Returns a tensor with the same ``(bra, MPO, ket)`` index ordering as
    :func:`build_left_env`, so both kinds of environment can be combined
    symmetrically inside the effective-Hamiltonian routine.
    """
    # R_prev: (D_a_bra', D_o', D_a_ket')
    tmp = np.tensordot(A, R_prev, axes=([2], [2]))  # (D_a_ket, d, D_a_bra', D_o')
    tmp = np.tensordot(tmp, W, axes=([1, 3], [2, 3]))  # (D_a_ket, D_a_bra', D_o, d)
    tmp = np.tensordot(tmp, A.conj(), axes=([1, 3], [2, 1]))  # (D_a_ket, D_o, D_a_bra)
    return tmp.transpose(2, 1, 0)  # (D_a_bra, D_o, D_a_ket)


class EnvironmentCache:
    """O(L)-storage left/right environment cache for two-site DMRG.

    Indexing convention (length-``L+1`` arrays):

    * ``L_envs[k]``  contracts sites ``0..k-1`` and is exposed at the *left*
      bond of site ``k``.  Boundary ``L_envs[0]`` is the trivial ``1×1×1`` env.
    * ``R_envs[k]``  contracts sites ``k..L-1`` and is exposed at the *left*
      bond of site ``k`` (= right bond of site ``k-1``).  Boundary
      ``R_envs[L]`` is the trivial ``1×1×1`` env.

    The two-site update at bond ``(i, i+1)`` therefore needs
    ``L_envs[i]``, ``R_envs[i+2]``, ``W[i]``, and ``W[i+1]``.

    Sweeping right (left) shifts the orthogonality centre by one and updates
    the affected environment slot in O(D³ d D_o + D² d² D_o²) work.
    """

    def __init__(self, psi: MPS, mpo: MPO) -> None:
        if psi.L != mpo.L:
            raise ValueError(f"length mismatch: psi.L={psi.L} mpo.L={mpo.L}")
        if psi.physical_dims != mpo.physical_dims:
            raise ValueError("physical dimensions of MPS and MPO disagree")
        self.psi = psi
        self.mpo = mpo
        self.L_envs: list[NDArray | None] = [None] * (psi.L + 1)
        self.R_envs: list[NDArray | None] = [None] * (psi.L + 1)
        self._init_boundary()

    def _init_boundary(self) -> None:
        dtype = np.result_type(self.psi.dtype, self.mpo.dtype)
        self.L_envs[0] = np.ones((1, 1, 1), dtype=dtype)
        self.R_envs[self.psi.L] = np.ones((1, 1, 1), dtype=dtype)

    def build_all_right(self) -> None:
        """Populate every right environment by sweeping from the right boundary."""
        for i in range(self.psi.L - 1, -1, -1):
            r_next = self.R_envs[i + 1]
            assert r_next is not None
            self.R_envs[i] = build_right_env(r_next, self.psi[i], self.mpo[i])

    def update_left(self, i: int) -> None:
        """Recompute ``L[i+1]`` from ``L[i]`` and the (possibly updated) site ``i``."""
        prev = self.L_envs[i]
        assert prev is not None, f"left env at {i} not yet built"
        self.L_envs[i + 1] = build_left_env(prev, self.psi[i], self.mpo[i])

    def update_right(self, i: int) -> None:
        """Recompute ``R[i]`` from ``R[i+1]`` and the (possibly updated) site ``i``."""
        nxt = self.R_envs[i + 1]
        assert nxt is not None, f"right env at {i + 1} not yet built"
        self.R_envs[i] = build_right_env(nxt, self.psi[i], self.mpo[i])

    # convenience accessors
    def left(self, i: int) -> NDArray:
        env = self.L_envs[i]
        if env is None:
            raise RuntimeError(f"left env at {i} not built yet")
        return env

    def right(self, i: int) -> NDArray:
        env = self.R_envs[i]
        if env is None:
            raise RuntimeError(f"right env at {i} not built yet")
        return env
