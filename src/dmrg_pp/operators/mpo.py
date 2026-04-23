"""Matrix Product Operator (MPO) container.

Conventions
-----------
For an ``L``-site MPO each tensor ``W[i]`` has shape

    W[i].shape == (D^O_i, d_i, d_i, D^O_{i+1})

with axis ordering ``(left bond, physical out, physical in, right bond)`` and
boundary bonds ``D^O_0 = D^O_L = 1``.  The acting convention is

    (Ô |ψ⟩)_{σ_0 … σ_{L-1}} = Σ_{τ_0 … τ_{L-1}} W^{σ_0}_{1, τ_0} … W^{σ_{L-1}}_{1, τ_{L-1}} ψ_{τ_0 … τ_{L-1}}.

That is, the *first* physical leg is the "ket" (output) index and the second
is the "bra" (input) index, matching the ``A_{i j} ψ_j = ψ'_i`` convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.states.mps import MPS

__all__ = ["MPO"]


class MPO:
    """Finite-system Matrix Product Operator with O(L) bond bookkeeping."""

    __slots__ = ("_dtype", "_tensors")

    def __init__(self, tensors: Sequence[NDArray], *, copy: bool = True) -> None:
        if len(tensors) == 0:
            raise ValueError("MPO must contain at least one site")
        ts: list[NDArray] = []
        prev_right = 1
        for i, w in enumerate(tensors):
            if w.ndim != 4:
                raise ValueError(f"site {i}: MPO tensor must be rank-4, got {w.ndim}")
            if w.shape[1] != w.shape[2]:
                raise ValueError(
                    f"site {i}: physical legs must match, got {w.shape[1]} vs {w.shape[2]}"
                )
            if w.shape[0] != prev_right:
                raise ValueError(
                    f"site {i}: left MPO bond {w.shape[0]} != previous right bond {prev_right}"
                )
            ts.append(np.array(w, copy=copy))
            prev_right = w.shape[3]
        if prev_right != 1:
            raise ValueError(f"final right MPO bond must be 1, got {prev_right}")
        self._tensors: list[NDArray] = ts
        self._dtype = np.result_type(*[t.dtype for t in ts])

    def __len__(self) -> int:
        return len(self._tensors)

    def __iter__(self) -> Iterable[NDArray]:
        return iter(self._tensors)

    def __getitem__(self, i: int) -> NDArray:
        return self._tensors[i]

    @property
    def L(self) -> int:
        return len(self._tensors)

    @property
    def physical_dims(self) -> tuple[int, ...]:
        return tuple(int(t.shape[1]) for t in self._tensors)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        dims = [int(self._tensors[0].shape[0])]
        for t in self._tensors:
            dims.append(int(t.shape[3]))
        return tuple(dims)

    @property
    def max_bond(self) -> int:
        return max(self.bond_dims)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # ----------------------------------------------------- expectation values
    def expectation(self, psi: MPS) -> complex:
        """Compute ``⟨ψ|Ô|ψ⟩``.

        Uses the standard left-to-right contraction with environment shape
        ``(D_bra, D_O, D_ket)``, costing ``O(L · D² · D_O · d²)``.
        """
        if psi.L != self.L:
            raise ValueError(f"length mismatch: MPO L={self.L}, MPS L={psi.L}")
        if psi.physical_dims != self.physical_dims:
            raise ValueError("physical dimensions do not match")
        env: NDArray = np.ones((1, 1, 1), dtype=np.result_type(self.dtype, psi.dtype))
        for A, W in zip(psi, self._tensors, strict=True):
            # env: (Da, Do, Da)  A: (Da, d, Da')  W: (Do, d, d, Do')
            env = np.tensordot(env, A, axes=([2], [0]))  # (Da, Do, d, Da')
            env = np.tensordot(env, W, axes=([1, 2], [0, 2]))  # (Da, Da', d, Do')
            env = np.tensordot(env, A.conj(), axes=([0, 2], [0, 1]))  # (Da', Do', Da'')
            env = env.transpose(2, 1, 0)  # (Da'', Do', Da')
        return complex(env.reshape(()))

    # ------------------------------------------------------- dense fallback
    def to_dense(self) -> NDArray:
        """Materialize the operator as a dense ``(D, D)`` matrix.

        Intended for tests on small systems; the dimension grows as ``∏ d_i``.
        """
        total = int(np.prod(self.physical_dims))
        if total > 1 << 12:
            raise MemoryError(f"refusing to materialize {total}×{total} dense MPO")
        op: NDArray = self._tensors[0]  # (1, d0, d0, D1)
        op = op.reshape(self.physical_dims[0], self.physical_dims[0], -1)
        for i in range(1, self.L):
            W = self._tensors[i]  # (Di, d, d, Di')
            op = np.tensordot(op, W, axes=([-1], [0]))  # (... , d, d, Di')
            # collapse pairs of physical legs into single rows / cols
            list(op.shape)
            # current op has shape (d0,d0, d1,d1, …, di,di, Di+1)
        # rebuild by explicit reshape: split out trailing virtual bond and group rows/cols
        # Easier: do it incrementally with explicit row/col tensors.
        return self._to_dense_explicit()

    def _to_dense_explicit(self) -> NDArray:
        """Reference implementation for ``to_dense`` with explicit reshaping."""
        d_tot = int(np.prod(self.physical_dims))
        if d_tot > 1 << 12:
            raise MemoryError(f"refusing to materialize {d_tot}×{d_tot} dense MPO")
        # Carry op of shape (rows, cols, D_right)
        W0 = self._tensors[0]  # (1, d, d, D1)
        op = W0.reshape(W0.shape[1], W0.shape[2], W0.shape[3])
        for i in range(1, self.L):
            W = self._tensors[i]  # (Di, d, d, Di+1)
            r, c, _D = op.shape
            d = W.shape[1]
            # contract bond
            op = np.tensordot(op, W, axes=([2], [0]))  # (r, c, d, d, Di+1)
            op = op.transpose(0, 2, 1, 3, 4).reshape(r * d, c * d, W.shape[3])
        return op.reshape(d_tot, d_tot)

    def __repr__(self) -> str:
        return (
            f"MPO(L={self.L}, max_bond={self.max_bond}, "
            f"physical={self.physical_dims}, dtype={self.dtype})"
        )
