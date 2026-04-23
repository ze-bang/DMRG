"""Two-point and equal-time correlation functions ``⟨A_i B_j⟩``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.states.mps import MPS

__all__ = ["correlation_matrix", "two_point"]


def two_point(
    psi: MPS,
    A: NDArray,
    B: NDArray,
    i: int,
    j: int,
    *,
    string: NDArray | None = None,
) -> complex:
    """Compute ``⟨ψ| A_i B_j |ψ⟩``, optionally with a Jordan–Wigner string in between.

    Parameters
    ----------
    A, B:
        Single-site operators applied at sites ``i`` and ``j`` respectively.
    string:
        If supplied, a single-site operator inserted on every site strictly
        between ``i`` and ``j`` (used for fermionic correlators that require
        the JW parity string).
    """
    if i == j:
        from dmrg_pp.measurements.observables import expectation_value

        return expectation_value(psi, A @ B, i)
    if i > j:
        # symmetry: <A_i B_j> = <B_j A_i> for commuting ops; for fermions, callers
        # should canonicalize the order themselves before calling.
        i, j = j, i
        A, B = B, A

    if not (i >= 0 and j < psi.L):
        raise ValueError(f"sites ({i}, {j}) out of range [0, {psi.L})")

    psi.orthogonalize(i)
    # carry env of shape (Dl_bra, Dl_ket); start with applying A on site i
    M = psi[i]
    # contract A with M's physical leg
    AM = np.tensordot(A, M, axes=([1], [1])).transpose(1, 0, 2)  # (Dl, d_out, Dr)
    env = np.tensordot(M.conj(), AM, axes=([0, 1], [0, 1]))  # (Dr_bra, Dr_ket)

    for k in range(i + 1, j):
        Mk = psi[k]
        if string is None:
            tmp = np.tensordot(env, Mk, axes=([1], [0]))  # (Dr_bra, d, Dr_ket)
            env = np.tensordot(Mk.conj(), tmp, axes=([0, 1], [0, 1]))  # (Dr_bra', Dr_ket')
        else:
            SM = np.tensordot(string, Mk, axes=([1], [1])).transpose(1, 0, 2)  # (Dl, d, Dr)
            tmp = np.tensordot(env, SM, axes=([1], [0]))
            env = np.tensordot(Mk.conj(), tmp, axes=([0, 1], [0, 1]))

    Mj = psi[j]
    BM = np.tensordot(B, Mj, axes=([1], [1])).transpose(1, 0, 2)
    tmp = np.tensordot(env, BM, axes=([1], [0]))  # (Dr_bra, d, Dr_ket)
    env = np.tensordot(Mj.conj(), tmp, axes=([0, 1], [0, 1]))  # (Dr_bra', Dr_ket')
    return complex(env.reshape(()))


def correlation_matrix(
    psi: MPS,
    A: NDArray,
    B: NDArray,
    *,
    string: NDArray | None = None,
) -> NDArray:
    """Full ``L × L`` matrix of ``⟨A_i B_j⟩`` (upper triangle filled, then symmetrized).

    For Hermitian Hermitian-conjugate pairs ``B = A†`` the result is Hermitian;
    for general operators only the upper triangle is meaningful (we copy it to
    the lower triangle for convenience).
    """
    L = psi.L
    out_dtype = np.result_type(psi.dtype, A, B, np.complex128)
    C = np.zeros((L, L), dtype=out_dtype)
    for i in range(L):
        for j in range(i, L):
            C[i, j] = two_point(psi, A, B, i, j, string=string)
            if i != j:
                C[j, i] = C[i, j]
    return C
