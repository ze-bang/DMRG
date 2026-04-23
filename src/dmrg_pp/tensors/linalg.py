"""Numerically robust tensor linear algebra primitives.

All routines here operate on dense ``numpy.ndarray`` objects.  They are written
to (a) gracefully fall back to a divide-and-conquer SVD when LAPACK's faster
``gesdd`` driver fails to converge, (b) deterministically fix gauge freedoms so
that downstream tests are reproducible, and (c) expose rich truncation
diagnostics for the DMRG sweep engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, qr, svd

__all__ = [
    "TruncationInfo",
    "left_qr",
    "right_qr",
    "truncated_svd",
]


@dataclass(frozen=True, slots=True)
class TruncationInfo:
    """Diagnostics returned by :func:`truncated_svd`.

    Attributes
    ----------
    bond_dim:
        Number of singular values retained.
    truncation_error:
        Sum of squared discarded singular values, ``Σ_{i>χ} s_i²``.
    norm_loss:
        ``1 - ||s_kept||²``, i.e. probability mass thrown away (after rescaling
        the input to unit Frobenius norm — useful as a fidelity proxy).
    largest_discarded:
        Largest discarded singular value, or ``0.0`` if nothing was dropped.
    spectrum:
        The retained singular values, sorted in descending order.
    """

    bond_dim: int
    truncation_error: float
    norm_loss: float
    largest_discarded: float
    spectrum: NDArray[np.float64]


def _safe_svd(matrix: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """SVD with a robust fallback path when LAPACK ``gesdd`` fails to converge."""
    try:
        return svd(matrix, full_matrices=False, lapack_driver="gesdd")
    except LinAlgError:
        return svd(matrix, full_matrices=False, lapack_driver="gesvd")


def truncated_svd(
    matrix: NDArray,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    min_bond: int = 1,
    renormalize: bool = False,
) -> tuple[NDArray, NDArray, NDArray, TruncationInfo]:
    """Truncated SVD with rich diagnostics.

    Decomposes ``matrix ≈ U · diag(s) · Vh`` keeping at most ``max_bond``
    singular values and discarding any with ``s_i² < cutoff * Σ_j s_j²``
    (relative cutoff on the *discarded* tail, matching the convention used by
    most modern DMRG codes).

    Parameters
    ----------
    matrix:
        2-D array to decompose.
    max_bond:
        Hard upper bound on the bond dimension.  ``None`` means no cap.
    cutoff:
        Relative truncation threshold.  Singular values are dropped from the
        smallest end as long as the cumulative discarded weight stays below
        this threshold.
    min_bond:
        Lower bound on the bond dimension (always at least 1 to avoid empty
        bonds).
    renormalize:
        If ``True``, rescale the kept singular values so that the truncated
        state is normalized.  Useful when SVD-truncating an MPS that should
        remain a unit vector.

    Returns
    -------
    U, s, Vh, info
        ``U`` has shape ``(m, χ)``, ``s`` shape ``(χ,)``, ``Vh`` shape
        ``(χ, n)``, and ``info`` carries truncation diagnostics.
    """
    if matrix.ndim != 2:
        raise ValueError(f"truncated_svd expects a 2-D array, got ndim={matrix.ndim}")

    U, s, Vh = _safe_svd(matrix)
    s = s.astype(np.float64, copy=False)

    total_weight = float(np.sum(s * s))
    if total_weight == 0.0:
        chi = max(min_bond, 1)
        chi = min(chi, s.size if s.size > 0 else 1)
        return (
            U[:, :chi],
            s[:chi],
            Vh[:chi, :],
            TruncationInfo(
                bond_dim=chi,
                truncation_error=0.0,
                norm_loss=0.0,
                largest_discarded=0.0,
                spectrum=s[:chi].copy(),
            ),
        )

    # cum_discard[k] = sum of squared singular values dropped if we keep k of them.
    # That is, cum_discard[k] = Σ_{i ≥ k} s_i², so cum_discard[0] = total and
    # cum_discard[n] = 0.  It is monotonically *non-increasing* in k.
    s2 = s * s
    cum_discard = np.concatenate(([0.0], np.cumsum(s2[::-1])))[::-1]

    threshold = cutoff * total_weight
    # Smallest χ with cum_discard[χ] ≤ threshold.  Search on the ascending
    # reversed array and remap back: if `j` of the n+1 entries are ≤ threshold,
    # the largest valid reversed-index is `j-1`, mapping to χ = (n+1)-j.
    j = int(np.searchsorted(cum_discard[::-1], threshold, side="right"))
    chi_cut = s.size + 1 - j
    chi_cut = max(chi_cut, 1)

    chi = chi_cut
    if max_bond is not None:
        chi = min(chi, max_bond)
    chi = max(chi, min(min_bond, s.size))
    chi = min(chi, s.size)

    kept = s[:chi]
    discarded = s[chi:]
    truncation_error = float(np.sum(discarded * discarded))
    norm_loss = truncation_error / total_weight
    largest_discarded = float(discarded[0]) if discarded.size else 0.0

    if renormalize:
        kept_norm = float(np.linalg.norm(kept))
        if kept_norm > 0.0:
            kept = kept / kept_norm

    info = TruncationInfo(
        bond_dim=chi,
        truncation_error=truncation_error,
        norm_loss=norm_loss,
        largest_discarded=largest_discarded,
        spectrum=kept.copy(),
    )

    return U[:, :chi], kept, Vh[:chi, :], info


def left_qr(matrix: NDArray) -> tuple[NDArray, NDArray]:
    """Thin QR with a deterministic, sign-fixed gauge.

    Returns ``(Q, R)`` such that ``Q @ R == matrix`` with ``Q`` left-isometric
    (``Q† Q == 1``) and the diagonal of ``R`` made non-negative — this removes
    the gauge ambiguity of QR and makes round-trips bit-for-bit reproducible.
    """
    if matrix.ndim != 2:
        raise ValueError(f"left_qr expects a 2-D array, got ndim={matrix.ndim}")
    Q, R = qr(matrix, mode="economic")
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign
    R = R * diag_sign[:, None]
    return Q, R


def right_qr(matrix: NDArray) -> tuple[NDArray, NDArray]:
    """Right-isometric (LQ) decomposition: ``matrix = L @ Q`` with ``Q Q† = 1``."""
    if matrix.ndim != 2:
        raise ValueError(f"right_qr expects a 2-D array, got ndim={matrix.ndim}")
    Qt, Lt = left_qr(matrix.conj().T)
    Q = Qt.conj().T
    L = Lt.conj().T
    return L, Q
