"""Sparse-eigensolver wrappers used by the DMRG sweep engine.

We expose a single :func:`lanczos_ground_state` helper that wraps SciPy's
ARPACK / LOBPCG drivers behind a uniform interface and gracefully falls back
to a dense ``eigh`` for tiny effective Hilbert spaces (where ARPACK is both
slower and less accurate).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh

__all__ = ["lanczos_ground_state"]


def lanczos_ground_state(
    matvec: Callable[[NDArray], NDArray],
    dim: int,
    *,
    initial: NDArray | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
    dtype: np.dtype = np.float64,
    dense_threshold: int = 32,
) -> tuple[float, NDArray]:
    """Compute the smallest eigen-pair of a Hermitian operator implicitly defined by ``matvec``.

    Parameters
    ----------
    matvec:
        Callable applying the operator to a 1-D vector of length ``dim``.
    dim:
        Dimension of the operator's domain.
    initial:
        Starting vector (will be normalized).  ``None`` uses a random vector.
    tol:
        Convergence tolerance forwarded to ARPACK.
    max_iter:
        Maximum ARPACK iterations.
    dtype:
        Working precision; controls both the LinearOperator dtype and the
        random initialization fallback.
    dense_threshold:
        For ``dim ≤ dense_threshold`` the operator is materialized and
        diagonalized with ``np.linalg.eigh`` (faster and more accurate than
        ARPACK for tiny problems).

    Returns
    -------
    eigenvalue, eigenvector
        ``eigenvalue`` is a real Python float; ``eigenvector`` is a numpy
        array of length ``dim`` with unit norm.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if dim == 1:
        v = np.ones(1, dtype=dtype)
        out = matvec(v)
        return float(np.real(out[0])), v

    if dim <= dense_threshold:
        H = np.empty((dim, dim), dtype=dtype)
        basis = np.eye(dim, dtype=dtype)
        for k in range(dim):
            H[:, k] = matvec(basis[:, k])
        H = 0.5 * (H + H.conj().T)  # symmetrize to kill round-off non-Hermiticity
        if np.iscomplexobj(H):
            evals, evecs = np.linalg.eigh(H)
        else:
            evals, evecs = np.linalg.eigh(H)
        return float(evals[0]), evecs[:, 0]

    op = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=dtype)
    if initial is None:
        rng = np.random.default_rng(0)
        v0 = rng.standard_normal(dim).astype(dtype, copy=False)
    else:
        v0 = initial.astype(dtype, copy=False).reshape(-1)
        if v0.size != dim:
            raise ValueError(f"initial vector size {v0.size} != dim {dim}")
    nrm = float(np.linalg.norm(v0))
    if nrm == 0.0:
        rng = np.random.default_rng(0)
        v0 = rng.standard_normal(dim).astype(dtype, copy=False)
        nrm = float(np.linalg.norm(v0))
    v0 = v0 / nrm

    ncv = min(max(2 * 1 + 1, 20), dim - 1)
    try:
        evals, evecs = eigsh(op, k=1, which="SA", v0=v0, tol=tol, maxiter=max_iter, ncv=ncv)
    except ArpackError:
        # Last-ditch fallback: shift-invert via dense eigh on a Krylov subspace.
        return _krylov_fallback(matvec, dim, dtype=dtype, max_iter=max_iter)
    return float(evals[0]), evecs[:, 0]


def _krylov_fallback(
    matvec: Callable[[NDArray], NDArray],
    dim: int,
    *,
    dtype: np.dtype,
    max_iter: int,
) -> tuple[float, NDArray]:
    """Tiny self-contained Lanczos routine used when ARPACK declines to converge."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(dim).astype(dtype, copy=False)
    v /= np.linalg.norm(v)

    Q: list[NDArray] = []
    alphas: list[float] = []
    betas: list[float] = []
    beta = 0.0
    v_prev = np.zeros_like(v)

    k = min(max(50, max_iter // 4), dim)
    for _ in range(k):
        w = matvec(v) - beta * v_prev
        alpha = float(np.real(np.vdot(v, w)))
        w = w - alpha * v
        # one round of full re-orthogonalization for numerical stability
        for q in Q:
            w = w - np.vdot(q, w) * q
        beta = float(np.linalg.norm(w))
        Q.append(v)
        alphas.append(alpha)
        if beta < 1e-14:
            break
        betas.append(beta)
        v_prev = v
        v = w / beta

    n = len(alphas)
    T = np.diag(alphas)
    for i, b in enumerate(betas):
        T[i, i + 1] = b
        T[i + 1, i] = b
    evals, evecs = np.linalg.eigh(T)
    coeffs = evecs[:, 0]
    out = np.zeros(dim, dtype=dtype)
    for c, q in zip(coeffs[:n], Q[:n], strict=True):
        out = out + c * q
    out /= np.linalg.norm(out)
    return float(evals[0]), out
