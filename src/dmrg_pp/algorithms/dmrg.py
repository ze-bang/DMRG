"""Two-site Density Matrix Renormalization Group sweep engine.

This module implements the workhorse algorithm: a finite-system, two-site DMRG
sweep that variationally minimizes ``⟨ψ|Ĥ|ψ⟩`` over the Matrix Product State
manifold of bounded bond dimension.

Algorithmic notes
-----------------
* Each "half-sweep" walks across every bond ``(i, i+1)``, contracts the local
  MPS pair into a 4-leg tensor, and minimizes the local energy via a Lanczos
  iteration on the effective Hamiltonian
  ``H_eff = L_i ⊗ W_i ⊗ W_{i+1} ⊗ R_{i+1}``.  The optimized two-site tensor
  is then split via SVD with bond truncation and the orthogonality centre is
  shifted by one site.
* We support a ramped bond-dimension schedule so that early sweeps run cheaply
  while the state has not yet refined; the cap is increased monotonically with
  the sweep index.
* Convergence diagnostics include per-sweep min/max/mean energy across bonds,
  per-bond truncation error, and the Δ-energy between the last two sweeps.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.algorithms.eigensolver import lanczos_ground_state
from dmrg_pp.algorithms.environments import EnvironmentCache
from dmrg_pp.operators.mpo import MPO
from dmrg_pp.states.mps import MPS
from dmrg_pp.states.random import random_mps
from dmrg_pp.tensors.linalg import truncated_svd
from dmrg_pp.utils.threading import single_threaded_blas

log = logging.getLogger(__name__)

__all__ = ["DMRGConfig", "DMRGResult", "SweepStats", "run_dmrg"]


# --------------------------------------------------------------------- config
@dataclass
class DMRGConfig:
    """User-facing convergence and runtime knobs for :func:`run_dmrg`."""

    n_sweeps: int = 10
    """Maximum number of full sweeps (one full sweep = right + left)."""

    max_bond: int | Sequence[int] = 64
    """Maximum MPS bond dimension.  Accepts a list to ramp it across sweeps."""

    cutoff: float | Sequence[float] = 1e-10
    """SVD truncation cutoff (relative discarded weight)."""

    e_tol: float = 1e-9
    """Stop when ``|ΔE_sweep| < e_tol``."""

    var_tol: float = 0.0
    """Stop when the per-sweep energy variance falls below this (0 disables)."""

    lanczos_tol: float = 1e-12
    """Tolerance forwarded to the inner eigen-solver."""

    lanczos_max_iter: int = 200
    """Maximum Krylov iterations per local solve."""

    seed: int | None = None
    """Seed for the random initial MPS (when none is provided)."""

    initial_bond: int = 4
    """Bond dimension used for the random initial MPS."""

    log_every: int = 1
    """Sweep stride at which to emit progress logs."""

    blas_threads: int | None = 1
    """Cap on BLAS thread-pool size during the sweep.  ``1`` (the default) is
    almost always optimal for small per-bond tensor blocks: it eliminates
    OpenBLAS / MKL parking overhead which otherwise bottlenecks DMRG by 10×–
    100× on hyper-threaded boxes (especially in WSL/containers).
    Set to ``None`` to leave the global thread pool untouched."""

    def bond_for_sweep(self, sweep: int) -> int:
        if isinstance(self.max_bond, int):
            return int(self.max_bond)
        idx = min(sweep, len(self.max_bond) - 1)
        return int(self.max_bond[idx])

    def cutoff_for_sweep(self, sweep: int) -> float:
        if isinstance(self.cutoff, (int, float)):
            return float(self.cutoff)
        idx = min(sweep, len(self.cutoff) - 1)
        return float(self.cutoff[idx])


# --------------------------------------------------------------------- result
@dataclass
class SweepStats:
    """Diagnostics for a single full sweep (right + left)."""

    sweep: int
    energy: float
    energy_variance: float
    max_truncation_error: float
    max_bond_used: int
    wall_time: float
    delta_energy: float


@dataclass
class DMRGResult:
    """Final DMRG output, including the optimized MPS."""

    mps: MPS
    energy: float
    sweeps: list[SweepStats] = field(default_factory=list)
    converged: bool = False
    config: DMRGConfig | None = None

    @property
    def energy_history(self) -> list[float]:
        return [s.energy for s in self.sweeps]


# ---------------------------------------------------------- effective H ops
def _apply_two_site_heff(
    psi2_flat: NDArray,
    L_env: NDArray,
    W1: NDArray,
    W2: NDArray,
    R_env: NDArray,
    shape: tuple[int, int, int, int],
) -> NDArray:
    """Apply the two-site effective Hamiltonian to a flattened state vector.

    Convention
    ----------
    ``shape = (Dl, d1, d2, Dr)`` is the natural rank-4 layout of the
    two-site tensor.  Index legs are contracted in the cheapest order
    (``L_env, W1, W2, R_env``) for total cost
    ``O(Dl² Do d1 d2 + Dl Do² d1² d2 Dr)`` per matvec.
    """
    psi2 = psi2_flat.reshape(shape)
    # L_env (a_bra, o, a_ket) · psi2 (a_ket, d1, d2, b_ket) over a_ket
    tmp = np.tensordot(L_env, psi2, axes=([2], [0]))  # (a_bra, o, d1, d2, b_ket)
    # W1 (o, d1', d1, o') — contract (o, d1)
    tmp = np.tensordot(tmp, W1, axes=([1, 2], [0, 2]))  # (a_bra, d2, b_ket, d1', o')
    # W2 (o', d2', d2, o'') — contract (d2, o')
    tmp = np.tensordot(tmp, W2, axes=([1, 4], [2, 0]))  # (a_bra, b_ket, d1', d2', o'')
    # R_env (b_bra, o'', b_ket) — contract (b_ket, o'')
    tmp = np.tensordot(tmp, R_env, axes=([1, 4], [2, 1]))  # (a_bra, d1', d2', b_bra)
    return tmp.reshape(-1)


def _two_site_solve(
    L_env: NDArray,
    R_env: NDArray,
    W1: NDArray,
    W2: NDArray,
    psi2_init: NDArray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[float, NDArray]:
    """Local ground-state problem on the two-site block."""
    shape = psi2_init.shape  # (Dl, d1, d2, Dr)
    dim = int(np.prod(shape))
    dtype = np.result_type(L_env.dtype, R_env.dtype, W1.dtype, W2.dtype, psi2_init.dtype)

    L_env = L_env.astype(dtype, copy=False)
    R_env = R_env.astype(dtype, copy=False)
    W1 = W1.astype(dtype, copy=False)
    W2 = W2.astype(dtype, copy=False)
    init_flat = psi2_init.astype(dtype, copy=False).reshape(-1)

    def matvec(v: NDArray) -> NDArray:
        return _apply_two_site_heff(v.astype(dtype, copy=False), L_env, W1, W2, R_env, shape)

    energy, vec = lanczos_ground_state(
        matvec, dim, initial=init_flat, tol=tol, max_iter=max_iter, dtype=dtype
    )
    return energy, vec.reshape(shape)


# ------------------------------------------------------------------ driver
def run_dmrg(
    mpo: MPO,
    *,
    psi0: MPS | None = None,
    config: DMRGConfig | None = None,
) -> DMRGResult:
    """Variationally find the ground state of ``mpo`` via two-site DMRG.

    Parameters
    ----------
    mpo:
        Hamiltonian as a Matrix Product Operator.
    psi0:
        Initial MPS guess.  If ``None``, a random MPS of bond dimension
        ``config.initial_bond`` is used.
    config:
        Convergence/runtime configuration.  Defaults to
        :class:`DMRGConfig()`.

    Returns
    -------
    :class:`DMRGResult`
        ``.mps`` is the (left-canonical) optimized MPS, ``.energy`` its
        ground-state energy estimate, ``.sweeps`` a list of per-sweep stats.
    """
    if config is None:
        config = DMRGConfig()
    L = mpo.L
    if L < 2:
        raise ValueError(f"DMRG requires L >= 2 sites, got L={L}")

    if psi0 is None:
        psi = random_mps(
            mpo.physical_dims,
            bond_dim=config.initial_bond,
            seed=config.seed,
            dtype=np.result_type(mpo.dtype, np.float64),
        )
    else:
        if psi0.physical_dims != mpo.physical_dims:
            raise ValueError("psi0 and MPO physical dimensions disagree")
        psi = psi0.copy()
    psi.orthogonalize(0)

    cache = EnvironmentCache(psi, mpo)
    cache.build_all_right()

    result = DMRGResult(mps=psi, energy=float("nan"), config=config)

    log.info(
        "DMRG started: L=%d, MPO bond max=%d, target sweeps=%d",
        L,
        mpo.max_bond,
        config.n_sweeps,
    )

    blas_ctx = single_threaded_blas() if config.blas_threads == 1 else _NullContext()
    with blas_ctx:
        _run_sweeps(psi, mpo, cache, config, result)
    return result


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _run_sweeps(
    psi: MPS,
    mpo: MPO,
    cache: EnvironmentCache,
    config: DMRGConfig,
    result: DMRGResult,
) -> None:
    L = mpo.L
    prev_energy = float("inf")
    for sweep in range(config.n_sweeps):
        t0 = time.perf_counter()
        energies: list[float] = []
        truncs: list[float] = []
        max_bond = config.bond_for_sweep(sweep)
        cutoff = config.cutoff_for_sweep(sweep)

        # ------------------------------------ right-going half-sweep
        for i in range(L - 1):
            E, info = _optimize_bond(
                psi,
                mpo,
                cache,
                i,
                max_bond,
                cutoff,
                lanczos_tol=config.lanczos_tol,
                lanczos_max_iter=config.lanczos_max_iter,
                direction="right",
            )
            energies.append(E)
            truncs.append(info)

        # ------------------------------------ left-going half-sweep
        for i in range(L - 2, -1, -1):
            E, info = _optimize_bond(
                psi,
                mpo,
                cache,
                i,
                max_bond,
                cutoff,
                lanczos_tol=config.lanczos_tol,
                lanczos_max_iter=config.lanczos_max_iter,
                direction="left",
            )
            energies.append(E)
            truncs.append(info)

        wall = time.perf_counter() - t0
        e_mean = float(np.mean(energies))
        e_var = float(np.var(energies))
        max_trunc = float(max(truncs)) if truncs else 0.0
        delta_e = e_mean - prev_energy

        stats = SweepStats(
            sweep=sweep,
            energy=e_mean,
            energy_variance=e_var,
            max_truncation_error=max_trunc,
            max_bond_used=psi.max_bond,
            wall_time=wall,
            delta_energy=delta_e,
        )
        result.sweeps.append(stats)
        result.energy = e_mean

        if sweep % max(config.log_every, 1) == 0:
            log.info(
                "  sweep %3d  E=%.12f  ΔE=%+.3e  var=%.3e  trunc=%.3e  D=%d  t=%.2fs",
                sweep,
                e_mean,
                delta_e,
                e_var,
                max_trunc,
                psi.max_bond,
                wall,
            )

        if sweep > 0 and abs(delta_e) < config.e_tol:
            result.converged = True
            log.info("DMRG converged: |ΔE| = %.3e < %.3e", abs(delta_e), config.e_tol)
            break
        if config.var_tol > 0 and e_var < config.var_tol:
            result.converged = True
            log.info("DMRG converged on variance: var = %.3e < %.3e", e_var, config.var_tol)
            break
        prev_energy = e_mean


def _optimize_bond(
    psi: MPS,
    mpo: MPO,
    cache: EnvironmentCache,
    i: int,
    max_bond: int,
    cutoff: float,
    *,
    lanczos_tol: float,
    lanczos_max_iter: int,
    direction: str,
) -> tuple[float, float]:
    """Run a single two-site update on bond ``(i, i+1)``.

    After the call, the orthogonality centre sits on site ``i+1`` (when sweeping
    right) or site ``i`` (when sweeping left), and the corresponding
    environment slot is refreshed.

    Returns
    -------
    energy, truncation_error
        Local ground-state energy from the inner Lanczos solve and the SVD
        truncation error of this bond update.
    """
    A, B = psi[i], psi[i + 1]
    Dl, d1, _ = A.shape
    _, d2, Dr = B.shape
    psi2_init = np.tensordot(A, B, axes=([2], [0]))  # (Dl, d1, d2, Dr)

    L_env = cache.left(i)
    # R_env is the contraction of sites i+2..L-1 (i.e. the env at the *right*
    # bond of site i+1).  In the cache convention R_envs[k] = sites k..L-1.
    R_env = cache.right(i + 2)
    W1, W2 = mpo[i], mpo[i + 1]

    energy, psi2 = _two_site_solve(
        L_env,
        R_env,
        W1,
        W2,
        psi2_init,
        tol=lanczos_tol,
        max_iter=lanczos_max_iter,
    )

    # SVD-truncate (Dl·d1, d2·Dr)
    psi2_mat = psi2.reshape(Dl * d1, d2 * Dr)
    U, s, Vh, info = truncated_svd(psi2_mat, max_bond=max_bond, cutoff=cutoff, renormalize=True)

    new_D = info.bond_dim
    if direction == "right":
        # Place singular values on the right (B becomes the new orthogonality centre).
        new_A = U.reshape(Dl, d1, new_D)
        new_B = (np.diag(s) @ Vh).reshape(new_D, d2, Dr)
        psi._tensors[i] = new_A
        psi._tensors[i + 1] = new_B
        psi._centre = i + 1
        cache.update_left(i)
    else:
        # Sweeping left: singular values on the left.
        new_A = (U @ np.diag(s)).reshape(Dl, d1, new_D)
        new_B = Vh.reshape(new_D, d2, Dr)
        psi._tensors[i] = new_A
        psi._tensors[i + 1] = new_B
        psi._centre = i
        cache.update_right(i + 1)

    return energy, info.truncation_error
