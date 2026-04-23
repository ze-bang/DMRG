"""Random MPS constructors used to seed DMRG."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dmrg_pp.states.mps import MPS

__all__ = ["random_mps"]


def random_mps(
    physical_dims: Sequence[int],
    *,
    bond_dim: int = 4,
    dtype: np.dtype | type = np.float64,
    seed: int | None = None,
) -> MPS:
    """Construct a random, normalized MPS with bounded virtual bond dimension.

    Parameters
    ----------
    physical_dims:
        Per-site physical dimensions.
    bond_dim:
        Maximum virtual bond dimension; bonds near the boundaries are
        truncated to ``min(bond_dim, ∏ d_i, ∏ d_{i+1:})`` so that no bond is
        larger than the dimension of the Hilbert space it could possibly
        compress.
    dtype:
        Numeric dtype (``np.float64`` or ``np.complex128`` typically).
    seed:
        RNG seed.  ``None`` uses a fresh non-deterministic seed.
    """
    rng = np.random.default_rng(seed)
    L = len(physical_dims)
    if L == 0:
        raise ValueError("physical_dims must be non-empty")

    # Compute capped bond dimensions: bond `b` (b ∈ [0, L]) sits between sites
    # b-1 and b, and cannot exceed dim(H_{<b}) on its left or dim(H_{≥b}) on
    # its right.  Boundary bonds 0 and L are always 1.
    left_caps = [1]
    cap = 1
    for d in physical_dims:
        cap = min(cap * int(d), bond_dim)
        left_caps.append(cap)
    right_caps = [1]
    cap = 1
    for d in reversed(physical_dims):
        cap = min(cap * int(d), bond_dim)
        right_caps.append(cap)
    right_caps = list(reversed(right_caps))
    bonds = [min(left_caps[i], right_caps[i]) for i in range(L + 1)]

    tensors = []
    for i, d in enumerate(physical_dims):
        Dl, Dr = bonds[i], bonds[i + 1]
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            t = rng.standard_normal((Dl, d, Dr)) + 1j * rng.standard_normal((Dl, d, Dr))
        else:
            t = rng.standard_normal((Dl, d, Dr))
        tensors.append(t.astype(dtype))

    psi = MPS(tensors, centre=None, copy=False)
    psi.normalize()
    return psi
