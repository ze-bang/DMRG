"""Matrix Product State (MPS) container with canonical-form bookkeeping.

Conventions
-----------
A finite MPS on ``L`` sites with physical dimensions ``d_0, …, d_{L-1}`` is
represented as a list of rank-3 tensors

    M[i].shape == (D_{i}, d_i, D_{i+1})

with virtual dimensions ``D_0 = D_L = 1`` and arbitrary internal bond
dimensions ``D_i``.  The first index is the *left* virtual index, the middle
index is the *physical* index, and the last index is the *right* virtual index.

Canonical form
~~~~~~~~~~~~~~
A site is "left-canonical" if reshaping ``M[i]`` to a matrix of shape
``(D_i d_i, D_{i+1})`` gives a left-isometry; right-canonical is the mirror
condition.  The class tracks the orthogonality centre (a single site index, or
``None`` if no canonical structure is currently maintained) and provides
:meth:`MPS.orthogonalize` to move it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ``typing.Self`` was added in Python 3.11; we only need it for type
    # checking, so keep the import behind ``TYPE_CHECKING`` to stay
    # compatible with Python 3.10.
    from typing import Self

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.tensors.linalg import left_qr, right_qr, truncated_svd

__all__ = ["MPS", "Canonical"]


class Canonical(str, Enum):
    """Canonical-form labels for individual MPS sites."""

    LEFT = "L"
    RIGHT = "R"
    NONE = "N"


class MPS:
    """A finite-system Matrix Product State.

    The class is intentionally light-weight: tensors are owned as a list of
    ``numpy.ndarray`` so power users can manipulate them directly, while the
    class methods provide the canonicalization, normalization, and bond-dim
    bookkeeping needed by the DMRG sweep engine.
    """

    __slots__ = ("_centre", "_dtype", "_tensors")

    def __init__(
        self,
        tensors: Sequence[NDArray],
        *,
        centre: int | None = None,
        copy: bool = True,
    ) -> None:
        if len(tensors) == 0:
            raise ValueError("MPS must contain at least one site")
        ts: list[NDArray] = []
        prev_right = 1
        for i, t in enumerate(tensors):
            if t.ndim != 3:
                raise ValueError(f"site {i}: tensor must be rank-3, got ndim={t.ndim}")
            if t.shape[0] != prev_right:
                raise ValueError(
                    f"site {i}: left bond {t.shape[0]} != previous right bond {prev_right}"
                )
            ts.append(np.array(t, copy=copy))
            prev_right = t.shape[2]
        if prev_right != 1:
            raise ValueError(f"final right bond must be 1, got {prev_right}")

        self._tensors: list[NDArray] = ts
        self._dtype = np.result_type(*[t.dtype for t in ts])
        if centre is not None and not (0 <= centre < len(ts)):
            raise ValueError(f"centre={centre} out of range [0, {len(ts)})")
        self._centre: int | None = centre

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self._tensors)

    def __iter__(self) -> Iterable[NDArray]:
        return iter(self._tensors)

    def __getitem__(self, i: int) -> NDArray:
        return self._tensors[i]

    def __setitem__(self, i: int, value: NDArray) -> None:
        if value.ndim != 3:
            raise ValueError(f"MPS site tensor must be rank-3, got ndim={value.ndim}")
        old = self._tensors[i]
        if value.shape[1] != old.shape[1]:
            raise ValueError(
                f"physical dim mismatch at site {i}: {value.shape[1]} vs {old.shape[1]}"
            )
        self._tensors[i] = value
        self._centre = None  # gauge information potentially invalidated

    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self._tensors)

    @property
    def physical_dims(self) -> tuple[int, ...]:
        return tuple(int(t.shape[1]) for t in self._tensors)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Virtual bond dimensions ``(D_0, D_1, …, D_L)``, length ``L+1``."""
        dims = [int(self._tensors[0].shape[0])]
        for t in self._tensors:
            dims.append(int(t.shape[2]))
        return tuple(dims)

    @property
    def max_bond(self) -> int:
        return max(self.bond_dims)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def centre(self) -> int | None:
        """Index of the orthogonality centre (or ``None`` if not canonical)."""
        return self._centre

    def copy(self) -> MPS:
        return MPS([t.copy() for t in self._tensors], centre=self._centre, copy=False)

    # ------------------------------------------------------- canonical sweeps
    def _left_canonicalize_site(self, i: int) -> None:
        """Left-canonicalize site ``i`` and absorb the rest into site ``i+1``."""
        t = self._tensors[i]
        Dl, d, Dr = t.shape
        Q, R = left_qr(t.reshape(Dl * d, Dr))
        new_Dr = Q.shape[1]
        self._tensors[i] = Q.reshape(Dl, d, new_Dr)
        if i + 1 < len(self._tensors):
            nxt = self._tensors[i + 1]
            self._tensors[i + 1] = np.tensordot(R, nxt, axes=([1], [0]))
        else:
            # absorb remaining scalar into the last site (preserving norm)
            self._tensors[i] = np.tensordot(self._tensors[i], R, axes=([2], [0]))

    def _right_canonicalize_site(self, i: int) -> None:
        """Right-canonicalize site ``i`` and absorb into site ``i-1``."""
        t = self._tensors[i]
        Dl, d, Dr = t.shape
        L, Q = right_qr(t.reshape(Dl, d * Dr))
        new_Dl = Q.shape[0]
        self._tensors[i] = Q.reshape(new_Dl, d, Dr)
        if i - 1 >= 0:
            prv = self._tensors[i - 1]
            self._tensors[i - 1] = np.tensordot(prv, L, axes=([2], [0]))
        else:
            self._tensors[i] = np.tensordot(L, self._tensors[i], axes=([1], [0]))

    def orthogonalize(self, centre: int) -> Self:
        """Move the orthogonality centre to site ``centre`` (in-place).

        Sites ``< centre`` become left-canonical, sites ``> centre`` become
        right-canonical.  After the call the wave function is unchanged but
        its gauge is fixed; in particular ``self.norm()`` is just
        ``np.linalg.norm(self[centre])``.
        """
        if not (0 <= centre < self.L):
            raise ValueError(f"centre={centre} out of range [0, {self.L})")

        for i in range(centre):
            self._left_canonicalize_site(i)
        for i in range(self.L - 1, centre, -1):
            self._right_canonicalize_site(i)
        self._centre = centre
        return self

    def normalize(self) -> Self:
        """Rescale to unit norm.  Requires (and re-establishes) canonical form."""
        if self._centre is None:
            self.orthogonalize(0)
        c = self._centre
        assert c is not None
        n = float(np.linalg.norm(self._tensors[c]))
        if n == 0.0:
            raise ValueError("Cannot normalize zero MPS")
        self._tensors[c] = self._tensors[c] / n
        return self

    def norm(self) -> float:
        """Compute ``||ψ||`` via canonicalization (cheap if already canonical)."""
        if self._centre is None:
            self.orthogonalize(0)
        c = self._centre
        assert c is not None
        return float(np.linalg.norm(self._tensors[c]))

    # ----------------------------------------------------- inner product etc.
    def overlap(self, other: MPS) -> complex:
        """Compute ``⟨self | other⟩``.  Both states must share the physical legs."""
        if other.L != self.L:
            raise ValueError(f"length mismatch: {self.L} vs {other.L}")
        if other.physical_dims != self.physical_dims:
            raise ValueError("physical dimensions do not match")
        # Contract from the left, carrying a (D_self, D_other) environment.
        env: NDArray = np.ones((1, 1), dtype=np.result_type(self.dtype, other.dtype))
        for a, b in zip(self._tensors, other._tensors, strict=True):
            # env: (Da, Db);  a: (Da, d, Da')  b: (Db, d, Db')
            env = np.tensordot(env, b, axes=([1], [0]))  # (Da, d, Db')
            env = np.tensordot(a.conj(), env, axes=([0, 1], [0, 1]))  # (Da', Db')
        val = env.reshape(())
        return complex(val)

    def to_state_vector(self) -> NDArray:
        """Contract the MPS into a dense state vector of size ``∏ d_i``.

        Intended for tests on small systems; the result has size growing
        exponentially with ``L``.
        """
        total = int(np.prod(self.physical_dims))
        if total > 1 << 20:
            raise MemoryError(f"refusing to materialize 2^{int(np.log2(total))}-dim state vector")
        # Carry a tensor of shape (D_left, d_0, d_1, …, d_i)
        psi: NDArray = self._tensors[0]  # (1, d_0, D_1)
        for i in range(1, self.L):
            psi = np.tensordot(psi, self._tensors[i], axes=([-1], [0]))
        # psi shape: (1, d_0, d_1, …, d_{L-1}, 1)
        return psi.reshape(-1)

    # ----------------------------------------------------- truncation helper
    def truncate(
        self,
        *,
        max_bond: int | None = None,
        cutoff: float = 0.0,
        renormalize: bool = True,
    ) -> Self:
        """Sweep-truncate every bond (in-place) using SVD.

        Brings the state to a left/right canonical form first so that the
        truncated SVD gives a quasi-optimal approximation.  When
        ``renormalize=True`` (the default) the resulting MPS is rescaled to
        unit Frobenius norm; pass ``renormalize=False`` to preserve the
        truncation-induced loss of norm (useful when monitoring fidelities).
        """
        self.orthogonalize(self.L - 1)
        for i in range(self.L - 1, 0, -1):
            t = self._tensors[i]
            Dl, d, Dr = t.shape
            U, s, Vh, _info = truncated_svd(
                t.reshape(Dl, d * Dr),
                max_bond=max_bond,
                cutoff=cutoff,
            )
            self._tensors[i] = Vh.reshape(-1, d, Dr)
            self._tensors[i - 1] = np.tensordot(self._tensors[i - 1], U * s, axes=([2], [0]))
        self._centre = 0
        if renormalize:
            self.normalize()
        return self

    # ------------------------------------------------------ classmethods etc.
    @classmethod
    def from_product_state(cls, configs: Sequence[Sequence[complex] | int]) -> MPS:
        """Build a product MPS.

        ``configs[i]`` may either be an integer (basis index) or a coefficient
        vector of length ``d_i``.
        """
        tensors: list[NDArray] = []
        for cfg in configs:
            if isinstance(cfg, int):
                # interpret as basis index; need a default d=2
                vec = np.zeros(2, dtype=np.complex128)
                vec[cfg] = 1.0
            else:
                vec = np.asarray(cfg, dtype=np.complex128)
            tensors.append(vec.reshape(1, -1, 1))
        return cls(tensors, centre=0)

    def __repr__(self) -> str:
        bonds = self.bond_dims
        return (
            f"MPS(L={self.L}, max_bond={max(bonds)}, "
            f"physical={self.physical_dims}, dtype={self.dtype}, centre={self._centre})"
        )
