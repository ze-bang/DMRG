"""Symbolic Matrix Product Operator builder.

Build MPOs from a list of *terms* of the form

    coeff · O^{j_1} ⊗ O^{j_2} ⊗ … ⊗ O^{j_n}

where each ``O`` is a local operator named by a string (looked up from a
per-site catalog) and the sites ``j_1 < j_2 < … < j_n`` are the sites the term
acts non-trivially on.  Identities are inserted automatically on intervening
sites, and term coefficients are absorbed into the leftmost local operator.

Internally the builder uses the canonical finite-state-machine construction:
each bond ``b`` carries

* a "start" state (no term has begun yet),
* one "transit" state per term in flight across bond ``b``,
* an "end" state (everything emitted so far),

producing exactly-right local ``W`` tensors.  For nearest-neighbour 2-body
Hamiltonians (Heisenberg, TFIM, …) this yields the optimal ``D = 5`` MPO bond
dimension out of the box.

Usage
-----
>>> from dmrg_pp.operators.builder import MPOBuilder, OpTerm
>>> from dmrg_pp.operators.local_ops import spin_half_ops
>>> ops = spin_half_ops()
>>> b = MPOBuilder(L=4, local_ops=ops)
>>> for i in range(3):
...     b.add(1.0, (i, "Sz"), (i + 1, "Sz"))
>>> mpo = b.build()
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.operators.mpo import MPO

__all__ = ["MPOBuilder", "OpTerm"]

SiteOp = tuple[int, str]


@dataclass(frozen=True)
class OpTerm:
    """A single Hamiltonian term: ``coeff · ⊗_i O^{site_i}``.

    The list ``site_ops`` must contain at most one operator per site, and the
    sites are kept in strictly increasing order.
    """

    coeff: complex
    site_ops: tuple[SiteOp, ...]

    def __post_init__(self) -> None:
        if len(self.site_ops) == 0:
            raise ValueError("OpTerm must act on at least one site")
        sites = [s for s, _ in self.site_ops]
        if any(sites[k] >= sites[k + 1] for k in range(len(sites) - 1)):
            raise ValueError(f"OpTerm sites must be strictly increasing, got {sites}")


@dataclass
class MPOBuilder:
    """Symbolic builder that turns a list of :class:`OpTerm` into an :class:`MPO`.

    Parameters
    ----------
    L:
        Number of sites.
    local_ops:
        Either a single ``dict[str, NDArray]`` shared across all sites, or a
        per-site sequence of such dicts when sites have different physical
        dimensions (e.g. mixed spin/fermion lattices).
    identity_name:
        Key in the local-op dictionaries that resolves to the identity matrix
        on each site.  Defaults to ``"Id"``.
    """

    L: int
    local_ops: Mapping[str, NDArray] | Sequence[Mapping[str, NDArray]]
    identity_name: str = "Id"
    _terms: list[OpTerm] = field(default_factory=list, init=False, repr=False)

    def _ops_at(self, site: int) -> Mapping[str, NDArray]:
        if isinstance(self.local_ops, Mapping):
            return self.local_ops
        return self.local_ops[site]

    def _id_at(self, site: int) -> NDArray:
        ops = self._ops_at(site)
        if self.identity_name not in ops:
            raise KeyError(
                f"site {site}: identity operator '{self.identity_name}' not found in local_ops"
            )
        return ops[self.identity_name]

    def _phys_dim(self, site: int) -> int:
        return int(self._id_at(site).shape[0])

    # ------------------------------------------------------------------ API
    def add(
        self,
        coeff: complex,
        *site_ops: SiteOp,
    ) -> MPOBuilder:
        """Append the term ``coeff · ⊗_i O^{site_ops[i]}`` to the Hamiltonian."""
        if abs(coeff) == 0.0:
            return self
        sorted_sops = tuple(sorted(site_ops, key=lambda x: x[0]))
        for site, name in sorted_sops:
            if not (0 <= site < self.L):
                raise ValueError(f"site {site} out of range [0, {self.L})")
            ops = self._ops_at(site)
            if name not in ops:
                raise KeyError(f"operator '{name}' not in local_ops at site {site}")
        self._terms.append(OpTerm(complex(coeff), sorted_sops))
        return self

    def add_term(self, term: OpTerm) -> MPOBuilder:
        return self.add(term.coeff, *term.site_ops)

    def extend(self, terms: Sequence[OpTerm]) -> MPOBuilder:
        for t in terms:
            self.add_term(t)
        return self

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    # -------------------------------------------------------- main routine
    def build(self, *, dtype: np.dtype | type | None = None) -> MPO:
        """Assemble the MPO tensors.

        The resulting bond dimension at bond ``b`` is
        ``2 + (#terms whose first site < b ≤ last site)``.
        For a nearest-neighbour 2-body Hamiltonian on a 1-D chain this is the
        usual ``D = 5`` (start, three Pauli transit states, end).
        """
        L = self.L
        terms = self._terms

        # Determine, for each term, the bonds it is "in flight" on.
        # A term with sites j_1 < ... < j_n is in flight at bonds j_1+1 ... j_n.
        # (bond b sits between site b-1 and site b).
        in_flight_at: list[list[int]] = [[] for _ in range(L + 1)]
        for k, t in enumerate(terms):
            j_first = t.site_ops[0][0]
            j_last = t.site_ops[-1][0]
            for b in range(j_first + 1, j_last + 1):
                in_flight_at[b].append(k)

        # Bond layout: index 0 = "start", indices 1..len = transit states,
        # index -1 (i.e. len+1) = "end".
        # bond_state_index[b] -> dict mapping term_idx -> column number (1..)
        bond_layout: list[dict[int, int]] = []
        bond_dims: list[int] = []
        for b in range(L + 1):
            layout = {term_idx: 1 + i for i, term_idx in enumerate(in_flight_at[b])}
            bond_layout.append(layout)
            bond_dims.append(2 + len(layout))

        # Boundary bonds (b=0 and b=L) have no transit states by construction,
        # so D=2 there. The MPO tensors themselves still must have their
        # outermost bond = 1; we enforce this by *projecting* W[0] onto its
        # "start" row and W[L-1] onto its "end" column at the end.

        # Resolve dtype
        if dtype is None:
            sample_dtypes = [np.array(self._id_at(i)).dtype for i in range(L)]
            sample_dtypes.append(np.array([t.coeff for t in terms] or [1.0]).dtype)
            dtype = np.result_type(*sample_dtypes, np.float64)
        dtype_resolved = np.dtype(dtype)

        # Allocate W tensors
        W_list: list[NDArray] = []
        for i in range(L):
            d = self._phys_dim(i)
            Dl = bond_dims[i]
            Dr = bond_dims[i + 1]
            W = np.zeros((Dl, d, d, Dr), dtype=dtype_resolved)
            Id = self._id_at(i).astype(dtype_resolved, copy=False)
            # start -> start: identity
            W[0, :, :, 0] = Id
            # end -> end: identity
            W[Dl - 1, :, :, Dr - 1] = Id
            # transit -> transit on identity, for terms in flight on both bonds
            # that have no operator at site i:
            for term_idx, col in bond_layout[i + 1].items():
                if term_idx not in bond_layout[i]:
                    continue
                t = terms[term_idx]
                # check if site i is in the support of t
                if any(site == i for site, _ in t.site_ops):
                    continue
                row = bond_layout[i][term_idx]
                W[row, :, :, col] = Id
            W_list.append(W)

        # Now place actual operator entries from each term.
        for term_idx, t in enumerate(terms):
            j_sites = [s for s, _ in t.site_ops]
            j_first, j_last = j_sites[0], j_sites[-1]
            for k_pos, (site, op_name) in enumerate(t.site_ops):
                op = np.asarray(self._ops_at(site)[op_name], dtype=dtype_resolved)
                if site == j_first:
                    row = 0  # start
                else:
                    row = bond_layout[site][term_idx]
                if site == j_last:
                    col = bond_dims[site + 1] - 1  # end
                else:
                    col = bond_layout[site + 1][term_idx]
                # absorb coefficient into the leftmost local operator
                local = op * t.coeff if k_pos == 0 else op
                W_list[site][row, :, :, col] = W_list[site][row, :, :, col] + local

        # Project boundary tensors so that the outermost MPO bonds have dim 1.
        # Convention: at left boundary the only meaningful row is "start" (row 0);
        # at right boundary the only meaningful column is "end" (col -1).
        W_list[0] = W_list[0][0:1, :, :, :]
        W_list[-1] = W_list[-1][:, :, :, -1:]

        return MPO(W_list, copy=False)


# Convenience alias when the caller wants a list-style API.
TermLike = Union[OpTerm, tuple[complex, Sequence[SiteOp]]]
