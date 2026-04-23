# DMRG++ — algorithmic notes

This document records the conventions, derivations, and references that the
DMRG++ code follows.  It is not an introduction to DMRG; for that we
recommend Schollwöck (2011) and the [TeNPy paper](https://arxiv.org/abs/1805.00055).

## 1. Matrix Product States

For an `L`-site chain with site-`i` physical Hilbert space `ℂ^{d_i}`, an MPS
factorizes the wave-function coefficients as

    ψ_{σ_0 σ_1 … σ_{L-1}} = M^{σ_0}_{0,a_1} M^{σ_1}_{a_1,a_2} … M^{σ_{L-1}}_{a_{L-1},0}.

Code convention: each site tensor `M[i]` has shape `(D_left, d, D_right)`, the
first and last virtual bonds being trivially 1.

A *left-canonical* site satisfies `Σ_{σ a} (M^σ_{a, b})* M^σ_{a, b'} = δ_{b b'}`
(it is a left-isometry when reshaped to `(D_left · d, D_right)`).
*Right-canonical* is the mirror condition.  In a *mixed-canonical* MPS, all
sites strictly left of an "orthogonality centre" are left-canonical, all
sites strictly right of it are right-canonical, and the centre site carries
all the wave-function norm.  Moving the centre by one site costs one
QR / LQ decomposition.

## 2. Matrix Product Operators

An MPO factorizes a Hilbert-space operator as

    Ĥ = Σ W^{σ_0 τ_0}_{1, w_1} W^{σ_1 τ_1}_{w_1, w_2} … W^{σ_{L-1} τ_{L-1}}_{w_{L-1}, 1} |σ⟩⟨τ|.

Code convention: each site tensor `W[i]` has shape `(D, d, d, D')` with leg
ordering `(left bond, σ, τ, right bond)`.  Acting on a ket `|ψ⟩` therefore
contracts `W[…, σ, τ, …]` with the input physical index `τ` and produces the
output index `σ`.

### 2.1 Finite-state-machine MPO builder

A sum of operator strings can be encoded as a finite-state automaton on the
virtual bond:

* state `0` (= "start")  : no term has begun yet
* states `1, …, k`        : one slot per term currently in flight at the bond
* state `−1` (= "end")    : every term has been emitted

The non-zero blocks of `W[i]` correspond to FSM transitions consistent with
the local operator catalog.  This construction always yields the optimal MPO
bond dimension for sparse local Hamiltonians; in particular nearest-neighbour
2-body Hamiltonians on a 1-D chain compress to `D = 5`.

### 2.2 Jordan–Wigner strings (fermions)

For a spinful fermion model we use the convention `c_iσ = K_i c̃_iσ` with
`K_i = ∏_{k<i} F_k` and `F_k = (−1)^{N_k}`.  Nearest-neighbour bilinears then
reduce to

    c†_iσ c_{i+1}σ = (c̃†_iσ F_i) ⊗ c̃_{i+1}σ
    c†_{i+1}σ c_iσ = (F_i c̃_iσ) ⊗ c̃†_{i+1}σ

— note that the F-string is absorbed into the *left* site so that the symbolic
MPO builder sees a single local operator per site.  See
`src/dmrg_pp/models/hubbard.py` for the concrete usage.

## 3. Two-site DMRG

The variational principle on the MPS manifold reduces the global
ground-state problem to a sequence of local effective ones.  At bond `(i, i+1)`
we contract the two adjacent MPS sites into a rank-4 block

    Θ^{σ_i σ_{i+1}}_{a, b} = M^{σ_i}_{a, c} M^{σ_{i+1}}_{c, b}

and solve the local eigenvalue problem `H_eff Θ = E Θ` where

    H_eff = L_i ⊗ W_i ⊗ W_{i+1} ⊗ R_{i+1}

`L_i` and `R_{i+1}` are the cumulative left/right environments — 3-leg
tensors of shape `(D_a, D_o, D_a)` indexed by `(bra, MPO, ket)`.
The optimized `Θ_*` is then split via SVD with bond cap `χ_max` and discarded
weight `Σ_{i>χ} s_i² ≤ ε · Σ_i s_i²`, and the orthogonality centre is
shifted by one site.  Sweeping right-then-left across every bond constitutes
one DMRG sweep.

### 3.1 Convergence criteria

We declare convergence when the per-sweep mean energy varies by less than
`e_tol` between two consecutive sweeps.  An optional `var_tol` switch stops
on the *variance of the local energies across bonds within a single sweep*,
which goes to zero exactly when the wave function is a true eigenstate.

### 3.2 Cost

Per local solve the dominant cost is

    O(D³ d² D_o + D² d² D_o²)

per Lanczos matvec, where `D` is the MPS bond, `D_o` is the MPO bond, and
`d` is the physical dimension.  Memory is dominated by the environment
storage at `O(L · D² D_o)`.

## 4. Measurements

* **Local observables** `⟨O_i⟩` are O(D³ d²) once the orthogonality centre is
  parked at site `i`.  A full sweep of the centre across the chain therefore
  produces the full vector `⟨O_i⟩_{i=0..L-1}` in `O(L · D³ d²)` time.
* **Two-point correlators** `⟨A_i B_j⟩` carry a 2-leg
  `(D_bra, D_ket)` "transfer" environment from `i` to `j`.  Optionally a
  Jordan–Wigner string is inserted on every intermediate site.
* **Bipartite entropies** are extracted from the Schmidt spectrum at any bond
  by canonicalizing through it and reading off the singular values; see
  `dmrg_pp/measurements/entanglement.py`.

## References

1. S. R. White, *Phys. Rev. Lett.* **69**, 2863 (1992).
2. U. Schollwöck, *Ann. Phys.* **326**, 96 (2011).
3. C. Hubig *et al.*, *SciPost Phys.* **3**, 002 (2017).
4. J. Hauschild and F. Pollmann, *SciPost Phys. Lect. Notes* 5 (2018).
