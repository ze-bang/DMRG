# DMRG++

[![CI](https://github.com/dmrg-pp/dmrg_pp/actions/workflows/ci.yml/badge.svg)](https://github.com/dmrg-pp/dmrg_pp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DMRG++** is a production-grade [Density Matrix Renormalization Group](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group)
library in pure Python (NumPy + SciPy under the hood).  It targets ground-state
and low-lying excited-state calculations of one-dimensional and quasi-one-dimensional
quantum lattice models, built on a clean Matrix Product State (MPS) /
Matrix Product Operator (MPO) core.

## Highlights

- **Clean core.**  Tiny MPS / MPO classes with explicit canonical-form
  bookkeeping, deterministic gauge-fixing in QR decompositions, and SVD with
  rich truncation diagnostics.
- **Symbolic MPO builder.**  A finite-state-machine `MPOBuilder` accepts
  arbitrary tensor-product terms and produces MPOs with the optimal bond
  dimension for sparse local Hamiltonians (`D = 5` for nearest-neighbour
  Heisenberg/TFIM out of the box).
- **Two-site DMRG.**  Variational ground-state search with a ramped
  bond-dimension schedule, dynamic SVD truncation, and ARPACK / dense Lanczos
  eigensolvers chosen automatically based on local-block size.
- **Models out of the box.**  Spin-½ XXZ Heisenberg chain, transverse-field
  Ising, and one-band Hubbard chain (with proper Jordan–Wigner strings).
- **Measurements.**  Local observables, two-point correlators, full
  correlation matrices, Schmidt spectra, von-Neumann entropies, and the
  entanglement Hamiltonian spectrum.
- **I/O & CLI.**  HDF5 checkpoints, YAML / pydantic-validated experiment
  configs, and a Typer-based `dmrg run` / `dmrg info` command line.
- **Production hygiene.**  Type-hinted public API, ruff-formatted, mypy-checked,
  pytest with integration tests against exact diagonalization, GitHub Actions
  CI matrix on Linux + macOS / Python 3.10–3.12.

## Installation

```bash
pip install -e ".[dev]"          # editable install with the full dev tool-chain
```

Hard runtime deps: `numpy ≥ 1.24`, `scipy ≥ 1.11`, `h5py`, `pydantic ≥ 2.5`,
`pyyaml`, `typer`, `rich`.  Tested on Python 3.10, 3.11, 3.12.

## Quick start (Python API)

```python
from dmrg_pp import DMRGConfig, run_dmrg
from dmrg_pp.models import HeisenbergXXZ

model = HeisenbergXXZ(L=40, Jxy=1.0, Jz=1.0)
config = DMRGConfig(
    n_sweeps=12,
    max_bond=[16, 32, 64, 128, 128, 256],   # ramped bond-dim schedule
    cutoff=1e-10,
    e_tol=1e-9,
    seed=2026,
)
result = run_dmrg(model.mpo(), config=config)
print(f"E = {result.energy:.12f}    E/L = {result.energy / 40:.12f}")
```

For an `L = 40` Heisenberg AFM chain this reproduces the Bethe-ansatz
energy density `e_∞ = 1/4 − ln 2 ≈ −0.4431` to ~4 significant digits in a few
seconds.  The same script (with central-charge fitting) for the Ising critical
point lives in `examples/tfim.py`.

## Quick start (CLI)

Define an experiment in YAML:

```yaml
# examples/configs/heisenberg_chain.yaml
name: heisenberg_L40
model:
  kind: heisenberg
  L: 40
  Jxy: 1.0
  Jz: 1.0
runtime:
  n_sweeps: 12
  max_bond: [16, 32, 64, 128, 128, 256]
  cutoff: 1.0e-10
  e_tol: 1.0e-9
  seed: 2026
output: runs/heisenberg_L40.h5
```

then run

```bash
dmrg run examples/configs/heisenberg_chain.yaml
dmrg info runs/heisenberg_L40.h5
```

## Architecture

```
src/dmrg_pp/
├── tensors/         # SVD / QR / contractions (pure NumPy)
├── states/          # MPS class, canonical forms, random-state builder
├── operators/       # MPO class, local-op catalogs, FSM-based builder
├── models/          # Heisenberg, Ising, Hubbard
├── algorithms/      # environments, eigensolver, two-site DMRG sweep engine
├── measurements/    # observables, correlations, entanglement diagnostics
├── io/              # HDF5 checkpoints, pydantic config schemas
├── utils/           # logging
└── cli.py           # `dmrg` command-line entry point
```

The package is intentionally layered: each module depends only on layers below
it, which makes the library straightforward to extend (new models, new
algorithms, alternative tensor backends).

## Conventions & implementation notes

- **MPS tensors** have shape `(D_left, d, D_right)`; boundary bonds are 1.
- **MPO tensors** have shape `(D_left, d_out, d_in, D_right)`; the action is
  `(Ô|ψ⟩)_σ = W^σ_τ ψ_τ`.
- **Environments** are 3-leg tensors with index ordering
  `(bra MPS, MPO, ket MPS)`; see `dmrg_pp/algorithms/environments.py`.
- **Truncation** uses a *relative discarded-weight* cutoff
  (`Σ_{i>χ} s_i² < cutoff · Σ s_i²`), matching the convention used by
  ITensor and TenPy.
- **Lanczos** uses `scipy.sparse.linalg.eigsh` with a graceful fallback to a
  hand-rolled re-orthogonalized Lanczos when ARPACK declines to converge,
  plus a dense `eigh` shortcut for tiny effective Hilbert spaces.
- **Fermions** use the Jordan–Wigner convention with the parity string
  absorbed into the *left* site of every bilinear; see `models/hubbard.py`.

## Validation

The integration test suite (`pytest -m integration`) checks DMRG energies
against full exact diagonalization for:

- Heisenberg XXZ with isotropic and anisotropic couplings (L = 6, 8)
- Transverse-field Ising both off-critical and critical (L = 6, 8)
- Half-filled Hubbard at finite U (L = 4)

All energies agree with ED to better than `1e-8`.

## Documentation

- A short, code-focused notes file: [`docs/theory.md`](docs/theory.md).
- A full pedagogical companion in LaTeX:
  [`docs/dmrg_pedagogical.tex`](docs/dmrg_pedagogical.tex) — derives the two-site
  finite-system DMRG algorithm from first principles (MPS, canonical forms,
  truncated SVD, FSM-based MPO construction, environments, effective
  Hamiltonian, Lanczos, sweep update) and maps every theoretical concept to
  the corresponding module in this repository. The compiled
  [`docs/dmrg_pedagogical.pdf`](docs/dmrg_pedagogical.pdf) is checked in for
  convenience; rebuild with

  ```bash
  cd docs && make
  ```

  (requires a TeX Live installation with `pdflatex`, and `latexmk` if you
  want incremental builds via `make watch`).

## Roadmap

- [ ] Block-sparse tensors for U(1) / SU(2) symmetry conservation
- [ ] Single-site DMRG with Hubig-style subspace expansion
- [ ] Time-evolution: TEBD, TDVP (1- and 2-site)
- [ ] Cylindrical 2D lattices via snake mapping
- [ ] GPU backend (CuPy / JAX)

Contributions welcome — see `CONTRIBUTING.md` (forthcoming) for setup and PR
guidelines.

## License

MIT — see [LICENSE](LICENSE).
