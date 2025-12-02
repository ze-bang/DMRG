# Infinite DMRG (iDMRG) for Generic 1D/2D Lattices

A high-performance implementation of the infinite Density Matrix Renormalization Group (iDMRG) algorithm using the ITensor library. This implementation supports generic 1D chains and 2D lattice geometries mapped to 1D with various boundary conditions.

## Features

- **Generic Lattice Support**: 1D chains, 2D square/triangular/honeycomb lattices
- **Multiple Physical Systems**:
  - Spin-1/2 and Spin-1 Heisenberg models
  - Hubbard model (with quantum number conservation)
  - t-J model
  - Custom Hamiltonians via AutoMPO
- **High Performance**:
  - ITensor C++17 backend with optimized tensor contractions
  - OpenMP parallelization support
  - Optional Intel MKL integration
  - Memory-efficient algorithms for large bond dimensions
- **Advanced Features**:
  - Quantum number conservation (U(1), SU(2))
  - Customizable unit cells
  - Correlation function measurements
  - Checkpointing and restart capability

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 6+)
- CMake 3.14+
- ITensor v3 library
- BLAS/LAPACK libraries

## Installation

1. Install ITensor v3:
```bash
git clone https://github.com/ITensor/ITensor itensor
cd itensor
cp options.mk.sample options.mk
# Edit options.mk to configure BLAS/LAPACK
make -j4
```

2. Build iDMRG:
```bash
mkdir build && cd build
cmake .. -DITENSOR_DIR=/path/to/itensor
make -j4
```

## Quick Start

```cpp
#include "idmrg/idmrg.h"
#include "idmrg/lattice/chain.h"
#include "idmrg/models/heisenberg.h"

using namespace itensor;
using namespace idmrg;

int main() {
    // Define unit cell size
    int Nuc = 2;  // Sites per unit cell
    
    // Create 1D chain lattice
    auto lattice = Chain(2 * Nuc);
    
    // Create spin-1/2 site set
    auto sites = SpinHalf(lattice.numSites(), {"ConserveQNs=", true});
    
    // Build Heisenberg Hamiltonian
    auto H = HeisenbergMPO(sites, lattice, {
        {"J", 1.0},
        {"Infinite", true}
    });
    
    // Configure DMRG sweeps
    auto sweeps = Sweeps(20);
    sweeps.maxdim() = 20, 50, 100, 200, 400;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 3, 2;
    
    // Initialize MPS
    auto psi = randomMPS(sites);
    
    // Run iDMRG
    auto result = idmrg(psi, H, sweeps, {"OutputLevel", 1});
    
    printfln("Energy per site = %.14f", result.energy / (2 * Nuc));
    
    return 0;
}
```

## Project Structure

```
├── include/
│   └── idmrg/
│       ├── idmrg.h              # Main iDMRG algorithm
│       ├── config.h             # Configuration options
│       ├── observer.h           # Measurement observer
│       ├── lattice/
│       │   ├── lattice.h        # Base lattice class
│       │   ├── chain.h          # 1D chain
│       │   ├── square.h         # 2D square lattice
│       │   ├── triangular.h     # 2D triangular lattice
│       │   └── honeycomb.h      # 2D honeycomb lattice
│       ├── models/
│       │   ├── heisenberg.h     # Heisenberg model
│       │   ├── hubbard.h        # Hubbard model
│       │   └── tj.h             # t-J model
│       └── util/
│           ├── timer.h          # Performance timing
│           └── io.h             # I/O utilities
├── examples/
│   ├── heisenberg_chain.cc      # 1D Heisenberg example
│   ├── hubbard_2d.cc            # 2D Hubbard example
│   └── generic_lattice.cc       # Custom lattice example
└── CMakeLists.txt
```

## Advanced Usage

### 2D Lattice with Cylinder Geometry

```cpp
// Create 6-leg cylinder with 2 unit cells along the cylinder
auto lattice = SquareLattice(6, 4, {"YBC", "Periodic"});
auto sites = SpinHalf(lattice.numSites(), {"ConserveQNs=", true});
auto H = HeisenbergMPO(sites, lattice, {"J", 1.0, "Infinite", true});
```

### Custom Measurements

```cpp
// Create observer for custom measurements
auto obs = iDMRGObserver(psi, sites);
obs.addMeasurement("Sz", [&](int site) {
    return inner(psi, op(sites, "Sz", site), psi);
});

auto result = idmrg(psi, H, sweeps, obs, {"OutputLevel", 1});
```

### Checkpointing

```cpp
// Save state for restart
writeToFile("checkpoint.idmrg", result);

// Resume from checkpoint
idmrgRVal last_result;
readFromFile("checkpoint.idmrg", last_result);
auto result = idmrg(psi, H, last_result, sweeps, {"OutputLevel", 1});
```

## Performance Tips

1. **Bond Dimension**: Start with small maxdim and gradually increase
2. **Cutoff**: Use `cutoff = 1E-10` for most calculations
3. **OpenMP**: Set `OMP_NUM_THREADS` for parallel tensor operations
4. **Memory**: Use `WriteDim` option for large bond dimensions
5. **2D Systems**: Use appropriate cylinder width (typically 4-8 for cylinders)

## References

1. White, S. R. (1992). Density matrix formulation for quantum renormalization groups. *Physical Review Letters*, 69(19), 2863.
2. McCulloch, I. P. (2008). Infinite size density matrix renormalization group, revisited. *arXiv:0804.2509*.
3. ITensor documentation: https://itensor.org

## License

MIT License - see LICENSE file for details.
