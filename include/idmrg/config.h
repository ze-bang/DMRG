//
// iDMRG Configuration Header
// High-performance infinite DMRG implementation using ITensor
//

#ifndef IDMRG_CONFIG_H
#define IDMRG_CONFIG_H

#include <string>
#include <iostream>

namespace idmrg {

//
// Version information
//
constexpr int IDMRG_VERSION_MAJOR = 1;
constexpr int IDMRG_VERSION_MINOR = 0;
constexpr int IDMRG_VERSION_PATCH = 0;

inline std::string version() {
    return std::to_string(IDMRG_VERSION_MAJOR) + "." +
           std::to_string(IDMRG_VERSION_MINOR) + "." +
           std::to_string(IDMRG_VERSION_PATCH);
}

//
// Default parameters
//
struct DefaultParams {
    // DMRG sweep parameters
    static constexpr int    max_sweeps = 20;
    static constexpr int    min_dim = 1;
    static constexpr int    max_dim = 200;
    static constexpr double cutoff = 1E-10;
    static constexpr int    niter = 2;
    static constexpr double noise = 0.0;
    
    // iDMRG specific
    static constexpr int    unit_cell_sweeps = 1;
    static constexpr double inverse_cutoff = 1E-8;
    static constexpr double convergence_threshold = 1E-8;
    
    // Output control
    static constexpr int    output_level = 1;
    static constexpr bool   quiet = false;
    static constexpr bool   show_overlap = false;
    
    // Performance options
    static constexpr int    write_dim = 0;  // Disable write-to-disk by default
    static constexpr bool   combine_mpo = true;
};

//
// Boundary condition types
//
enum class BoundaryCondition {
    Open,
    Periodic,
    Antiperiodic
};

inline std::string to_string(BoundaryCondition bc) {
    switch (bc) {
        case BoundaryCondition::Open: return "Open";
        case BoundaryCondition::Periodic: return "Periodic";
        case BoundaryCondition::Antiperiodic: return "Antiperiodic";
        default: return "Unknown";
    }
}

//
// Lattice geometry types
//
enum class LatticeType {
    Chain,
    Ladder,
    Square,
    Triangular,
    Honeycomb,
    Kagome,
    Custom
};

inline std::string to_string(LatticeType lt) {
    switch (lt) {
        case LatticeType::Chain: return "Chain";
        case LatticeType::Ladder: return "Ladder";
        case LatticeType::Square: return "Square";
        case LatticeType::Triangular: return "Triangular";
        case LatticeType::Honeycomb: return "Honeycomb";
        case LatticeType::Kagome: return "Kagome";
        case LatticeType::Custom: return "Custom";
        default: return "Unknown";
    }
}

//
// Model types
//
enum class ModelType {
    Heisenberg,
    XXZ,
    XY,
    Ising,
    Hubbard,
    ExtendedHubbard,
    tJ,
    Custom
};

inline std::string to_string(ModelType mt) {
    switch (mt) {
        case ModelType::Heisenberg: return "Heisenberg";
        case ModelType::XXZ: return "XXZ";
        case ModelType::XY: return "XY";
        case ModelType::Ising: return "Ising";
        case ModelType::Hubbard: return "Hubbard";
        case ModelType::ExtendedHubbard: return "Extended Hubbard";
        case ModelType::tJ: return "t-J";
        case ModelType::Custom: return "Custom";
        default: return "Unknown";
    }
}

//
// Spin types
//
enum class SpinType {
    Half,       // S = 1/2
    One,        // S = 1
    ThreeHalf,  // S = 3/2
    Two         // S = 2
};

inline int spinDim(SpinType st) {
    switch (st) {
        case SpinType::Half: return 2;
        case SpinType::One: return 3;
        case SpinType::ThreeHalf: return 4;
        case SpinType::Two: return 5;
        default: return 2;
    }
}

inline double spinValue(SpinType st) {
    switch (st) {
        case SpinType::Half: return 0.5;
        case SpinType::One: return 1.0;
        case SpinType::ThreeHalf: return 1.5;
        case SpinType::Two: return 2.0;
        default: return 0.5;
    }
}

} // namespace idmrg

#endif // IDMRG_CONFIG_H
