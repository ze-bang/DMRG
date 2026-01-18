//
// Example: Anisotropic Spin Model on Triangular Lattice
//
// This example demonstrates the iDMRG simulation of an anisotropic
// spin-1/2 model on a triangular lattice with bond-dependent interactions.
//
// The Hamiltonian is:
//
// H = H_XXZ + H_bd
//
// H_XXZ = J Σ_{<ij>} (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)
//
// H_bd = Σ_{<ij>} { 2J_±± (cos(φ̃_α)[x,y]_ij - sin(φ̃_α){x,y}_ij)
//                 + J_z± (cos(φ̃_α){y,z}_ij - sin(φ̃_α){x,z}_ij) }
//
// where:
//   [a,b]_ij = S^a_i S^a_j - S^b_i S^b_j
//   {a,b}_ij = S^a_i S^b_j + S^b_i S^a_j
//   φ̃_α = {0, -2π/3, 2π/3} for bonds along δ_1, δ_2, δ_3
//

#include "idmrg/all.h"
#include "idmrg/models/anisotropic_triangular.h"

using namespace itensor;
using namespace idmrg;

int main(int argc, char* argv[]) {
    // =========================================
    // System parameters
    // =========================================
    
    // Lattice dimensions (Lx x Ly cylinder)
    int Lx = 4;   // Length in x-direction (will grow in iDMRG)
    int Ly = 4;   // Width in y-direction (circumference of cylinder)
    
    // Hamiltonian parameters
    double J      = 1.0;   // Exchange coupling
    double Delta  = 0.8;   // Ising anisotropy (0 ≤ Δ ≤ 1 for easy-plane)
    double Jpmpm  = 0.05;  // J_±± bond-dependent coupling
    double Jzpm   = 0.1;   // J_z± mixed coupling
    double hz     = 0.0;   // External magnetic field
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--Lx" && i + 1 < argc) Lx = std::stoi(argv[++i]);
        else if (arg == "--Ly" && i + 1 < argc) Ly = std::stoi(argv[++i]);
        else if (arg == "--J" && i + 1 < argc) J = std::stod(argv[++i]);
        else if (arg == "--Delta" && i + 1 < argc) Delta = std::stod(argv[++i]);
        else if (arg == "--Jpmpm" && i + 1 < argc) Jpmpm = std::stod(argv[++i]);
        else if (arg == "--Jzpm" && i + 1 < argc) Jzpm = std::stod(argv[++i]);
        else if (arg == "--hz" && i + 1 < argc) hz = std::stod(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            println("Usage: anisotropic_triangular [options]");
            println("");
            println("Options:");
            println("  --Lx N       Length in x-direction (default: 4)");
            println("  --Ly N       Width in y-direction (default: 4)");
            println("  --J val      Exchange coupling J (default: 1.0)");
            println("  --Delta val  Ising anisotropy Δ (default: 0.8)");
            println("  --Jpmpm val  J_±± bond-dependent coupling (default: 0.05)");
            println("  --Jzpm val   J_z± mixed coupling (default: 0.1)");
            println("  --hz val     External field (default: 0.0)");
            return 0;
        }
    }
    
    // =========================================
    // Print simulation info
    // =========================================
    
    println("=============================================");
    println("Anisotropic Triangular Lattice iDMRG");
    println("=============================================");
    println("");
    printfln("Lattice: %d x %d triangular", Lx, Ly);
    printfln("Total sites in unit cell: %d", Lx * Ly);
    println("");
    println("Hamiltonian: H = H_XXZ + H_bd");
    println("");
    println("H_XXZ = J Σ (S^x S^x + S^y S^y + Δ S^z S^z)");
    println("H_bd  = Σ { 2J_±±(cos(φ̃)[x,y] - sin(φ̃){x,y})");
    println("          + J_z±(cos(φ̃){y,z} - sin(φ̃){x,z}) }");
    println("");
    println("Parameters:");
    printfln("  J     = %.4f", J);
    printfln("  Δ     = %.4f (0 ≤ Δ ≤ 1 for easy-plane)", Delta);
    printfln("  J_±±  = %.4f", Jpmpm);
    printfln("  J_z±  = %.4f", Jzpm);
    printfln("  h_z   = %.4f", hz);
    println("");
    println("Phase factors φ̃_α for bond directions:");
    println("  δ_1 (a1): φ̃ = 0");
    println("  δ_2 (a2): φ̃ = -2π/3");
    println("  δ_3 (a3): φ̃ = 2π/3");
    println("");
    
    // =========================================
    // Create lattice and sites
    // =========================================
    
    // Create triangular lattice with periodic boundary in y
    auto lattice = TriangularLattice(Lx, Ly);
    lattice.setYBC(BoundaryCondition::Periodic);  // Cylinder geometry
    
    int N = lattice.numSites();
    printfln("Number of sites: %d", N);
    printfln("Number of bonds: %d", lattice.bonds().size());
    
    // Create spin-1/2 site set
    // Note: For complex Hamiltonians with bond-dependent terms,
    // we don't conserve Sz as the Hamiltonian breaks U(1) symmetry
    bool conserveQNs = (std::abs(Jpmpm) < 1e-10 && std::abs(Jzpm) < 1e-10);
    auto sites = SpinHalf(N, {"ConserveQNs=", conserveQNs});
    
    if (conserveQNs) {
        println("Conserving total Sz quantum number");
    } else {
        println("Not conserving Sz (Hamiltonian breaks U(1) symmetry)");
    }
    println("");
    
    // =========================================
    // Build Hamiltonian MPO
    // =========================================
    
    println("Building Hamiltonian MPO...");
    
    auto H = AnisotropicTriangularMPO(sites, lattice, {
        "J", J,
        "Delta", Delta,
        "Jpmpm", Jpmpm,
        "Jzpm", Jzpm,
        "hz", hz,
        "Infinite", true
    });
    
    printfln("MPO bond dimension: %d", maxLinkDim(H));
    println("");
    
    // =========================================
    // Configure DMRG sweeps
    // =========================================
    
    auto sweeps = Sweeps(30);
    sweeps.maxdim() = 50, 100, 200, 400, 800, 1000;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 10, 5, 3, 2;
    sweeps.noise() = 1E-6, 1E-7, 1E-8, 0;
    
    println("DMRG sweep schedule:");
    println(sweeps);
    println("");
    
    // =========================================
    // Initialize MPS
    // =========================================
    
    println("Initializing MPS...");
    
    // Random initial state
    auto psi = randomMPS(sites);
    printfln("Initial MPS bond dimension: %d", maxLinkDim(psi));
    println("");
    
    // =========================================
    // Run iDMRG
    // =========================================
    
    println("Starting iDMRG...");
    println("");
    
    auto result = idmrg(psi, H, sweeps, {"OutputLevel", 1});
    
    println("");
    println("=============================================");
    println("Results");
    println("=============================================");
    printfln("Ground state energy: %.14f", result.energy);
    printfln("Energy per site: %.14f", result.energy / N);
    printfln("Final MPS bond dimension: %d", maxLinkDim(psi));
    println("");
    
    // =========================================
    // Measure observables
    // =========================================
    
    println("Measuring local observables...");
    
    // Measure <Sz> on each site
    println("");
    println("Local magnetization <S^z_i>:");
    double total_Sz = 0.0;
    for (int i = 1; i <= N; ++i) {
        auto Sz_i = op(sites, "Sz", i);
        psi.position(i);
        auto val = eltC(dag(prime(psi(i), "Site")) * Sz_i * psi(i));
        printfln("  Site %d: <Sz> = %.6f", i, val.real());
        total_Sz += val.real();
    }
    printfln("Total <Sz> = %.6f", total_Sz);
    printfln("Average <Sz> per site = %.6f", total_Sz / N);
    
    // Measure nearest-neighbor correlations
    println("");
    println("Sample nearest-neighbor correlations:");
    
    int num_samples = std::min(5, (int)lattice.bonds().size());
    for (int b = 0; b < num_samples; ++b) {
        auto& bond = lattice.bonds()[b];
        int i = bond.s1;
        int j = bond.s2;
        
        // <Sz_i Sz_j>
        auto SzSz = correlationFunction(psi, sites, "Sz", i, "Sz", j);
        
        printfln("  Bond (%d,%d) [%s]: <Sz Sz> = %.6f", 
                 i, j, bond.type.c_str(), SzSz.real());
    }
    
    println("");
    println("Done!");
    
    return 0;
}
