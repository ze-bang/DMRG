//
// Example: 2D Heisenberg Model on Square Lattice Cylinder (Finite DMRG)
//
// This benchmark computes the ground state of the 2D Heisenberg model
// on a square lattice with cylinder geometry using finite DMRG.
//
// The lattice is periodic in Y (around cylinder) and open in X.
//
// Reference energy per site for the 2D square lattice:
//   E/N ≈ -0.6694 (thermodynamic limit, QMC)
//

#include "idmrg/all.h"
#include <random>

using namespace itensor;
using namespace idmrg;

int main(int argc, char* argv[]) {
    // Default parameters
    int Ly = 4;          // Cylinder circumference
    int Lx = 8;          // Length in X direction
    double J = 1.0;      // Exchange coupling
    int maxDim = 500;    // Maximum bond dimension
    int numSweeps = 20;  // Number of DMRG sweeps
    
    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--Ly" && i+1 < argc) Ly = std::stoi(argv[++i]);
        else if (arg == "--Lx" && i+1 < argc) Lx = std::stoi(argv[++i]);
        else if (arg == "--J" && i+1 < argc) J = std::stod(argv[++i]);
        else if (arg == "--maxdim" && i+1 < argc) maxDim = std::stoi(argv[++i]);
        else if (arg == "--sweeps" && i+1 < argc) numSweeps = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            println("Usage: heisenberg_square_finite [options]");
            println("  --Lx N      Cylinder length (default: 8)");
            println("  --Ly N      Cylinder circumference (default: 4)");
            println("  --J val     Exchange coupling (default: 1.0)");
            println("  --maxdim N  Max bond dimension (default: 500)");
            println("  --sweeps N  Number of sweeps (default: 20)");
            return 0;
        }
    }
    
    int N = Lx * Ly;
    
    println("\n==============================================");
    println("  2D Heisenberg Model - Square Lattice Cylinder");
    println("            Finite DMRG Calculation           ");
    println("==============================================\n");
    
    printfln("System: %d x %d cylinder (%d sites)", Lx, Ly, N);
    printfln("  X direction: Open (length %d)", Lx);
    printfln("  Y direction: Periodic (circumference %d)", Ly);
    printfln("Exchange coupling J = %.4f", J);
    println("");
    
    printfln("Reference (thermodynamic limit): E/N ≈ -0.6694");
    println("");
    
    // Site ordering: snake pattern for better bond dimension
    // Row 0: sites 1, 2, 3, ..., Lx
    // Row 1: sites 2*Lx, 2*Lx-1, ..., Lx+1
    // etc.
    
    auto coords = [Lx, Ly](int i) -> std::pair<int,int> {
        int row = (i - 1) / Lx;
        int col = (i - 1) % Lx;
        if (row % 2 == 1) col = Lx - 1 - col;  // Snake pattern
        return {col, row};
    };
    
    auto siteIndex = [Lx, Ly, &coords](int x, int y) -> int {
        // Find site index from coordinates
        for (int i = 1; i <= Lx*Ly; ++i) {
            auto [cx, cy] = coords(i);
            if (cx == x && cy == y) return i;
        }
        return -1;
    };
    
    // Create sites
    auto sites = SpinHalf(N, {"ConserveQNs=", true});
    
    // Build Hamiltonian using AutoMPO
    auto ampo = AutoMPO(sites);
    
    println("Building 2D Heisenberg Hamiltonian...");
    
    int bondCount = 0;
    for (int i = 1; i <= N; ++i) {
        auto [x, y] = coords(i);
        
        // Horizontal bond (+x)
        if (x < Lx - 1) {
            int j = siteIndex(x + 1, y);
            if (j > 0 && j != i) {
                ampo += J,     "Sz", i, "Sz", j;
                ampo += 0.5*J, "S+", i, "S-", j;
                ampo += 0.5*J, "S-", i, "S+", j;
                bondCount++;
            }
        }
        
        // Vertical bond (+y) with periodic BC
        int y2 = (y + 1) % Ly;
        int j = siteIndex(x, y2);
        if (j > 0 && j != i) {
            ampo += J,     "Sz", i, "Sz", j;
            ampo += 0.5*J, "S+", i, "S-", j;
            ampo += 0.5*J, "S-", i, "S+", j;
            bondCount++;
        }
    }
    
    auto H = toMPO(ampo);
    printfln("Number of bonds: %d", bondCount);
    printfln("MPO bond dimension: %d\n", maxLinkDim(H));
    
    // Set up sweeps
    auto sweeps = Sweeps(numSweeps);
    sweeps.maxdim() = 50, 100, 200, 300, 400, maxDim;
    sweeps.cutoff() = 1E-8, 1E-10, 1E-12;
    sweeps.niter() = 4, 3, 2;
    sweeps.noise() = 1E-6, 1E-7, 1E-8, 1E-10, 0;
    
    println("Sweep schedule:");
    println(sweeps);
    println("");
    
    // Initialize with Néel state
    println("Initializing MPS with Néel pattern...");
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        auto [x, y] = coords(i);
        if ((x + y) % 2 == 0) {
            state.set(i, "Up");
        } else {
            state.set(i, "Dn");
        }
    }
    auto psi = MPS(state);
    printfln("Initial MPS bond dimension: %d\n", maxLinkDim(psi));
    
    // Run DMRG
    println("Starting DMRG optimization...\n");
    
    Timer timer;
    timer.start();
    
    auto [energy, psi_gs] = dmrg(H, psi, sweeps, {"Quiet", false});
    
    timer.stop();
    
    // Results
    println("\n==============================================");
    println("                  Results                    ");
    println("==============================================");
    printfln("  Ground state energy: %.14f", energy);
    printfln("  Energy per site: %.14f", energy / N);
    printfln("  Final bond dimension: %d", maxLinkDim(psi_gs));
    printfln("  Time: %s", timer.elapsedStr().c_str());
    println("");
    
    // Compare to reference
    double E_ref = -0.6694;
    double E_per_site = energy / N;
    double error_pct = 100.0 * std::abs(E_per_site - E_ref) / std::abs(E_ref);
    
    printfln("Comparison to thermodynamic limit (E/N ≈ %.4f):", E_ref);
    printfln("  Finite-size E/N = %.6f", E_per_site);
    printfln("  Difference: %.4f%%", error_pct);
    
    // Note: finite-size effects are significant!
    println("\nNote: Finite-size effects are significant for small systems.");
    println("Energy approaches -0.6694 as Lx, Ly → ∞ and χ → ∞.");
    println("");
    
    // Compute local magnetization
    println("Local <Sz> per site:");
    for (int y = 0; y < Ly; ++y) {
        print("  ");
        for (int x = 0; x < Lx; ++x) {
            int i = siteIndex(x, y);
            psi_gs.position(i);
            auto Sz_op = op(sites, "Sz", i);
            auto val = eltC(dag(prime(psi_gs(i), "Site")) * Sz_op * psi_gs(i));
            printf("%+.4f ", real(val));
        }
        println("");
    }
    println("");
    
    // Note: Entanglement entropy is printed during DMRG sweeps at vN Entropy lines
    println("(See 'vN Entropy at center bond' in sweep output above)");
    println("");
    
    return 0;
}
