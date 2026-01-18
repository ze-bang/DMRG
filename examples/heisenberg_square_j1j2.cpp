//
// Example: J1-J2 Heisenberg Model on 2D Square Lattice
//
// This benchmark computes the ground state of the frustrated J1-J2 Heisenberg
// model on a square lattice cylinder using iDMRG.
//
// Hamiltonian:
//   H = J1 Σ_{<i,j>} S_i · S_j + J2 Σ_{<<i,j>>} S_i · S_j
//
// where <i,j> are nearest neighbors and <<i,j>> are next-nearest neighbors.
//
// The model exhibits a rich phase diagram:
//   - J2/J1 < 0.4:  Néel antiferromagnetic order
//   - 0.4 < J2/J1 < 0.6: Possible spin liquid or valence bond solid
//   - J2/J1 > 0.6:  Stripe (collinear) antiferromagnetic order
//
// Reference values for energy per site (from various methods):
//   J2/J1 = 0.0: E/N ≈ -0.6694 (square lattice Heisenberg AFM)
//   J2/J1 = 0.5: E/N ≈ -0.495  (maximally frustrated point)
//

#include "idmrg/all.h"

using namespace itensor;
using namespace idmrg;

//
// Build J1-J2 Heisenberg Hamiltonian on square lattice
//
MPO buildJ1J2Hamiltonian(SiteSet const& sites, 
                         SquareLattice const& lattice,
                         double J1, double J2,
                         bool infinite = false)
{
    AutoMPO ampo(sites);
    int Lx = lattice.Lx();
    int Ly = lattice.Ly();
    
    // Get boundary conditions
    bool yPeriodic = (lattice.yBC() == BoundaryCondition::Periodic);
    bool xPeriodic = (lattice.xBC() == BoundaryCondition::Periodic);
    
    for (int x = 0; x < Lx; ++x) {
        for (int y = 0; y < Ly; ++y) {
            int s = lattice.siteIndex(x, y);
            
            // J1: Nearest-neighbor bonds
            
            // +x direction
            if (x < Lx - 1 || xPeriodic) {
                int x2 = (x + 1) % Lx;
                int t = lattice.siteIndex(x2, y);
                ampo += J1,     "Sz", s, "Sz", t;
                ampo += 0.5*J1, "S+", s, "S-", t;
                ampo += 0.5*J1, "S-", s, "S+", t;
            }
            
            // +y direction
            if (y < Ly - 1 || yPeriodic) {
                int y2 = (y + 1) % Ly;
                int t = lattice.siteIndex(x, y2);
                ampo += J1,     "Sz", s, "Sz", t;
                ampo += 0.5*J1, "S+", s, "S-", t;
                ampo += 0.5*J1, "S-", s, "S+", t;
            }
            
            // J2: Next-nearest-neighbor bonds (diagonals)
            if (std::abs(J2) > 1e-10) {
                // +x+y diagonal
                if ((x < Lx - 1 || xPeriodic) && (y < Ly - 1 || yPeriodic)) {
                    int x2 = (x + 1) % Lx;
                    int y2 = (y + 1) % Ly;
                    int t = lattice.siteIndex(x2, y2);
                    ampo += J2,     "Sz", s, "Sz", t;
                    ampo += 0.5*J2, "S+", s, "S-", t;
                    ampo += 0.5*J2, "S-", s, "S+", t;
                }
                
                // +x-y diagonal
                if ((x < Lx - 1 || xPeriodic) && (y > 0 || yPeriodic)) {
                    int x2 = (x + 1) % Lx;
                    int y2 = (y - 1 + Ly) % Ly;
                    int t = lattice.siteIndex(x2, y2);
                    ampo += J2,     "Sz", s, "Sz", t;
                    ampo += 0.5*J2, "S+", s, "S-", t;
                    ampo += 0.5*J2, "S-", s, "S+", t;
                }
            }
        }
    }
    
    auto H = toMPO(ampo);
    
    // Set up infinite boundary if needed
    if (infinite && sites.length() >= 1) {
        auto ll = leftLinkIndex(H, 1);
        H.ref(0) = ITensor(dag(ll));
        H.ref(0).set(dag(ll)(2), 1.0);
        
        auto rl = rightLinkIndex(H, sites.length());
        H.ref(sites.length() + 1) = ITensor(rl);
        H.ref(sites.length() + 1).set(rl(1), 1.0);
    }
    
    return H;
}

int main(int argc, char* argv[]) {
    // =========================================
    // Default parameters
    // =========================================
    
    int Ly = 4;         // Cylinder circumference
    int Lx = 2;         // Columns per unit cell (half of total)
    double J1 = 1.0;    // Nearest-neighbor coupling
    double J2 = 0.0;    // Next-nearest-neighbor coupling
    int maxSweeps = 30; // Maximum iDMRG sweeps
    int maxDim = 1000;  // Maximum bond dimension
    bool computeSq = true;  // Compute structure factor
    
    // =========================================
    // Parse command line arguments
    // =========================================
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--Ly" && i + 1 < argc) Ly = std::stoi(argv[++i]);
        else if (arg == "--Lx" && i + 1 < argc) Lx = std::stoi(argv[++i]);
        else if (arg == "--J1" && i + 1 < argc) J1 = std::stod(argv[++i]);
        else if (arg == "--J2" && i + 1 < argc) J2 = std::stod(argv[++i]);
        else if (arg == "--maxdim" && i + 1 < argc) maxDim = std::stoi(argv[++i]);
        else if (arg == "--sweeps" && i + 1 < argc) maxSweeps = std::stoi(argv[++i]);
        else if (arg == "--no-sq") computeSq = false;
        else if (arg == "--help" || arg == "-h") {
            println("Usage: heisenberg_square_j1j2 [options]");
            println("");
            println("Options:");
            println("  --Ly N        Cylinder circumference (default: 4)");
            println("  --Lx N        Columns per unit cell (default: 2)");
            println("  --J1 val      Nearest-neighbor coupling (default: 1.0)");
            println("  --J2 val      Next-nearest-neighbor coupling (default: 0.0)");
            println("  --maxdim N    Maximum bond dimension (default: 1000)");
            println("  --sweeps N    Maximum iDMRG sweeps (default: 30)");
            println("  --no-sq       Skip structure factor calculation");
            return 0;
        }
    }
    
    // =========================================
    // Print simulation info
    // =========================================
    
    println("\n=============================================");
    println("  J1-J2 Heisenberg Model on Square Lattice  ");
    println("           Cylinder Geometry (iDMRG)        ");
    println("=============================================\n");
    
    int N = 2 * Lx * Ly;  // Total sites (2 unit cells for iDMRG)
    int Nuc = Lx * Ly;    // Sites per unit cell
    
    printfln("Lattice geometry:");
    printfln("  Cylinder width (Ly): %d (periodic)", Ly);
    printfln("  Unit cell length (Lx): %d columns", Lx);
    printfln("  Sites per unit cell: %d", Nuc);
    printfln("  Total sites: %d (2 unit cells)", N);
    println("");
    
    printfln("Hamiltonian parameters:");
    printfln("  J1 = %.4f (nearest neighbor)", J1);
    printfln("  J2 = %.4f (next-nearest neighbor)", J2);
    printfln("  J2/J1 = %.4f", J2/J1);
    println("");
    
    // Expected phase based on J2/J1 ratio
    double ratio = J2 / J1;
    if (ratio < 0.4) {
        println("Expected phase: Néel antiferromagnet (J2/J1 < 0.4)");
    } else if (ratio < 0.6) {
        println("Expected phase: Frustrated regime - possible spin liquid or VBS");
    } else {
        println("Expected phase: Stripe antiferromagnet (J2/J1 > 0.6)");
    }
    println("");
    
    // Reference energies
    println("Reference energy per site (thermodynamic limit):");
    printfln("  J2/J1 = 0.0:  E/N ≈ -0.6694 (Néel AFM)");
    printfln("  J2/J1 = 0.5:  E/N ≈ -0.495  (frustrated)");
    println("");
    
    // =========================================
    // Create lattice and sites
    // =========================================
    
    // Create square lattice with snake ordering for better entanglement
    auto lattice = SquareLattice(2 * Lx, Ly, SiteOrdering::Snake);
    lattice.setYBC(BoundaryCondition::Periodic);  // Periodic in y (cylinder)
    lattice.setXBC(BoundaryCondition::Open);      // Open in x (infinite direction)
    lattice.setInfinite(true);
    
    println(lattice.info());
    
    // Create spin-1/2 sites without Sz conservation for iDMRG compatibility
    auto sites = SpinHalf(N, {"ConserveQNs=", false});
    
    // =========================================
    // Build Hamiltonian
    // =========================================
    
    println("Building J1-J2 Hamiltonian MPO...");
    auto H = buildJ1J2Hamiltonian(sites, lattice, J1, J2, true);
    printfln("MPO bond dimension: %d\n", maxLinkDim(H));
    
    // =========================================
    // Configure DMRG sweeps
    // =========================================
    
    auto sweeps = Sweeps(maxSweeps);
    
    // Gradual increase in bond dimension
    if (maxDim >= 1000) {
        sweeps.maxdim() = 100, 200, 400, 600, 800, 1000, maxDim;
    } else {
        sweeps.maxdim() = 50, 100, 200, 400, std::min(600, maxDim), maxDim;
    }
    
    sweeps.cutoff() = 1E-8, 1E-10, 1E-12;
    sweeps.niter() = 4, 3, 2;
    sweeps.noise() = 1E-6, 1E-7, 1E-8, 1E-9, 0;
    
    println("DMRG sweep schedule:");
    println(sweeps);
    println("");
    
    // =========================================
    // Initialize MPS
    // =========================================
    
    println("Initializing MPS with Néel state...");
    
    // Start with checkerboard Néel pattern
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        auto [x, y] = lattice.siteCoords(i);
        if ((x + y) % 2 == 0) {
            state.set(i, "Up");
        } else {
            state.set(i, "Dn");
        }
    }
    auto psi = MPS(state);
    printfln("Initial MPS bond dimension: %d\n", maxLinkDim(psi));
    
    // =========================================
    // Run iDMRG
    // =========================================
    
    println("Starting iDMRG optimization...\n");
    
    Timer timer;
    timer.start();
    
    auto result = idmrg::idmrg(psi, H, sweeps, {"OutputLevel", 1});
    
    timer.stop();
    
    // =========================================
    // Print results
    // =========================================
    
    println("\n=============================================");
    println("                 Results                    ");
    println("=============================================");
    printfln("  Ground state energy (per sweep): %.14f", result.energy);
    printfln("  Energy per site: %.14f", result.energy_per_site);
    printfln("  Entanglement entropy: %.10f", result.entropy);
    printfln("  Final bond dimension: %d", maxLinkDim(psi));
    printfln("  Total iDMRG steps: %d", result.num_sweeps);
    printfln("  Converged: %s", result.converged ? "Yes" : "No");
    printfln("  Computation time: %s", timer.elapsedStr().c_str());
    println("=============================================\n");
    
    // =========================================
    // Measure local observables
    // =========================================
    
    println("Local magnetization <S^z_i> in unit cell:");
    println("Site (x,y)    <Sz>");
    println("-------------------");
    
    double total_Sz = 0.0;
    double stagger_Sz = 0.0;  // Staggered magnetization
    
    for (int i = 1; i <= Nuc; ++i) {
        psi.position(i);
        auto Sz_op = op(sites, "Sz", i);
        auto val = elt(dag(prime(psi(i), "Site")) * Sz_op * psi(i));
        
        auto [x, y] = lattice.siteCoords(i);
        printfln("%3d (%d,%d)   %+.6f", i, x, y, val);
        
        total_Sz += val;
        stagger_Sz += ((x + y) % 2 == 0 ? 1.0 : -1.0) * val;
    }
    
    println("-------------------");
    printfln("Total <Sz>: %.6f", total_Sz);
    printfln("Staggered magnetization: %.6f", std::abs(stagger_Sz / Nuc));
    println("");
    
    // =========================================
    // Compute spin-spin correlations
    // =========================================
    
    if (computeSq) {
        println("Computing spin-spin correlations...\n");
        
        int ref_site = Nuc / 2;  // Reference site (center of unit cell)
        psi.position(ref_site);
        
        println("Distance    <S_ref · S_j>    Site j (x,y)");
        println("-------------------------------------------");
        
        std::vector<double> corr_values;
        std::vector<double> distances;
        
        auto [x0, y0] = lattice.siteCoords(ref_site);
        
        for (int j = 1; j <= Nuc; ++j) {
            auto [xj, yj] = lattice.siteCoords(j);
            
            // Physical distance on cylinder
            double dx = xj - x0;
            double dy = yj - y0;
            // Handle periodic y
            if (std::abs(dy) > Ly / 2.0) {
                dy = (dy > 0) ? dy - Ly : dy + Ly;
            }
            double dist = std::sqrt(dx*dx + dy*dy);
            
            // Compute <S_ref · S_j>
            Cplx corr = spinSpinCorrelation(psi, sites, ref_site, j);
            
            printfln("%.4f       %+.8f      %d (%d,%d)", 
                     dist, corr.real(), j, xj, yj);
            
            corr_values.push_back(corr.real());
            distances.push_back(dist);
        }
        
        println("");
        
        // =========================================
        // Static structure factor S(q)
        // =========================================
        
        println("Computing static structure factor S(q)...\n");
        
        auto Sq_result = staticStructureFactor2D(psi, sites, lattice, 10, 10);
        
        // Find peak position
        int peak_idx = 0;
        double peak_val = 0.0;
        for (size_t iq = 0; iq < Sq_result.Sq.size(); ++iq) {
            if (Sq_result.Sq[iq].real() > peak_val) {
                peak_val = Sq_result.Sq[iq].real();
                peak_idx = iq;
            }
        }
        
        println("Structure factor peak:");
        printfln("  q = (%.4f, %.4f) [in units of 2π]", 
                 Sq_result.qpoints[peak_idx][0] / (2*M_PI),
                 Sq_result.qpoints[peak_idx][1] / (2*M_PI));
        printfln("  S(q_peak) = %.6f", peak_val);
        println("");
        
        // Expected peak positions
        if (ratio < 0.4) {
            println("Expected: Peak at (π, π) for Néel order");
        } else if (ratio > 0.6) {
            println("Expected: Peak at (π, 0) or (0, π) for stripe order");
        }
    }
    
    // =========================================
    // Save results
    // =========================================
    
    println("\nSaving results...");
    ensureDirectory("output");
    
    // Save energy vs J2/J1 data point
    std::ofstream outfile("output/j1j2_benchmark.dat", std::ios::app);
    outfile << std::fixed << std::setprecision(6);
    outfile << J2/J1 << "\t" 
            << result.energy_per_site << "\t"
            << result.entropy << "\t"
            << maxLinkDim(psi) << "\t"
            << Ly << "x" << Lx << "\n";
    outfile.close();
    
    println("Results appended to output/j1j2_benchmark.dat");
    println("\nDone!");
    
    return 0;
}
