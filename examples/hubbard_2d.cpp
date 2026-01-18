//
// Example: 2D Hubbard Model on Cylinder
//
// Computes the ground state of the Hubbard model on a cylinder
// geometry using iDMRG. This is particularly useful for studying
// quasi-2D systems with cylinder width Ly.
//
// The cylinder is periodic in y and infinite in x.
//

#include "itensor/all.h"
#include "idmrg/idmrg.h"
#include "idmrg/lattice/square.h"
#include "idmrg/models/hubbard.h"
#include "idmrg/util/timer.h"
#include "idmrg/util/io.h"

using namespace itensor;
using namespace idmrg;

int main(int argc, char* argv[]) {
    printfln("\n=============================================");
    printfln("  2D Hubbard Model on Cylinder - iDMRG");
    printfln("=============================================\n");
    
    // Geometry: Ly x Lx cylinder
    // For iDMRG, we work with 2 columns as the unit cell
    int Ly = 4;   // Cylinder circumference (periodic)
    int Lx = 2;   // Columns per unit cell (half of full system)
    
    int N = 2 * Lx * Ly;  // Total sites (2 unit cells)
    
    printfln("Cylinder geometry: Ly = %d (periodic), Lx = %d unit cells", Ly, Lx);
    printfln("Total sites: %d", N);
    
    // Hubbard model parameters
    double t = 1.0;   // Hopping amplitude
    double U = 4.0;   // On-site interaction
    double mu = 0.0;  // Chemical potential (half-filling for mu = U/2)
    
    printfln("\nModel parameters:");
    printfln("  t = %.2f (hopping)", t);
    printfln("  U = %.2f (interaction)", U);
    printfln("  U/t = %.2f", U/t);
    printfln("  μ = %.2f", mu);
    
    // Create Hubbard sites (conserving N_up and N_dn separately)
    auto sites = Hubbard(N, {"ConserveNf", true, "ConserveSz", true});
    
    // Create cylinder lattice with snake ordering
    auto lattice = SquareLattice(2 * Lx, Ly, SiteOrdering::Snake);
    lattice.setYBC(BoundaryCondition::Periodic);  // Periodic in y
    lattice.setXBC(BoundaryCondition::Open);      // Open (infinite) in x
    lattice.setInfinite(true);
    
    println("\n" + lattice.info());
    
    // Build Hubbard Hamiltonian
    MPO H = HubbardMPO(sites, lattice, {
        "t", t,
        "U", U,
        "mu", mu,
        "Infinite", true
    });
    
    // Sweep schedule for 2D systems (more conservative)
    auto sweeps = Sweeps(30);
    sweeps.maxdim() = 100, 200, 400, 600, 800, 1000, 1200;
    sweeps.cutoff() = 1E-6, 1E-8, 1E-10, 1E-12;
    sweeps.niter() = 4, 3, 2;
    sweeps.noise() = 1E-6, 1E-7, 1E-8, 1E-9, 0;
    
    println("DMRG sweep schedule:");
    println(sweeps);
    
    // Initialize MPS - half-filling with alternating spin pattern
    // This is a good starting point for antiferromagnetic Hubbard
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        // Checkerboard pattern for antiferromagnet
        auto [x, y] = lattice.siteCoords(i);
        if ((x + y) % 2 == 0) {
            state.set(i, "Up");      // Spin up electron
        } else {
            state.set(i, "Dn");      // Spin down electron
        }
    }
    auto psi = MPS(state);
    
    // Verify particle number
    int Nup = 0, Ndn = 0;
    for (int i = 1; i <= N; ++i) {
        psi.position(i);
        auto phi = psi(i);
        Nup += static_cast<int>(elt(dag(prime(phi, "Site")) * op(sites, "Nup", i) * phi));
        Ndn += static_cast<int>(elt(dag(prime(phi, "Site")) * op(sites, "Ndn", i) * phi));
    }
    printfln("\nInitial state: N_up = %d, N_dn = %d, N_total = %d", Nup, Ndn, Nup + Ndn);
    printfln("Filling: %.2f\n", static_cast<double>(Nup + Ndn) / N);
    
    // Timer
    Timer timer;
    timer.start();
    
    // Run iDMRG
    printfln("Starting iDMRG calculation...\n");
    auto result = idmrg::idmrg(psi, H, sweeps, {"OutputLevel", 1});
    
    timer.stop();
    
    // Results
    println("\n=============================================");
    println("  Results");
    println("=============================================");
    printfln("  Ground state energy per site: %.14f", result.energy_per_site);
    printfln("  Entanglement entropy:         %.10f", result.entropy);
    printfln("  Total iDMRG steps:            %d", result.num_sweeps);
    printfln("  Converged:                    %s", result.converged ? "Yes" : "No");
    printfln("  Computation time:             %s", timer.elapsedStr().c_str());
    println("=============================================\n");
    
    // Measure local observables
    println("Local measurements:");
    println("Site   <n_up>      <n_dn>      <n_up n_dn>  <Sz>");
    println("-----------------------------------------------");
    
    int Nuc = Lx * Ly;  // Unit cell size
    
    for (int i = 1; i <= Nuc; ++i) {
        psi.position(i);
        
        auto ket = psi(i);
        auto bra = dag(prime(ket, "Site"));
        
        double nup = elt(bra * op(sites, "Nup", i) * ket);
        double ndn = elt(bra * op(sites, "Ndn", i) * ket);
        double nupdn = elt(bra * op(sites, "Nupdn", i) * ket);
        double sz = 0.5 * (nup - ndn);
        
        auto [x, y] = lattice.siteCoords(i);
        printfln("%3d (%d,%d)  %.6f    %.6f    %.6f     %.6f", 
                i, x, y, nup, ndn, nupdn, sz);
    }
    
    // Save results
    println("\nSaving results...");
    ensureDirectory("output");
    
    // Optional: save wavefunction
    // saveResult("output/hubbard_2d_result.idmrg", result, psi);
    
    println("\nDone!");
    
    return 0;
}
