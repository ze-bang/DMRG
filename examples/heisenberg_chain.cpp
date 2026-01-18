//
// Example: 1D Heisenberg Chain with iDMRG
//
// Computes the ground state energy of the infinite S=1 Heisenberg chain
// and measures correlation functions.
//
// Reference: White, Huse, PRB 48, 3844 (1993)
// Exact energy per site (S=1): E/N ≈ -1.401484039...
//

#include "itensor/all.h"
#include "idmrg/idmrg.h"
#include "idmrg/lattice/chain.h"
#include "idmrg/models/heisenberg.h"
#include "idmrg/util/timer.h"
#include "idmrg/util/io.h"

using namespace itensor;
using namespace idmrg;

int main(int argc, char* argv[]) {
    printfln("\n=============================================");
    printfln("  1D Heisenberg Chain - Infinite DMRG");
    printfln("=============================================\n");
    
    // Unit cell size (total sites in finite system = 2 * Nuc)
    int Nuc = 2;      // Sites per unit cell
    int N = 2 * Nuc;  // Total sites for iDMRG (two unit cells)
    
    printfln("Unit cell size: %d sites", Nuc);
    printfln("Working with %d sites (2 unit cells)\n", N);
    
    // Create spin-1 sites without quantum number conservation
    // (QN conservation causes issues with iDMRG tensor mixing)
    auto sites = SpinOne(N, {"ConserveQNs=", false});
    
    // Build Heisenberg Hamiltonian for infinite system
    // H = J Σ (Sz_i Sz_j + 0.5 S+_i S-_j + 0.5 S-_i S+_j)
    auto lattice = Chain(N);
    lattice.setInfinite(true);
    
    MPO H = HeisenbergMPO(sites, lattice, {
        "J", 1.0,
        "Jz", 1.0,  // Isotropic
        "Infinite", true
    });
    
    // Configure DMRG sweeps
    // Start with small bond dimension and gradually increase
    auto sweeps = Sweeps(20);
    sweeps.maxdim() = 20, 50, 100, 200, 400, 600, 800;
    sweeps.cutoff() = 1E-10, Args("Repeat", 10), 1E-12, 1E-14;
    sweeps.niter() = 3, 2;
    sweeps.noise() = 1E-7, 1E-8, 1E-10, 0;
    
    println("DMRG sweep schedule:");
    println(sweeps);
    
    // Initialize MPS in Néel state (optimal for antiferromagnet)
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        if (i % 2 == 1) {
            state.set(i, "Up");  // Sz = +1
        } else {
            state.set(i, "Dn");  // Sz = -1
        }
    }
    auto psi = MPS(state);
    
    println("\nInitial state: Néel configuration\n");
    
    // Start timer
    Timer timer;
    timer.start();
    
    // Run iDMRG
    printfln("Starting iDMRG calculation...\n");
    auto result = idmrg::idmrg(psi, H, sweeps, {"OutputLevel", 1});
    
    timer.stop();
    
    // Print results
    println("\n=============================================");
    println("  Results");
    println("=============================================");
    printfln("  Ground state energy per site: %.14f", result.energy_per_site);
    printfln("  Entanglement entropy:         %.10f", result.entropy);
    printfln("  Total iDMRG steps:            %d", result.num_sweeps);
    printfln("  Converged:                    %s", result.converged ? "Yes" : "No");
    printfln("  Computation time:             %s", timer.elapsedStr().c_str());
    println("=============================================\n");
    
    // Compare with exact result for S=1 chain
    double exact_energy = -1.401484039;
    double error = std::abs(result.energy_per_site - exact_energy);
    printfln("Exact energy (S=1): %.10f", exact_energy);
    printfln("Error:              %.2e\n", error);
    
    // Measure correlation functions using the infinite MPS
    println("Measuring Sz-Sz correlation function...\n");
    
    int xrange = 50;  // Correlation distance
    
    printfln("j   <Sz_1 Sz_j>");
    printfln("--------------------");
    
    // The iDMRG MPS is stored with the center matrix at psi(0)
    auto wf1 = psi(0) * psi(1);
    auto oi = uniqueIndex(psi(0), psi(1), "Link");
    
    // Left part of correlation tensor
    auto lcorr = prime(wf1, oi) * op(sites, "Sz", 1) * dag(prime(wf1));
    
    std::vector<double> corr_values;
    
    for (int j = 2; j <= xrange; ++j) {
        int n = (j - 2) % Nuc + 1;  // Map to unit cell site
        
        auto ui = uniqueIndex(psi(n), lcorr, "Link");
        auto temp = prime(psi(n));
        auto val = elt(dag(temp) * lcorr * 
                      noPrime(prime(psi(n), ui)) * op(sites, "Sz", n));
        
        printfln("%3d   %.12f", j, val);
        corr_values.push_back(val);
        
        // Update correlation tensor
        lcorr *= psi(n);
        lcorr *= dag(prime(psi(n)));
    }
    
    // Save results
    println("\nSaving results...");
    ensureDirectory("output");
    writeCorrelationCSV("output/heisenberg_correlation.csv", corr_values, "Sz", "Sz");
    
    // Optional: Save wavefunction for later analysis
    // saveResult("output/heisenberg_result.idmrg", result, psi);
    
    println("\nDone!");
    
    return 0;
}
