//
// Example: Structure Factor Calculations
//
// This example demonstrates how to compute:
// 1. Static spin structure factor S(q)
// 2. Dynamical spin structure factor S(q,ω)
//
// For a Heisenberg spin chain as a simple test case.
//

#include "idmrg/all.h"

using namespace itensor;
using namespace idmrg;

int main(int argc, char* argv[]) {
    // =========================================
    // System parameters
    // =========================================
    
    int N = 20;           // Chain length
    double J = 1.0;       // Exchange coupling
    int nq = 50;          // Number of q-points for S(q)
    int n_omega = 100;    // Number of frequency points for S(q,ω)
    int n_chebyshev = 50; // Number of Chebyshev moments
    double eta = 0.1;     // Broadening for S(q,ω)
    bool compute_dynamic = false;  // Dynamical calculation is expensive
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "--J" && i + 1 < argc) J = std::stod(argv[++i]);
        else if (arg == "--nq" && i + 1 < argc) nq = std::stoi(argv[++i]);
        else if (arg == "--nomega" && i + 1 < argc) n_omega = std::stoi(argv[++i]);
        else if (arg == "--nchebyshev" && i + 1 < argc) n_chebyshev = std::stoi(argv[++i]);
        else if (arg == "--eta" && i + 1 < argc) eta = std::stod(argv[++i]);
        else if (arg == "--dynamic") compute_dynamic = true;
        else if (arg == "--help" || arg == "-h") {
            println("Usage: structure_factor [options]");
            println("");
            println("Options:");
            println("  --N n           Chain length (default: 20)");
            println("  --J val         Exchange coupling (default: 1.0)");
            println("  --nq n          Number of q-points (default: 50)");
            println("  --nomega n      Number of frequency points (default: 100)");
            println("  --nchebyshev n  Number of Chebyshev moments (default: 50)");
            println("  --eta val       Broadening for S(q,ω) (default: 0.1)");
            println("  --dynamic       Compute dynamical structure factor");
            return 0;
        }
    }
    
    // =========================================
    // Print simulation info
    // =========================================
    
    println("=============================================");
    println("Structure Factor Calculation");
    println("=============================================");
    println("");
    printfln("System: 1D Heisenberg chain, N = %d sites", N);
    printfln("Exchange coupling J = %.4f", J);
    println("");
    
    // =========================================
    // Create lattice and Hamiltonian
    // =========================================
    
    // Create spin-1/2 site set with Sz conservation
    auto sites = SpinHalf(N, {"ConserveQNs=", true});
    
    // Build Heisenberg Hamiltonian
    auto ampo = AutoMPO(sites);
    for (int i = 1; i < N; ++i) {
        ampo += J, "Sz", i, "Sz", i+1;
        ampo += 0.5 * J, "S+", i, "S-", i+1;
        ampo += 0.5 * J, "S-", i, "S+", i+1;
    }
    auto H = toMPO(ampo);
    
    // =========================================
    // Find ground state with DMRG
    // =========================================
    
    println("Running DMRG to find ground state...");
    println("");
    
    // Initial state: Neel state
    auto init_state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        init_state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    auto psi = MPS(init_state);
    
    // DMRG sweeps
    auto sweeps = Sweeps(10);
    sweeps.maxdim() = 50, 100, 200, 300, 400;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 4, 3, 2;
    
    auto [E0, psi_gs] = dmrg(H, psi, sweeps, {"Quiet", true});
    psi = psi_gs;
    
    printfln("Ground state energy: E_0 = %.10f", E0);
    printfln("Energy per site: E_0/N = %.10f", E0 / N);
    printfln("MPS bond dimension: %d", maxLinkDim(psi));
    println("");
    
    // =========================================
    // Compute Static Structure Factor S(q)
    // =========================================
    
    println("=============================================");
    println("Static Structure Factor S(q)");
    println("=============================================");
    println("");
    
    auto Sq_result = staticStructureFactor1D(psi, sites, nq, {"Components", true});
    
    println("");
    println("Results for S(q):");
    println("");
    println("q/π\t\tS(q)\t\tS_zz(q)\t\tS_xy(q)");
    println("------------------------------------------------------------");
    
    // Print at key q-values
    std::vector<int> key_indices;
    for (int iq = 0; iq < nq; iq += nq/10) {
        key_indices.push_back(iq);
    }
    key_indices.push_back(nq - 1);
    
    for (int iq : key_indices) {
        double q = Sq_result.qpoints[iq][0];
        printfln("%.4f\t\t%.6f\t%.6f\t%.6f", 
                 q / M_PI,
                 Sq_result.Sq[iq].real(),
                 Sq_result.Sq_components[1][iq].real(),  // Szz
                 Sq_result.Sq_components[0][iq].real()); // Sxx+Syy
    }
    
    // Find the peak (should be at q = π for AFM Heisenberg)
    int peak_idx = 0;
    double peak_val = 0.0;
    for (int iq = 0; iq < nq; ++iq) {
        if (Sq_result.Sq[iq].real() > peak_val) {
            peak_val = Sq_result.Sq[iq].real();
            peak_idx = iq;
        }
    }
    
    println("");
    printfln("Peak location: q = %.4f π", Sq_result.qpoints[peak_idx][0] / M_PI);
    printfln("Peak value: S(q_peak) = %.6f", peak_val);
    println("");
    
    // Save to file
    printStructureFactor(Sq_result, "Sq_heisenberg.dat");
    println("Static structure factor saved to 'Sq_heisenberg.dat'");
    
    // =========================================
    // Compute Dynamical Structure Factor S(q,ω)
    // =========================================
    
    if (compute_dynamic) {
        println("");
        println("=============================================");
        println("Dynamical Structure Factor S(q,ω)");
        println("=============================================");
        println("");
        
        // Create a simple 1D lattice for coordinate info
        // For 1D chain, coordinates are just site indices
        auto lattice = Chain(N);
        
        // Select q-points of interest
        std::vector<std::vector<double>> qpoints_dyn;
        qpoints_dyn.push_back({0.0});           // q = 0
        qpoints_dyn.push_back({M_PI / 2});      // q = π/2
        qpoints_dyn.push_back({M_PI});          // q = π (AFM wavevector)
        qpoints_dyn.push_back({3 * M_PI / 2});  // q = 3π/2
        
        // Energy bounds (estimate from J)
        double E_min = E0 - 0.5;
        double E_max = E0 + 4.0 * std::abs(J);  // Upper bound for spinon continuum
        
        printfln("Computing S(q,ω) using Chebyshev expansion");
        printfln("  Energy range: [%.4f, %.4f]", E_min, E_max);
        printfln("  Broadening: η = %.4f", eta);
        printfln("  Chebyshev moments: %d", n_chebyshev);
        println("");
        
        auto Sqw_result = dynamicalStructureFactorChebyshev(
            psi, H, sites, lattice,
            E_min, E_max,
            qpoints_dyn,
            n_omega,
            n_chebyshev,
            {"MaxDim", 200, "Cutoff", 1E-10}
        );
        
        println("");
        println("Sample S(q,ω) values at q = π:");
        println("ω\t\tS(π,ω)");
        println("--------------------");
        
        // q = π is index 2
        int iq_pi = 2;
        for (int iw = 0; iw < n_omega; iw += n_omega / 10) {
            printfln("%.4f\t\t%.6f", 
                     Sqw_result.omega[iw],
                     Sqw_result.Sqw[iq_pi][iw].real());
        }
        
        // Save to file
        printDynamicalStructureFactor(Sqw_result, "Sqw_heisenberg.dat");
        println("");
        println("Dynamical structure factor saved to 'Sqw_heisenberg.dat'");
    } else {
        println("");
        println("Dynamical structure factor calculation skipped.");
        println("Use --dynamic flag to compute S(q,ω).");
    }
    
    // =========================================
    // Compare with exact results for Heisenberg chain
    // =========================================
    
    println("");
    println("=============================================");
    println("Comparison with Bethe Ansatz (for reference)");
    println("=============================================");
    println("");
    println("For the infinite Heisenberg chain:");
    printfln("  Exact E_0/N = -ln(2) + 1/4 ≈ %.10f", -std::log(2.0) + 0.25);
    printfln("  Our E_0/N = %.10f (finite size: N = %d)", E0 / N, N);
    println("");
    println("S(q) should peak at q = π with logarithmic divergence");
    println("in the thermodynamic limit (Bethe Ansatz result).");
    println("");
    
    println("Done!");
    
    return 0;
}
