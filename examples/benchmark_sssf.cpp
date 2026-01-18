//
// Benchmark: Standard DMRG with Static Spin Structure Factor
//
// Runs finite DMRG on a 1D Heisenberg chain and computes S(q)
//

#include "itensor/all.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace itensor;

int main(int argc, char* argv[]) {
    // =========================================
    // System parameters
    // =========================================
    
    int N = 40;           // Chain length
    double J = 1.0;       // Exchange coupling (AFM)
    int maxdim = 200;     // Max bond dimension
    int nq = 100;         // Number of q-points
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "--J" && i + 1 < argc) J = std::stod(argv[++i]);
        else if (arg == "--maxdim" && i + 1 < argc) maxdim = std::stoi(argv[++i]);
        else if (arg == "--nq" && i + 1 < argc) nq = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            println("Usage: benchmark_sssf [options]");
            println("");
            println("Options:");
            println("  --N n        Chain length (default: 40)");
            println("  --J val      Exchange coupling (default: 1.0)");
            println("  --maxdim n   Max bond dimension (default: 200)");
            println("  --nq n       Number of q-points (default: 100)");
            return 0;
        }
    }
    
    println("=============================================");
    println("DMRG Benchmark: Heisenberg Chain + SSSF");
    println("=============================================");
    println("");
    printfln("Chain length N = %d", N);
    printfln("Exchange J = %.4f", J);
    printfln("Max bond dimension = %d", maxdim);
    println("");
    
    // =========================================
    // Create sites and Hamiltonian
    // =========================================
    
    println("Setting up Hamiltonian...");
    
    // Spin-1/2 sites with Sz conservation
    auto sites = SpinHalf(N, {"ConserveQNs=", true});
    
    // Heisenberg Hamiltonian: H = J Σ S_i · S_{i+1}
    auto ampo = AutoMPO(sites);
    for (int i = 1; i < N; ++i) {
        ampo += J, "Sz", i, "Sz", i+1;
        ampo += 0.5*J, "S+", i, "S-", i+1;
        ampo += 0.5*J, "S-", i, "S+", i+1;
    }
    auto H = toMPO(ampo);
    
    printfln("MPO bond dimension: %d", maxLinkDim(H));
    println("");
    
    // =========================================
    // Initialize MPS (Neel state)
    // =========================================
    
    println("Initializing MPS (Neel state)...");
    
    auto init = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        init.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    auto psi = MPS(init);
    
    // =========================================
    // Run DMRG
    // =========================================
    
    println("");
    println("Running DMRG...");
    println("");
    
    auto sweeps = Sweeps(10);
    sweeps.maxdim() = 20, 50, 100, maxdim;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 4, 3, 2;
    sweeps.noise() = 1E-6, 1E-7, 1E-8, 0;
    
    println(sweeps);
    println("");
    
    auto [E0, psi_gs] = dmrg(H, psi, sweeps, {"Quiet", false});
    psi = psi_gs;
    
    println("");
    println("=============================================");
    println("DMRG Results");
    println("=============================================");
    printfln("Ground state energy E0 = %.12f", E0);
    printfln("Energy per site E0/N = %.12f", E0/N);
    printfln("Final bond dimension = %d", maxLinkDim(psi));
    println("");
    
    // Exact Bethe ansatz result for comparison
    double E_exact_per_site = -std::log(2.0) + 0.25;  // ≈ -0.4431...
    printfln("Bethe ansatz E/N (infinite) = %.12f", E_exact_per_site);
    printfln("Finite size error ~ %.2e", std::abs(E0/N - E_exact_per_site));
    println("");
    
    // =========================================
    // Compute Static Spin Structure Factor S(q)
    // =========================================
    
    println("=============================================");
    println("Computing Static Spin Structure Factor S(q)");
    println("=============================================");
    println("");
    
    // Compute spin-spin correlations from center
    int i0 = N / 2;  // Reference site
    
    println("Computing spin-spin correlations...");
    
    std::vector<Real> Czz(N, 0.0);  // <Sz_i0 Sz_j>
    std::vector<Real> Cpm(N, 0.0);  // 0.5*(<S+_i0 S-_j> + <S-_i0 S+_j>)
    
    psi.position(i0);
    
    for (int j = 1; j <= N; ++j) {
        int r = std::abs(j - i0);
        
        if (j == i0) {
            // On-site: <Sz^2> = 1/4 for spin-1/2
            Czz[r] = 0.25;
            Cpm[r] = 0.25;  // <Sx^2 + Sy^2> = 0.5 - <Sz^2> = 0.25... but actually <S+S->/2 + <S-S+>/2
            // For spin-1/2: <S+ S-> = <Sz> + 1/2, but on same site it's the identity/2
            // Actually for Sz=±1/2, <S+S-> = (1-2Sz)/2, <S-S+> = (1+2Sz)/2
            // So <S+S- + S-S+>/2 = 1/2
            Cpm[r] = 0.25;  // This is (Sx^2 + Sy^2) = S(S+1) - Sz^2 = 3/4 - 1/4 = 1/2, so per component 1/4
        } else {
            // Two-point correlations
            int site_i = i0;
            int site_j = j;
            if (site_i > site_j) std::swap(site_i, site_j);
            
            // <Sz_i Sz_j>
            psi.position(site_i);
            auto Cij = psi(site_i) * op(sites, "Sz", site_i);
            Cij *= dag(prime(psi(site_i), "Site", "Link"));
            
            for (int k = site_i + 1; k < site_j; ++k) {
                Cij *= psi(k);
                Cij *= dag(prime(psi(k), "Link"));
            }
            
            Cij *= psi(site_j) * op(sites, "Sz", site_j);
            Cij *= dag(prime(psi(site_j), "Site", "Link"));
            
            Czz[r] = elt(Cij);
            
            // <S+_i S-_j>
            psi.position(site_i);
            auto Cpm_ij = psi(site_i) * op(sites, "S+", site_i);
            Cpm_ij *= dag(prime(psi(site_i), "Site", "Link"));
            
            for (int k = site_i + 1; k < site_j; ++k) {
                Cpm_ij *= psi(k);
                Cpm_ij *= dag(prime(psi(k), "Link"));
            }
            
            Cpm_ij *= psi(site_j) * op(sites, "S-", site_j);
            Cpm_ij *= dag(prime(psi(site_j), "Site", "Link"));
            
            Real SpSm = elt(Cpm_ij);
            
            // <S-_i S+_j>
            psi.position(site_i);
            auto Cmp_ij = psi(site_i) * op(sites, "S-", site_i);
            Cmp_ij *= dag(prime(psi(site_i), "Site", "Link"));
            
            for (int k = site_i + 1; k < site_j; ++k) {
                Cmp_ij *= psi(k);
                Cmp_ij *= dag(prime(psi(k), "Link"));
            }
            
            Cmp_ij *= psi(site_j) * op(sites, "S+", site_j);
            Cmp_ij *= dag(prime(psi(site_j), "Site", "Link"));
            
            Real SmSp = elt(Cmp_ij);
            
            // Cpm = (Sx Sx + Sy Sy) = 0.5 * (S+ S- + S- S+)
            Cpm[r] = 0.5 * (SpSm + SmSp);
        }
        
        if (j % 10 == 0) {
            printfln("  Computed correlations up to site %d", j);
        }
    }
    
    // Print sample correlations
    println("");
    println("Sample correlations C(r) = <S_0 · S_r>:");
    println("  r\t\tC_zz(r)\t\tC_xy(r)\t\tC_total(r)");
    println("  --------------------------------------------------------");
    for (int r = 0; r <= std::min(10, N/2); ++r) {
        printfln("  %d\t\t%.6f\t%.6f\t%.6f", r, Czz[r], Cpm[r], Czz[r] + Cpm[r]);
    }
    
    // =========================================
    // Fourier transform to get S(q)
    // =========================================
    
    println("");
    println("Computing Fourier transform...");
    
    std::vector<Real> q_vals(nq);
    std::vector<Real> Sq_zz(nq, 0.0);
    std::vector<Real> Sq_xy(nq, 0.0);
    std::vector<Real> Sq_total(nq, 0.0);
    
    for (int iq = 0; iq < nq; ++iq) {
        double q = M_PI * iq / (nq - 1);  // q in [0, π]
        q_vals[iq] = q;
        
        // S(q) = Σ_r e^{iqr} C(r)  (summing both +r and -r)
        for (int r = 0; r < N/2; ++r) {
            double weight = (r == 0) ? 1.0 : 2.0;  // Count both +r and -r
            double phase = std::cos(q * r);
            
            Sq_zz[iq] += weight * Czz[r] * phase;
            Sq_xy[iq] += weight * Cpm[r] * phase;
        }
        
        Sq_total[iq] = Sq_zz[iq] + Sq_xy[iq];
    }
    
    // =========================================
    // Output results
    // =========================================
    
    println("");
    println("Static Spin Structure Factor S(q):");
    println("  q/π\t\tS_zz(q)\t\tS_xy(q)\t\tS(q)");
    println("  --------------------------------------------------------");
    
    // Print at selected q values
    std::vector<int> print_indices;
    for (int i = 0; i < nq; i += nq/10) print_indices.push_back(i);
    print_indices.push_back(nq - 1);
    
    for (int iq : print_indices) {
        printfln("  %.4f\t\t%.6f\t%.6f\t%.6f", 
                 q_vals[iq]/M_PI, Sq_zz[iq], Sq_xy[iq], Sq_total[iq]);
    }
    
    // Find peak
    int peak_idx = 0;
    Real peak_val = 0.0;
    for (int iq = 0; iq < nq; ++iq) {
        if (Sq_total[iq] > peak_val) {
            peak_val = Sq_total[iq];
            peak_idx = iq;
        }
    }
    
    println("");
    printfln("Peak: S(q=%.4fπ) = %.6f", q_vals[peak_idx]/M_PI, peak_val);
    println("Expected: Peak at q = π for AFM Heisenberg chain");
    
    // =========================================
    // Save to file
    // =========================================
    
    std::string filename = "Sq_heisenberg_N" + std::to_string(N) + ".dat";
    std::ofstream outfile(filename);
    outfile << "# Static Spin Structure Factor for Heisenberg chain\n";
    outfile << "# N = " << N << ", J = " << J << ", E0/N = " << E0/N << "\n";
    outfile << "# q/pi\tS_zz(q)\tS_xy(q)\tS(q)\n";
    
    for (int iq = 0; iq < nq; ++iq) {
        outfile << std::fixed << std::setprecision(8)
                << q_vals[iq]/M_PI << "\t"
                << Sq_zz[iq] << "\t"
                << Sq_xy[iq] << "\t"
                << Sq_total[iq] << "\n";
    }
    outfile.close();
    
    println("");
    printfln("Results saved to: %s", filename.c_str());
    
    println("");
    println("=============================================");
    println("Benchmark Complete!");
    println("=============================================");
    
    return 0;
}
