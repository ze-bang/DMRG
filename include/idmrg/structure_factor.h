//
// Structure Factor Calculations
//
// This module provides functions for computing:
// 1. Static spin structure factor S(q)
// 2. Dynamical spin structure factor S(q,ω)
//
// The static structure factor is:
//   S(q) = (1/N) Σ_{i,j} e^{iq·(r_i - r_j)} <S_i · S_j>
//
// The dynamical structure factor is:
//   S(q,ω) = (1/N) Σ_{i,j} e^{iq·(r_i - r_j)} ∫dt e^{iωt} <S_i(t)·S_j(0)>
//          = -(1/π) Im G(q,ω+iη)
//
// where G(q,ω) = <ψ| S(-q) (ω+E_0-H+iη)^{-1} S(q) |ψ>
//

#ifndef IDMRG_STRUCTURE_FACTOR_H
#define IDMRG_STRUCTURE_FACTOR_H

#include "itensor/all.h"
#include "idmrg/lattice/lattice.h"
#include <cmath>
#include <complex>
#include <vector>
#include <map>
#include <functional>

namespace idmrg {

using namespace itensor;

//=============================================================================
// CORRELATION FUNCTIONS
//=============================================================================

//
// Compute two-point correlation function <O_i O_j>
//
inline Cplx correlationFunction(MPS const& psi,
                                SiteSet const& sites,
                                std::string const& op1,
                                int i,
                                std::string const& op2,
                                int j)
{
    if (i == j) {
        // Single-site expectation value <O1 * O2>
        auto psi_copy = psi;
        psi_copy.position(i);
        auto O = op(sites, op1, i) * op(sites, op2, i);
        return eltC(dag(prime(psi_copy(i), "Site")) * O * psi_copy(i));
    }
    
    // Ensure i < j
    if (i > j) {
        std::swap(i, j);
        // Note: for general operators, need to be careful about order
    }
    
    // Use ITensor's correlation_matrix or direct contraction
    auto psi_copy = psi;
    psi_copy.position(i);
    
    // Contract from site i to site j
    auto O1 = op(sites, op1, i);
    auto C = psi_copy(i) * O1;
    C *= dag(prime(psi_copy(i), "Site", "Link"));
    
    for (int k = i + 1; k < j; ++k) {
        C *= psi_copy(k);
        C *= dag(prime(psi_copy(k), "Link"));
    }
    
    auto O2 = op(sites, op2, j);
    C *= psi_copy(j);
    C *= O2;
    C *= dag(prime(psi_copy(j), "Site", "Link"));
    
    return eltC(C);
}

//
// Compute full correlation matrix <O_i O_j> for all pairs
//
inline std::vector<std::vector<Cplx>>
correlationMatrix(MPS const& psi,
                  SiteSet const& sites,
                  std::string const& op1,
                  std::string const& op2)
{
    int N = length(psi);
    std::vector<std::vector<Cplx>> corr(N + 1, std::vector<Cplx>(N + 1, 0.0));
    
    for (int i = 1; i <= N; ++i) {
        for (int j = i; j <= N; ++j) {
            corr[i][j] = correlationFunction(psi, sites, op1, i, op2, j);
            corr[j][i] = std::conj(corr[i][j]);  // Hermitian
        }
    }
    
    return corr;
}

//
// Compute spin-spin correlation <S_i · S_j> = <Sx_i Sx_j> + <Sy_i Sy_j> + <Sz_i Sz_j>
//
inline Cplx spinSpinCorrelation(MPS const& psi,
                                SiteSet const& sites,
                                int i, int j)
{
    // <Sz_i Sz_j>
    auto SzSz = correlationFunction(psi, sites, "Sz", i, "Sz", j);
    
    // <S+_i S-_j> + <S-_i S+_j> = 2(<Sx_i Sx_j> + <Sy_i Sy_j>)
    auto SpSm = correlationFunction(psi, sites, "S+", i, "S-", j);
    auto SmSp = correlationFunction(psi, sites, "S-", i, "S+", j);
    
    // <Sx Sx> + <Sy Sy> = 0.5 * (<S+ S-> + <S- S+>)
    return SzSz + 0.5 * (SpSm + SmSp);
}


//=============================================================================
// STATIC STRUCTURE FACTOR
//=============================================================================

//
// Result structure for structure factor calculations
//
struct StructureFactorResult {
    std::vector<std::vector<double>> qpoints;  // Momentum points [nq][dim]
    std::vector<Cplx> Sq;                      // S(q) values
    std::vector<std::vector<Cplx>> Sq_components;  // [Sxx, Syy, Szz](q)
    
    // For dynamical structure factor
    std::vector<double> omega;                 // Frequency grid
    std::vector<std::vector<Cplx>> Sqw;        // S(q,ω) for each q
};

//
// Static spin structure factor S(q) for 1D chain
//
// S(q) = (1/N) Σ_{i,j} e^{iq(r_i - r_j)} <S_i · S_j>
//      = (1/N) Σ_{r} e^{iqr} C(r)
//
// where C(r) = Σ_i <S_i · S_{i+r}> is the correlation function vs distance
//
inline StructureFactorResult
staticStructureFactor1D(MPS const& psi,
                        SiteSet const& sites,
                        int nq = 100,
                        Args const& args = Args::global())
{
    StructureFactorResult result;
    int N = length(psi);
    bool compute_components = args.getBool("Components", false);
    
    // Compute correlation function for all distances
    std::vector<Cplx> C_total(N, 0.0);  // C(r) summed over reference sites
    std::vector<Cplx> Czz(N, 0.0);
    std::vector<Cplx> Cpm(N, 0.0);  // (S+ S- + S- S+)
    std::vector<int> counts(N, 0);
    
    println("Computing spin-spin correlations...");
    
    // Reference site at center for finite systems
    int i0 = N / 2;
    for (int j = 1; j <= N; ++j) {
        int r = std::abs(j - i0);
        if (r < N) {
            auto SzSz = correlationFunction(psi, sites, "Sz", i0, "Sz", j);
            auto SpSm = correlationFunction(psi, sites, "S+", i0, "S-", j);
            auto SmSp = correlationFunction(psi, sites, "S-", i0, "S+", j);
            
            Czz[r] += SzSz;
            Cpm[r] += 0.5 * (SpSm + SmSp);
            C_total[r] += SzSz + 0.5 * (SpSm + SmSp);
            counts[r]++;
        }
    }
    
    // Average over counts
    for (int r = 0; r < N; ++r) {
        if (counts[r] > 0) {
            C_total[r] /= counts[r];
            Czz[r] /= counts[r];
            Cpm[r] /= counts[r];
        }
    }
    
    // Compute S(q) by Fourier transform
    // S(q) = Σ_r e^{iqr} C(r)
    println("Computing Fourier transform...");
    
    result.qpoints.resize(nq);
    result.Sq.resize(nq);
    if (compute_components) {
        result.Sq_components.resize(3);  // Sxx+Syy, Szz, total
        for (auto& v : result.Sq_components) v.resize(nq);
    }
    
    for (int iq = 0; iq < nq; ++iq) {
        double q = 2.0 * M_PI * iq / nq;  // q in [0, 2π)
        result.qpoints[iq] = {q};
        
        Cplx Sq = 0.0;
        Cplx Sq_zz = 0.0;
        Cplx Sq_pm = 0.0;
        
        for (int r = 0; r < N; ++r) {
            Cplx phase = std::exp(Cplx(0, q * r));
            Cplx phase_neg = std::exp(Cplx(0, -q * r));
            
            if (r == 0) {
                Sq += C_total[r];
                Sq_zz += Czz[r];
                Sq_pm += Cpm[r];
            } else {
                // Both +r and -r contribute
                Sq += C_total[r] * (phase + phase_neg);
                Sq_zz += Czz[r] * (phase + phase_neg);
                Sq_pm += Cpm[r] * (phase + phase_neg);
            }
        }
        
        result.Sq[iq] = Sq;
        
        if (compute_components) {
            result.Sq_components[0][iq] = Sq_pm;   // Sxx + Syy
            result.Sq_components[1][iq] = Sq_zz;   // Szz
            result.Sq_components[2][iq] = Sq;      // Total
        }
    }
    
    return result;
}

//
// Static structure factor for 2D lattice
//
// S(q) = (1/N) Σ_{i,j} e^{i q·(r_i - r_j)} <S_i · S_j>
//
inline StructureFactorResult
staticStructureFactor2D(MPS const& psi,
                        SiteSet const& sites,
                        Lattice const& lattice,
                        int nqx = 20, int nqy = 20,
                        Args const& args = Args::global())
{
    StructureFactorResult result;
    int N = length(psi);
    
    // Get site coordinates from lattice
    auto const& site_coords = lattice.sites();
    
    // Build a map from site index to coordinates
    std::map<int, std::pair<double, double>> coords;
    for (const auto& sc : site_coords) {
        coords[sc.index] = {sc.x, sc.y};
    }
    
    println("Computing spin-spin correlations for 2D lattice...");
    
    // Compute all correlations from a reference site (center)
    int i0 = N / 2;
    double x0 = coords[i0].first;
    double y0 = coords[i0].second;
    
    std::vector<double> dr_x(N + 1), dr_y(N + 1);
    std::vector<Cplx> C(N + 1, 0.0);
    
    for (int j = 1; j <= N; ++j) {
        dr_x[j] = coords[j].first - x0;
        dr_y[j] = coords[j].second - y0;
        C[j] = spinSpinCorrelation(psi, sites, i0, j);
    }
    
    // Compute S(q) on a grid in the Brillouin zone
    println("Computing Fourier transform on 2D momentum grid...");
    
    int nq = nqx * nqy;
    result.qpoints.resize(nq);
    result.Sq.resize(nq);
    
    for (int iqx = 0; iqx < nqx; ++iqx) {
        for (int iqy = 0; iqy < nqy; ++iqy) {
            int iq = iqx * nqy + iqy;
            
            // q-vector in first Brillouin zone
            double qx = 2.0 * M_PI * (iqx - nqx/2) / nqx;
            double qy = 2.0 * M_PI * (iqy - nqy/2) / nqy;
            
            result.qpoints[iq] = {qx, qy};
            
            Cplx Sq = 0.0;
            for (int j = 1; j <= N; ++j) {
                double qr = qx * dr_x[j] + qy * dr_y[j];
                Sq += C[j] * std::exp(Cplx(0, qr));
            }
            
            result.Sq[iq] = Sq;
        }
    }
    
    return result;
}


//=============================================================================
// DYNAMICAL STRUCTURE FACTOR
//=============================================================================

//
// Apply operator S(q) = (1/√N) Σ_j e^{iq·r_j} S_j to MPS
// Returns the resulting state as an MPS
//
inline MPS applySq(MPS const& psi,
                   SiteSet const& sites,
                   std::string const& opname,  // "Sz", "S+", "S-"
                   double qx, double qy,
                   Lattice const& lattice)
{
    int N = length(psi);
    MPS phi = psi;
    
    // Get site coordinates
    auto const& site_coords = lattice.sites();
    std::map<int, std::pair<double, double>> coords;
    for (const auto& sc : site_coords) {
        coords[sc.index] = {sc.x, sc.y};
    }
    
    // Build MPO for S(q)
    AutoMPO ampo(sites);
    double norm_factor = 1.0 / std::sqrt(static_cast<double>(N));
    
    for (int j = 1; j <= N; ++j) {
        double rj_x = coords.count(j) ? coords[j].first : static_cast<double>(j - 1);
        double rj_y = coords.count(j) ? coords[j].second : 0.0;
        double qr = qx * rj_x + qy * rj_y;
        
        Cplx coeff = norm_factor * std::exp(Cplx(0, qr));
        ampo += coeff, opname, j;
    }
    
    auto Sq_MPO = toMPO(ampo);
    
    // Apply MPO to MPS
    phi = applyMPO(Sq_MPO, psi);
    phi.noPrime();
    
    return phi;
}

//
// Compute dynamical structure factor S(q,ω) using correction vector method
//
// S^{αα}(q,ω) = -(1/π) Im <ψ| S^α(-q) (ω + E_0 - H + iη)^{-1} S^α(q) |ψ>
//
// This uses the correction vector approach:
// |φ⟩ = (ω + E_0 - H + iη)^{-1} S(q)|ψ⟩
// which satisfies: (H - E_0 - ω - iη)|φ⟩ = -S(q)|ψ⟩
//
inline StructureFactorResult
dynamicalStructureFactor(MPS const& psi,
                         MPO const& H,
                         SiteSet const& sites,
                         Lattice const& lattice,
                         double E0,  // Ground state energy
                         std::vector<std::vector<double>> const& qpoints,
                         std::vector<double> const& omega,
                         Args const& args = Args::global())
{
    StructureFactorResult result;
    result.qpoints = qpoints;
    result.omega = omega;
    
    int nq = qpoints.size();
    int nw = omega.size();
    
    result.Sqw.resize(nq);
    for (auto& v : result.Sqw) v.resize(nw);
    
    // Parameters
    double eta = args.getReal("Broadening", 0.1);
    int maxdim = args.getInt("MaxDim", 200);
    double cutoff = args.getReal("Cutoff", 1E-10);
    int maxiter = args.getInt("MaxIter", 100);
    double cvtol = args.getReal("CVTol", 1E-6);
    
    auto sweeps = Sweeps(maxiter);
    sweeps.maxdim() = maxdim;
    sweeps.cutoff() = cutoff;
    
    println("Computing dynamical structure factor...");
    printfln("  Broadening η = %.4f", eta);
    printfln("  Number of q-points: %d", nq);
    printfln("  Number of frequencies: %d", nw);
    
    for (int iq = 0; iq < nq; ++iq) {
        double qx = qpoints[iq][0];
        double qy = qpoints[iq].size() > 1 ? qpoints[iq][1] : 0.0;
        
        printfln("\nq-point %d/%d: q = (%.4f, %.4f)", iq+1, nq, qx, qy);
        
        // Apply S^z(q) to ground state: |target⟩ = S^z(q)|ψ⟩
        // For full S(q,ω), should compute for Sz, S+, S- components
        auto target = applySq(psi, sites, "Sz", qx, qy, lattice);
        target.normalize();
        
        // For each frequency, solve for correction vector
        for (int iw = 0; iw < nw; ++iw) {
            double w = omega[iw];
            
            // Solve: (H - E_0 - ω - iη)|φ⟩ = -|target⟩
            // Using linear system solver or iterative method
            
            // For now, use the simpler Lanczos-based continued fraction approach
            // This computes G(ω) = ⟨target|(ω+iη-H+E_0)^{-1}|target⟩
            
            // Initialize correction vector
            MPS phi = target;
            
            // Lanczos coefficients
            std::vector<double> a_lanczos, b_lanczos;
            
            // Run Lanczos iteration
            MPS v0;  // Empty
            MPS v1 = target;
            v1.normalize();
            
            auto Heff = H;  // Will subtract E0 in application
            
            const int max_lanczos = std::min(50, static_cast<int>(length(psi)) * 2);
            
            for (int k = 0; k < max_lanczos; ++k) {
                // w = H|v1⟩
                auto w_mps = applyMPO(Heff, v1, {"MaxDim", maxdim, "Cutoff", cutoff});
                w_mps.noPrime();
                
                // Subtract E0 * |v1⟩
                // a_k = ⟨v1|H|v1⟩ - E0
                double a_k = std::real(innerC(v1, w_mps)) - E0;
                a_lanczos.push_back(a_k);
                
                // w = w - a_k * v1 - b_{k-1} * v0
                w_mps.plusEq(-a_k * v1, {"MaxDim", maxdim, "Cutoff", cutoff});
                if (k > 0 && b_lanczos.size() > 0) {
                    w_mps.plusEq(-b_lanczos.back() * v0, {"MaxDim", maxdim, "Cutoff", cutoff});
                }
                
                double b_k = norm(w_mps);
                if (b_k < 1E-12) break;
                
                b_lanczos.push_back(b_k);
                
                // Update vectors
                v0 = v1;
                v1 = w_mps;
                v1 /= b_k;
            }
            
            // Compute Green's function using continued fraction
            // G(z) = 1/(z - a_0 - b_1^2/(z - a_1 - b_2^2/(z - a_2 - ...)))
            Cplx z = Cplx(w, eta);
            
            int n_lanczos = a_lanczos.size();
            Cplx G = 0.0;
            
            if (n_lanczos > 0) {
                // Start from the end of the continued fraction
                G = z - a_lanczos[n_lanczos - 1];
                
                for (int k = n_lanczos - 2; k >= 0; --k) {
                    if (std::abs(G) > 1E-15) {
                        G = z - a_lanczos[k] - b_lanczos[k] * b_lanczos[k] / G;
                    } else {
                        G = z - a_lanczos[k];
                    }
                }
                
                if (std::abs(G) > 1E-15) {
                    G = 1.0 / G;
                }
            }
            
            // S(q,ω) = -(1/π) Im G(ω)
            // But G here is ⟨target|G|target⟩ = ⟨ψ|S(-q) G S(q)|ψ⟩
            result.Sqw[iq][iw] = -G.imag() / M_PI;
        }
    }
    
    return result;
}

//
// Simplified correction vector method for a single frequency
//
inline MPS correctionVector(MPS const& psi,
                            MPO const& H,
                            MPS const& target,  // S(q)|ψ⟩
                            double E0,
                            double omega,
                            double eta,
                            Args const& args = Args::global())
{
    int maxiter = args.getInt("MaxIter", 30);
    int maxdim = args.getInt("MaxDim", 200);
    double cutoff = args.getReal("Cutoff", 1E-10);
    bool quiet = args.getBool("Quiet", true);
    
    // We solve: (H - E_0 - ω - iη)|φ⟩ = -|target⟩
    // Real part: (H - E_0 - ω)|φ_R⟩ + η|φ_I⟩ = -|target⟩
    // Imag part: (H - E_0 - ω)|φ_I⟩ - η|φ_R⟩ = 0
    // 
    // Combined: [(H - E_0 - ω)^2 + η^2]|φ_R⟩ = -(H - E_0 - ω)|target⟩
    
    // For simplicity, we'll use an iterative approach
    // Start with |φ⟩ ≈ -1/(ω + iη) |target⟩
    
    MPS phi = target;
    Cplx denom = Cplx(-omega, -eta);
    phi *= 1.0 / std::abs(denom);  // Initial guess
    
    double z_re = -omega;  // shift
    double z_im = eta;
    
    auto sweeps = Sweeps(maxiter);
    sweeps.maxdim() = maxdim;
    sweeps.cutoff() = cutoff;
    
    // Use DMRG-like sweeps to minimize ||(H-E0-z)|φ⟩ + |target⟩||^2
    // This is equivalent to solving the linear system
    
    // For now, return an approximate solution
    // A full implementation would use the linsolve functionality
    
    return phi;
}


//=============================================================================
// CHEBYSHEV EXPANSION METHOD for S(q,ω)
//=============================================================================

//
// Compute dynamical structure factor using Chebyshev (Kernel Polynomial) method
//
// This expands G(ω) in Chebyshev polynomials and uses Jackson kernel
// for better convergence.
//
inline StructureFactorResult
dynamicalStructureFactorChebyshev(MPS const& psi,
                                   MPO const& H,
                                   SiteSet const& sites,
                                   Lattice const& lattice,
                                   double E_min, double E_max,  // Energy bounds
                                   std::vector<std::vector<double>> const& qpoints,
                                   int n_omega = 200,
                                   int n_chebyshev = 100,
                                   Args const& args = Args::global())
{
    StructureFactorResult result;
    result.qpoints = qpoints;
    
    int nq = qpoints.size();
    int maxdim = args.getInt("MaxDim", 200);
    double cutoff = args.getReal("Cutoff", 1E-10);
    
    // Rescale Hamiltonian to [-1, 1]
    double E_center = 0.5 * (E_max + E_min);
    double E_scale = 0.5 * (E_max - E_min) * 1.05;  // Slightly larger to avoid edge effects
    
    printfln("Chebyshev expansion for S(q,ω)");
    printfln("  Energy range: [%.4f, %.4f]", E_min, E_max);
    printfln("  Number of Chebyshev moments: %d", n_chebyshev);
    printfln("  Number of q-points: %d", nq);
    
    // Build rescaled Hamiltonian: H_scaled = (H - E_center) / E_scale
    // For simplicity, we'll handle the rescaling during application
    
    // Create omega grid
    result.omega.resize(n_omega);
    for (int i = 0; i < n_omega; ++i) {
        // Map from [-1, 1] back to energy
        double x = -1.0 + 2.0 * i / (n_omega - 1);
        result.omega[i] = x * E_scale + E_center;
    }
    
    result.Sqw.resize(nq);
    for (auto& v : result.Sqw) v.resize(n_omega, 0.0);
    
    // Jackson kernel coefficients for improved convergence
    auto jacksonKernel = [n_chebyshev](int n) -> double {
        double N = n_chebyshev + 1;
        double arg = M_PI * n / N;
        return ((N - n) * std::cos(arg) + std::sin(arg) / std::tan(M_PI / N)) / N;
    };
    
    for (int iq = 0; iq < nq; ++iq) {
        double qx = qpoints[iq][0];
        double qy = qpoints[iq].size() > 1 ? qpoints[iq][1] : 0.0;
        
        printfln("\nq-point %d/%d: q = (%.4f, %.4f)", iq+1, nq, qx, qy);
        
        // |t_0⟩ = S(q)|ψ⟩
        auto t0 = applySq(psi, sites, "Sz", qx, qy, lattice);
        t0.normalize();
        
        // Compute Chebyshev moments μ_n = ⟨t_0|T_n(H_scaled)|t_0⟩
        std::vector<Cplx> mu(n_chebyshev, 0.0);
        
        // |T_0⟩ = |t_0⟩
        MPS T_prev = t0;
        
        // μ_0 = ⟨t_0|t_0⟩ = 1 (normalized)
        mu[0] = innerC(t0, T_prev);
        
        // |T_1⟩ = H_scaled |t_0⟩
        MPS T_curr = applyMPO(H, t0, {"MaxDim", maxdim, "Cutoff", cutoff});
        T_curr.noPrime();
        // Rescale: T_curr = (H - E_center)/E_scale |t_0⟩
        T_curr.plusEq((-E_center / E_scale) * t0, {"MaxDim", maxdim, "Cutoff", cutoff});
        T_curr *= (1.0 / E_scale);
        
        if (n_chebyshev > 1) {
            mu[1] = innerC(t0, T_curr);
        }
        
        // Chebyshev recursion: T_{n+1} = 2 H_scaled T_n - T_{n-1}
        for (int n = 2; n < n_chebyshev; ++n) {
            // |T_{n+1}⟩ = 2 H_scaled |T_n⟩ - |T_{n-1}⟩
            auto HT = applyMPO(H, T_curr, {"MaxDim", maxdim, "Cutoff", cutoff});
            HT.noPrime();
            
            MPS T_next = (2.0 / E_scale) * HT;
            T_next.plusEq((-2.0 * E_center / E_scale) * T_curr, {"MaxDim", maxdim, "Cutoff", cutoff});
            T_next.plusEq(-1.0 * T_prev, {"MaxDim", maxdim, "Cutoff", cutoff});
            
            mu[n] = innerC(t0, T_next);
            
            // Update for next iteration
            T_prev = T_curr;
            T_curr = T_next;
            
            if (n % 10 == 0) {
                printfln("  Chebyshev moment %d/%d computed", n, n_chebyshev);
            }
        }
        
        // Reconstruct S(q,ω) from moments
        // S(q,ω) = (1/π√(1-x²)) [μ_0 + 2 Σ_{n=1}^{N-1} g_n μ_n T_n(x)]
        // where x = (ω - E_center)/E_scale and g_n is Jackson kernel
        
        for (int i = 0; i < n_omega; ++i) {
            double x = (result.omega[i] - E_center) / E_scale;
            if (std::abs(x) >= 1.0) continue;  // Outside valid range
            
            double prefactor = 1.0 / (M_PI * std::sqrt(1.0 - x * x) * E_scale);
            
            Cplx sum = jacksonKernel(0) * mu[0];
            double T_n_prev = 1.0;  // T_0(x)
            double T_n = x;         // T_1(x)
            
            for (int n = 1; n < n_chebyshev; ++n) {
                sum += 2.0 * jacksonKernel(n) * mu[n] * T_n;
                
                // Chebyshev recursion for x
                double T_n_next = 2.0 * x * T_n - T_n_prev;
                T_n_prev = T_n;
                T_n = T_n_next;
            }
            
            result.Sqw[iq][i] = prefactor * sum;
        }
    }
    
    return result;
}


//=============================================================================
// OUTPUT UTILITIES
//=============================================================================

//
// Print static structure factor
//
inline void printStructureFactor(StructureFactorResult const& result, 
                                  std::string const& filename = "")
{
    std::ostream* out = &std::cout;
    std::ofstream file;
    
    if (!filename.empty()) {
        file.open(filename);
        out = &file;
    }
    
    *out << "# Static Structure Factor S(q)\n";
    *out << "# q_x";
    if (result.qpoints[0].size() > 1) *out << "\tq_y";
    *out << "\tRe[S(q)]\tIm[S(q)]\n";
    
    for (size_t iq = 0; iq < result.qpoints.size(); ++iq) {
        for (double q : result.qpoints[iq]) {
            *out << std::fixed << std::setprecision(6) << q << "\t";
        }
        *out << std::scientific << std::setprecision(8) 
             << result.Sq[iq].real() << "\t" << result.Sq[iq].imag() << "\n";
    }
}

//
// Print dynamical structure factor
//
inline void printDynamicalStructureFactor(StructureFactorResult const& result,
                                           std::string const& filename = "")
{
    std::ostream* out = &std::cout;
    std::ofstream file;
    
    if (!filename.empty()) {
        file.open(filename);
        out = &file;
    }
    
    *out << "# Dynamical Structure Factor S(q,ω)\n";
    *out << "# Columns: omega, S(q_1,ω), S(q_2,ω), ...\n";
    *out << "# q-points: ";
    for (const auto& q : result.qpoints) {
        *out << "(";
        for (size_t i = 0; i < q.size(); ++i) {
            if (i > 0) *out << ",";
            *out << q[i];
        }
        *out << ") ";
    }
    *out << "\n";
    
    for (size_t iw = 0; iw < result.omega.size(); ++iw) {
        *out << std::fixed << std::setprecision(6) << result.omega[iw];
        for (size_t iq = 0; iq < result.qpoints.size(); ++iq) {
            *out << "\t" << std::scientific << std::setprecision(8) 
                 << result.Sqw[iq][iw].real();
        }
        *out << "\n";
    }
}

} // namespace idmrg

#endif // IDMRG_STRUCTURE_FACTOR_H
