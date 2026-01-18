//
// Infinite DMRG (iDMRG) Algorithm
// High-performance implementation using ITensor v3
//
// Based on:
//   - White, Huse, PRB 48, 3844 (1993)
//   - McCulloch, arXiv:0804.2509 (2008)
//

#ifndef IDMRG_IDMRG_H
#define IDMRG_IDMRG_H

#include "itensor/all.h"
#include "idmrg/config.h"
#include "idmrg/observer.h"
#include <iostream>
#include <cmath>
#include <fstream>

namespace idmrg {

using namespace itensor;

//
// Return value structure for iDMRG
//
struct iDMRGResult {
    Real energy;        // Energy (total or per site depending on context)
    Real energy_per_site; // Energy per site
    ITensor HL;         // Left environment tensor
    ITensor HR;         // Right environment tensor
    ITensor IL;         // Left identity tensor for energy subtraction
    ITensor V;          // Singular value matrix (inverse, for gauge fixing)
    int num_sweeps;     // Number of iDMRG steps performed
    Real truncation_error;  // Final truncation error
    Real entropy;       // Central entanglement entropy
    bool converged;     // Whether calculation converged
    
    iDMRGResult() 
        : energy(0), energy_per_site(0), num_sweeps(0), 
          truncation_error(0), entropy(0), converged(false) {}
};

// I/O for checkpointing
void inline read(std::istream& s, iDMRGResult& r) {
    itensor::read(s, r.energy);
    itensor::read(s, r.energy_per_site);
    itensor::read(s, r.HL);
    itensor::read(s, r.HR);
    itensor::read(s, r.IL);
    itensor::read(s, r.V);
    itensor::read(s, r.num_sweeps);
    // Note: truncation_error, entropy, converged are not saved
}

void inline write(std::ostream& s, iDMRGResult const& r) {
    itensor::write(s, r.energy);
    itensor::write(s, r.energy_per_site);
    itensor::write(s, r.HL);
    itensor::write(s, r.HR);
    itensor::write(s, r.IL);
    itensor::write(s, r.V);
    itensor::write(s, r.num_sweeps);
}

//
// Helper: Pseudo-inverse functor for singular values
//
namespace detail {

struct PseudoInvert {
    Real cutoff;
    PseudoInvert(Real cut = 1E-8) : cutoff(cut) {}
    
    Real operator()(Real x) const {
        return (x > cutoff) ? 1.0 / x : 0.0;
    }
};

} // namespace detail

//
// Helper: Swap unit cells in an MPS or MPO
// Given A1 A2 A3 | A4 A5 A6 -> A4 A5 A6 | A1 A2 A3
//
template<class MPSType>
void swapUnitCells(MPSType& psi) {
    int N = length(psi);
    int Nuc = N / 2;
    for (int n = 1; n <= Nuc; ++n) {
        psi.ref(n).swap(psi.ref(Nuc + n));
    }
}

//
// Main iDMRG function with observer
//
iDMRGResult inline
idmrg(MPS& psi,
      MPO H,  // Copy to allow in-place modification
      iDMRGResult last_result,
      Sweeps const& sweeps,
      DMRGObserver& obs,
      Args args) {
    
    // Parse arguments
    auto olevel = args.getInt("OutputLevel", 1);
    auto quiet = args.getBool("Quiet", olevel == 0);
    auto nucsweeps = args.getInt("NUCSweeps", 1);
    auto nucs_decr = args.getInt("NUCSweepsDecrement", 0);
    auto do_randomize = args.getBool("Randomize", false);
    auto show_overlap = args.getBool("ShowOverlap", false);
    auto inverse_cut = args.getReal("InverseCut", 1E-8);
    auto actual_nucsweeps = nucsweeps;
    auto combine_mpo = args.getBool("CombineMPO", true);
    
    int N0 = length(psi);  // Total sites in finite system (2 unit cells)
    int Nuc = N0 / 2;      // Number of sites per unit cell
    int N = N0;            // Running count of effective system size
    
    if (N0 == 2) combine_mpo = false;
    
    Real energy = NAN;
    
    // Initialize from last result if available
    auto lastV = last_result.V;
    ITensor D;  // Current singular value matrix
    
    // Handle existing center matrix in psi(0)
    if (psi(0)) {
        lastV = dag(psi(0));
        lastV /= norm(lastV);
        lastV.apply(detail::PseudoInvert(0));
    }
    
    // Environment tensors
    ITensor HL(last_result.HL);
    ITensor HR(last_result.HR);
    ITensor IL = last_result.IL;
    
    // If no previous result, get edge tensors from MPO
    if (!HL) HL = H(0);
    if (!HR) HR = H(N0 + 1);
    
    if (!quiet) {
        println("\n========================================");
        println("       Infinite DMRG Calculation        ");
        println("========================================");
        printfln("  Unit cell size: %d sites", Nuc);
        printfln("  Max sweeps: %d", sweeps.nsweep());
        println("========================================\n");
    }
    
    int sw = 1;
    
    //
    // Step 1: Initial optimization with two unit cells
    //
    {
        if (!quiet) {
            printfln("\niDMRG Step = %d, N = %d sites", sw, N);
        }
        
        // Set up sweeps for initial step
        auto ucsweeps = Sweeps(actual_nucsweeps);
        ucsweeps.mindim() = sweeps.mindim(sw);
        ucsweeps.maxdim() = sweeps.maxdim(sw);
        ucsweeps.cutoff() = sweeps.cutoff(sw);
        ucsweeps.noise() = sweeps.noise(sw);
        ucsweeps.niter() = sweeps.niter(sw);
        
        if (olevel >= 2) print(ucsweeps);
        
        auto extra_args = Args("Quiet", olevel < 2,
                               "iDMRG_Step", sw,
                               "NSweep", ucsweeps.nsweep());
        
        // Run finite DMRG on initial system
        energy = dmrg(psi, H, HL, HR, ucsweeps, obs, args + extra_args);
        
        if (do_randomize) {
            println("Randomizing psi");
            for (int j = 1; j <= length(psi); ++j) {
                psi.ref(j).randomize();
            }
            psi.normalize();
        }
        
        printfln("\n    Energy per site = %.14f\n", energy / N0);
        
        // Position at center and get center matrix
        psi.position(Nuc);
        
        args.add("Sweep", sw);
        args.add("AtBond", Nuc);
        args.add("Energy", energy);
        obs.measure(args + Args("AtCenter", true, "NoMeasure", true));
        
        // SVD to get center matrix D
        svd(psi(Nuc) * psi(Nuc + 1), psi.ref(Nuc), D, psi.ref(Nuc + 1));
        D /= norm(D);
        
        // Build new environment tensors by contracting with optimized MPS
        for (int j = 1; j <= Nuc; ++j) {
            HL *= psi(j);
            HL *= H(j);
            HL *= dag(prime(psi(j)));
            
            IL *= psi(j);
            IL *= H(j);
            IL *= dag(prime(psi(j)));
            
            HR *= psi(N0 - j + 1);
            HR *= H(N0 - j + 1);
            HR *= dag(prime(psi(N0 - j + 1)));
        }
        
        // Swap MPO unit cells
        swapUnitCells(H);
        
        // Subtract energy to make effective Hamiltonian for next step
        HL += -energy * IL;
        
        // Prepare MPS for next step
        swapUnitCells(psi);
        if (lastV) psi.ref(Nuc + 1) *= lastV;
        psi.ref(1) *= D;
        psi.ref(N0) *= D;
        psi.position(1);
        
        ++sw;
    }
    
    Spectrum spec;
    
    //
    // Main iDMRG loop
    //
    for (; sw <= sweeps.nsweep(); ++sw) {
        if (!quiet) printfln("\niDMRG Step = %d, N = %d sites", sw, N);
        
        auto initPsi = psi;
        
        // Create local MPO with environment
        auto PH = LocalMPO(H, HL, HR, args);
        
        // Debug: check initial energy
        if (olevel >= 1) {
            auto E = HL;
            for (auto n : range1(length(psi))) {
                E = E * dag(prime(psi(n))) * H(n) * psi(n);
            }
            E *= HR;
            auto ien = real(eltC(E));
            printfln("Initial energy = %.14f", ien);
        }
        
        // Set up sweeps for this step
        auto ucsweeps = Sweeps(actual_nucsweeps);
        ucsweeps.mindim() = sweeps.mindim(sw);
        ucsweeps.maxdim() = sweeps.maxdim(sw);
        ucsweeps.cutoff() = sweeps.cutoff(sw);
        ucsweeps.noise() = sweeps.noise(sw);
        ucsweeps.niter() = sweeps.niter(sw);
        args.add("MaxDim", sweeps.maxdim(sw));
        
        if (olevel >= 2) print(ucsweeps);
        
        if (actual_nucsweeps > 1) actual_nucsweeps -= nucs_decr;
        
        N += N0;
        
        // Run DMRG on unit cell
        auto extra_args = Args("Quiet", olevel < 2, 
                               "NoMeasure", sw % 2 == 0,
                               "iDMRG_Step", sw, 
                               "NSweep", ucsweeps.nsweep());
        energy = DMRGWorker(psi, PH, ucsweeps, obs, args + extra_args);
        
        // Calculate overlap with initial state
        if (show_overlap || olevel >= 1) {
            auto O = dag(initPsi(1)) * psi(1);
            for (auto n : range1(2, length(psi))) {
                O = O * dag(initPsi(n)) * psi(n);
            }
            auto ovrlap = real(eltC(O));
            print("\n    Overlap of initial and final psi = ");
            printfln((std::fabs(ovrlap) > 1E-4 ? "%.10f" : "%.10E"), std::fabs(ovrlap));
            print("    1-Overlap = ");
            printfln((1 - std::fabs(ovrlap) > 1E-4 ? "%.10f" : "%.10E"), 1 - std::fabs(ovrlap));
        }
        
        printfln("    Energy per site = %.14f", energy / N0);
        
        // Save last center matrix (inverse)
        lastV = dag(D);
        lastV /= norm(lastV);
        lastV.apply(detail::PseudoInvert(inverse_cut));
        
        // Calculate new center matrix
        psi.position(Nuc);
        
        args.add("Sweep", sw);
        args.add("AtBond", Nuc);
        args.add("Energy", energy);
        obs.measure(args + Args("AtCenter", true, "NoMeasure", true));
        
        D = ITensor();
        svd(psi(Nuc) * psi(Nuc + 1), psi.ref(Nuc), D, psi.ref(Nuc + 1), args);
        D /= norm(D);
        
        // Update environment tensors
        for (int j = 1; j <= Nuc; ++j) {
            HL *= psi(j);
            HL *= H(j);
            HL *= dag(prime(psi(j)));
            
            IL *= psi(j);
            IL *= H(j);
            IL *= dag(prime(psi(j)));
            
            HR *= psi(N0 - j + 1);
            HR *= H(N0 - j + 1);
            HR *= dag(prime(psi(N0 - j + 1)));
        }
        swapUnitCells(H);
        
        HL += -energy * IL;
        
        // Prepare MPS for next step
        swapUnitCells(psi);
        
        psi.ref(N0) *= D;
        
        // Check convergence
        if ((obs.checkDone(args) && sw % 2 == 0) || sw == sweeps.nsweep()) {
            // Convert A tensors to B tensors for proper Lambda-Gamma form
            for (int b = N0 - 1; b >= Nuc + 1; --b) {
                ITensor d;
                svd(psi(b) * psi(b + 1), psi.ref(b), d, psi.ref(b + 1));
                psi.ref(b) *= d;
            }
            psi.ref(Nuc + 1) *= lastV;
            
            psi.ref(0) = D;  // Store center matrix
            
            break;
        }
        
        // Optional: write checkpoint
        if (fileExists("WRITE_WF") && sw % 2 == 0) {
            println("File WRITE_WF found: writing wavefunction after step ", sw);
            system("rm -f WRITE_WF");
            
            auto wpsi = psi;
            for (int b = N0 - 1; b >= Nuc + 1; --b) {
                ITensor d;
                svd(wpsi(b) * wpsi(b + 1), wpsi.ref(b), d, wpsi.ref(b + 1));
                wpsi.ref(b) *= d;
            }
            wpsi.ref(Nuc + 1) *= lastV;
            wpsi.ref(0) = D;
            writeToFile(tinyformat::format("psi_%d", sw), wpsi);
        }
        
        psi.ref(Nuc + 1) *= lastV;
        psi.ref(1) *= D;
        
        psi.orthogonalize();
        psi.normalize();
        
    } // Main iDMRG loop
    
    // Build result structure
    auto result = iDMRGResult();
    result.energy = energy;
    result.energy_per_site = energy / N0;
    result.HL = HL;
    result.HR = HR;
    result.IL = IL;
    result.V = lastV;
    result.num_sweeps = sw;
    result.converged = (sw < sweeps.nsweep());
    
    // Calculate final entanglement entropy
    if (D) {
        result.entropy = 0.0;
        for (int n = 1; n <= dim(D.inds()[0]); ++n) {
            double p = sqr(elt(D, n, n));
            if (p > 1E-15) {
                result.entropy -= p * std::log(p);
            }
        }
    }
    
    if (!quiet) {
        println("\n========================================");
        println("         iDMRG Results                  ");
        println("========================================");
        printfln("  Final energy/site: %.14f", result.energy_per_site);
        printfln("  Entanglement entropy: %.10f", result.entropy);
        printfln("  Sweeps performed: %d", result.num_sweeps);
        printfln("  Converged: %s", result.converged ? "Yes" : "No");
        println("========================================\n");
    }
    
    return result;
}

//
// Convenience overloads
//

// With observer, starting fresh
iDMRGResult inline
idmrg(MPS& psi,
      MPO const& H,
      Sweeps const& sweeps,
      DMRGObserver& obs,
      Args const& args = Args::global()) {
    // Initialize IL from MPO edge vector
    auto lval = iDMRGResult();
    lval.IL = ITensor(dag(H(length(H) + 1)));
    return idmrg(psi, H, lval, sweeps, obs, args);
}

// Without observer
iDMRGResult inline
idmrg(MPS& psi,
      MPO const& H,
      Sweeps const& sweeps,
      Args const& args = Args::global()) {
    auto obs = DMRGObserver(psi);
    return idmrg(psi, H, sweeps, obs, args);
}

// Restart from previous result (without observer)
iDMRGResult inline
idmrg(MPS& psi,
      MPO const& H,
      iDMRGResult const& last_result,
      Sweeps const& sweeps,
      Args const& args = Args::global()) {
    auto obs = DMRGObserver(psi);
    return idmrg(psi, H, last_result, sweeps, obs, args);
}

//
// Utility: Measure infinite MPS correlation function
// Exploits translational invariance of iDMRG ground state
//
template<typename SiteSetType>
std::vector<double>
measureInfiniteCorrelation(MPS const& psi,
                           SiteSetType const& sites,
                           std::string const& op1,
                           std::string const& op2,
                           int max_distance) {
    int N = length(psi);
    int Nuc = N / 2;
    
    std::vector<double> corr(max_distance);
    
    // Multiply in center matrix psi(0)
    auto wf1 = psi(0) * psi(1);
    auto oi = uniqueIndex(psi(0), psi(1), "Link");
    
    // Build left correlation tensor
    auto lcorr = prime(wf1, oi) * op(sites, op1, 1) * dag(prime(wf1));
    
    for (int j = 2; j <= max_distance + 1; ++j) {
        int n = ((j - 2) % Nuc) + 1;  // Map to unit cell site
        
        // Get unique index for contraction
        auto ui = uniqueIndex(psi(n), lcorr, "Link");
        
        // Compute correlation
        corr[j - 2] = elt(dag(prime(psi(n))) * lcorr * 
                         prime(psi(n), ui) * op(sites, op2, n));
        
        // Update correlation tensor
        lcorr *= psi(n);
        lcorr *= dag(prime(psi(n), "Link"));
    }
    
    return corr;
}

} // namespace idmrg

#endif // IDMRG_IDMRG_H
