//
// Heisenberg Model Hamiltonian
// Supports spin-1/2, spin-1, and arbitrary spin via SiteSet
//

#ifndef IDMRG_MODELS_HEISENBERG_H
#define IDMRG_MODELS_HEISENBERG_H

#include "itensor/all.h"
#include "idmrg/lattice/lattice.h"
#include "idmrg/config.h"

namespace idmrg {

using namespace itensor;

//
// Heisenberg/XXZ Model Hamiltonian Builder
//
// H = J Σ_{<i,j>} [Δ Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j)]
//   = J Σ_{<i,j>} [Δ Sz_i Sz_j + Sx_i Sx_j + Sy_i Sy_j]
//
// Parameters:
//   J:  Exchange coupling (> 0 antiferromagnetic)
//   Jz: Anisotropy (Δ = Jz/J), default Jz = J (isotropic)
//   J2: Next-nearest neighbor coupling (for J1-J2 model)
//   hz: Zeeman field in z-direction
//
class HeisenbergHamiltonian {
public:
    HeisenbergHamiltonian(SiteSet const& sites,
                         Lattice const& lattice,
                         Args const& args = Args::global())
        : sites_(sites),
          lattice_(lattice),
          N_(sites.length()),
          infinite_(args.getBool("Infinite", false))
    {
        J_  = args.getReal("J", 1.0);
        Jz_ = args.getReal("Jz", J_);
        J2_ = args.getReal("J2", 0.0);
        hz_ = args.getReal("hz", 0.0);
    }
    
    // Convert to MPO
    operator MPO() { return buildMPO(); }
    
    // Get the AutoMPO for inspection or modification
    AutoMPO getAutoMPO() const {
        return buildAutoMPO();
    }
    
    // Build parameters
    double J() const { return J_; }
    double Jz() const { return Jz_; }
    double J2() const { return J2_; }
    double hz() const { return hz_; }
    bool isInfinite() const { return infinite_; }

private:
    SiteSet const& sites_;
    Lattice const& lattice_;
    int N_;
    double J_, Jz_, J2_, hz_;
    bool infinite_;
    
    AutoMPO buildAutoMPO() const {
        AutoMPO ampo(sites_);
        
        // Exchange interactions on all bonds
        for (const auto& bond : lattice_.bonds()) {
            int i = bond.s1;
            int j = bond.s2;
            double w = bond.weight;
            
            // Determine coupling based on bond type
            double Jbond = J_;
            if (bond.type == "nnn") {
                Jbond = J2_;
            }
            
            if (std::abs(Jbond) < 1e-15) continue;
            
            double Jxy = Jbond * w;
            double Jzz = Jz_ * w * (bond.type == "nnn" ? J2_/J_ : 1.0);
            
            // Sz Sz term
            ampo += Jzz, "Sz", i, "Sz", j;
            
            // S+ S- + S- S+ = 2(Sx Sx + Sy Sy) terms
            ampo += 0.5 * Jxy, "S+", i, "S-", j;
            ampo += 0.5 * Jxy, "S-", i, "S+", j;
        }
        
        // Magnetic field term
        if (std::abs(hz_) > 1e-15) {
            for (int i = 1; i <= N_; ++i) {
                ampo += -hz_, "Sz", i;
            }
        }
        
        return ampo;
    }
    
    MPO buildMPO() const {
        auto ampo = buildAutoMPO();
        auto H = toMPO(ampo);
        
        if (infinite_) {
            // For infinite DMRG, we need special edge handling
            // The MPO is constructed for a finite system but will be
            // used with special boundary conditions
            setupInfiniteBoundary(H);
        }
        
        return H;
    }
    
    void setupInfiniteBoundary(MPO& H) const {
        // For iDMRG, we need to store the edge vectors
        // at positions 0 and N+1 (ITensor convention)
        // These are set up based on the MPO W-matrix structure
        
        // Get the MPO W-matrix structure
        // For Heisenberg: states are [end, start, Sz, S+, S-]
        // end state = 1, start state = 2
        
        if (H.length() >= 1) {
            // Left edge: picks out "start" state (row 2)
            auto ll = leftLinkIndex(H, 1);
            H.ref(0) = ITensor(dag(ll));
            H.ref(0).set(dag(ll)(2), 1.0);
            
            // Right edge: picks out "end" state (column 1)
            auto rl = rightLinkIndex(H, N_);
            H.ref(N_ + 1) = ITensor(rl);
            H.ref(N_ + 1).set(rl(1), 1.0);
        }
    }
};

//
// Convenience function to create Heisenberg MPO
//
inline MPO HeisenbergMPO(SiteSet const& sites, 
                        Lattice const& lattice,
                        Args const& args = Args::global()) {
    return HeisenbergHamiltonian(sites, lattice, args);
}

//
// Simplified version for 1D chain without lattice object
//
inline MPO HeisenbergChainMPO(SiteSet const& sites, Args const& args = Args::global()) {
    int N = sites.length();
    double J  = args.getReal("J", 1.0);
    double Jz = args.getReal("Jz", J);
    double hz = args.getReal("hz", 0.0);
    bool infinite = args.getBool("Infinite", false);
    
    auto ampo = AutoMPO(sites);
    
    for (int i = 1; i < N; ++i) {
        ampo += Jz, "Sz", i, "Sz", i+1;
        ampo += 0.5 * J, "S+", i, "S-", i+1;
        ampo += 0.5 * J, "S-", i, "S+", i+1;
    }
    
    if (std::abs(hz) > 1e-15) {
        for (int i = 1; i <= N; ++i) {
            ampo += -hz, "Sz", i;
        }
    }
    
    auto H = toMPO(ampo);
    
    if (infinite) {
        // Setup infinite boundary
        auto ll = leftLinkIndex(H, 1);
        H.ref(0) = ITensor(dag(ll));
        H.ref(0).set(dag(ll)(2), 1.0);
        
        auto rl = rightLinkIndex(H, N);
        H.ref(N + 1) = ITensor(rl);
        H.ref(N + 1).set(rl(1), 1.0);
    }
    
    return H;
}

//
// Transverse Field Ising Model
// H = -J Σ Sz_i Sz_j - h Σ Sx_i
//
inline MPO IsingMPO(SiteSet const& sites, 
                   Lattice const& lattice,
                   Args const& args = Args::global()) {
    double J = args.getReal("J", 1.0);
    double h = args.getReal("h", 0.0);
    bool infinite = args.getBool("Infinite", false);
    int N = sites.length();
    
    auto ampo = AutoMPO(sites);
    
    for (const auto& bond : lattice.bonds()) {
        ampo += -J * bond.weight, "Sz", bond.s1, "Sz", bond.s2;
    }
    
    for (int i = 1; i <= N; ++i) {
        ampo += -h, "Sx", i;
    }
    
    auto H = toMPO(ampo);
    
    if (infinite) {
        auto ll = leftLinkIndex(H, 1);
        H.ref(0) = ITensor(dag(ll));
        H.ref(0).set(dag(ll)(2), 1.0);
        
        auto rl = rightLinkIndex(H, N);
        H.ref(N + 1) = ITensor(rl);
        H.ref(N + 1).set(rl(1), 1.0);
    }
    
    return H;
}

} // namespace idmrg

#endif // IDMRG_MODELS_HEISENBERG_H
