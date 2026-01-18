//
// Hubbard Model Hamiltonian
// Supports 1D chains and 2D lattices with quantum number conservation
//

#ifndef IDMRG_MODELS_HUBBARD_H
#define IDMRG_MODELS_HUBBARD_H

#include "itensor/all.h"
#include "idmrg/lattice/lattice.h"
#include "idmrg/config.h"

namespace idmrg {

using namespace itensor;

//
// Hubbard Model Hamiltonian
//
// H = -t Σ_{<i,j>,σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
//   + V Σ_{<i,j>} n_i n_j - μ Σ_i n_i
//
// Parameters:
//   t:  Hopping amplitude
//   U:  On-site Coulomb repulsion
//   V:  Nearest-neighbor interaction (extended Hubbard)
//   mu: Chemical potential
//   t': Next-nearest neighbor hopping
//
class HubbardHamiltonian {
public:
    HubbardHamiltonian(SiteSet const& sites,
                      Lattice const& lattice,
                      Args const& args = Args::global())
        : sites_(sites),
          lattice_(lattice),
          N_(sites.length()),
          infinite_(args.getBool("Infinite", false))
    {
        t_  = args.getReal("t", 1.0);
        U_  = args.getReal("U", 4.0);
        V_  = args.getReal("V", 0.0);
        mu_ = args.getReal("mu", 0.0);
        tp_ = args.getReal("tp", 0.0);  // NNN hopping
    }
    
    operator MPO() { return buildMPO(); }
    
    AutoMPO getAutoMPO() const { return buildAutoMPO(); }
    
    double t() const { return t_; }
    double U() const { return U_; }
    double V() const { return V_; }
    double mu() const { return mu_; }
    double tp() const { return tp_; }
    bool isInfinite() const { return infinite_; }

private:
    SiteSet const& sites_;
    Lattice const& lattice_;
    int N_;
    double t_, U_, V_, mu_, tp_;
    bool infinite_;
    
    AutoMPO buildAutoMPO() const {
        AutoMPO ampo(sites_);
        
        // Assume Hubbard/Electron site type (siteType() removed in ITensor v3)
        bool isHubbardSite = true;
        
        // Hopping terms
        for (const auto& bond : lattice_.bonds()) {
            int i = bond.s1;
            int j = bond.s2;
            double w = bond.weight;
            
            // Determine hopping amplitude
            double hop = t_;
            if (bond.type == "nnn") {
                hop = tp_;
            }
            
            if (std::abs(hop) < 1e-15) continue;
            
            double th = hop * w;
            
            if (isHubbardSite) {
                // Using Hubbard site operators
                ampo += -th, "Cdagup", i, "Cup", j;
                ampo += -th, "Cdagup", j, "Cup", i;
                ampo += -th, "Cdagdn", i, "Cdn", j;
                ampo += -th, "Cdagdn", j, "Cdn", i;
            } else {
                // Using Electron site operators with explicit spin
                ampo += -th, "Cdagup", i, "Cup", j;
                ampo += -th, "Cdagup", j, "Cup", i;
                ampo += -th, "Cdagdn", i, "Cdn", j;
                ampo += -th, "Cdagdn", j, "Cdn", i;
            }
        }
        
        // On-site interaction
        if (std::abs(U_) > 1e-15) {
            for (int i = 1; i <= N_; ++i) {
                ampo += U_, "Nupdn", i;
            }
        }
        
        // Nearest-neighbor interaction (extended Hubbard)
        if (std::abs(V_) > 1e-15) {
            for (const auto& bond : lattice_.bonds()) {
                if (bond.type != "nnn") {
                    ampo += V_ * bond.weight, "Ntot", bond.s1, "Ntot", bond.s2;
                }
            }
        }
        
        // Chemical potential
        if (std::abs(mu_) > 1e-15) {
            for (int i = 1; i <= N_; ++i) {
                ampo += -mu_, "Ntot", i;
            }
        }
        
        return ampo;
    }
    
    MPO buildMPO() const {
        auto ampo = buildAutoMPO();
        auto H = toMPO(ampo);
        
        if (infinite_) {
            setupInfiniteBoundary(H);
        }
        
        return H;
    }
    
    void setupInfiniteBoundary(MPO& H) const {
        if (H.length() >= 1) {
            auto ll = leftLinkIndex(H, 1);
            H.ref(0) = ITensor(dag(ll));
            H.ref(0).set(dag(ll)(2), 1.0);
            
            auto rl = rightLinkIndex(H, N_);
            H.ref(N_ + 1) = ITensor(rl);
            H.ref(N_ + 1).set(rl(1), 1.0);
        }
    }
};

//
// Convenience function
//
inline MPO HubbardMPO(SiteSet const& sites,
                     Lattice const& lattice,
                     Args const& args = Args::global()) {
    return HubbardHamiltonian(sites, lattice, args);
}

//
// Simplified 1D Hubbard chain
//
inline MPO HubbardChainMPO(SiteSet const& sites, Args const& args = Args::global()) {
    int N = sites.length();
    double t  = args.getReal("t", 1.0);
    double U  = args.getReal("U", 4.0);
    double mu = args.getReal("mu", 0.0);
    bool infinite = args.getBool("Infinite", false);
    
    auto ampo = AutoMPO(sites);
    
    for (int i = 1; i < N; ++i) {
        ampo += -t, "Cdagup", i, "Cup", i+1;
        ampo += -t, "Cdagup", i+1, "Cup", i;
        ampo += -t, "Cdagdn", i, "Cdn", i+1;
        ampo += -t, "Cdagdn", i+1, "Cdn", i;
    }
    
    for (int i = 1; i <= N; ++i) {
        ampo += U, "Nupdn", i;
        if (std::abs(mu) > 1e-15) {
            ampo += -mu, "Ntot", i;
        }
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

#endif // IDMRG_MODELS_HUBBARD_H
