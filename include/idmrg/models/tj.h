//
// t-J Model Hamiltonian
//

#ifndef IDMRG_MODELS_TJ_H
#define IDMRG_MODELS_TJ_H

#include "itensor/all.h"
#include "idmrg/lattice/lattice.h"
#include "idmrg/config.h"

namespace idmrg {

using namespace itensor;

//
// t-J Model Hamiltonian
//
// H = -t Σ_{<i,j>,σ} P(c†_{iσ} c_{jσ} + h.c.)P 
//   + J Σ_{<i,j>} (S_i · S_j - n_i n_j / 4)
//
// where P is the Gutzwiller projector (no double occupancy)
//
// Parameters:
//   t: Hopping amplitude
//   J: Exchange coupling
//   t': NNN hopping
//   J': NNN exchange
//
class tJHamiltonian {
public:
    tJHamiltonian(SiteSet const& sites,
                 Lattice const& lattice,
                 Args const& args = Args::global())
        : sites_(sites),
          lattice_(lattice),
          N_(sites.length()),
          infinite_(args.getBool("Infinite", false))
    {
        t_  = args.getReal("t", 1.0);
        J_  = args.getReal("J", 0.3);
        tp_ = args.getReal("tp", 0.0);
        Jp_ = args.getReal("Jp", 0.0);
    }
    
    operator MPO() { return buildMPO(); }
    
    AutoMPO getAutoMPO() const { return buildAutoMPO(); }

private:
    SiteSet const& sites_;
    Lattice const& lattice_;
    int N_;
    double t_, J_, tp_, Jp_;
    bool infinite_;
    
    AutoMPO buildAutoMPO() const {
        AutoMPO ampo(sites_);
        
        for (const auto& bond : lattice_.bonds()) {
            int i = bond.s1;
            int j = bond.s2;
            double w = bond.weight;
            
            double th = t_ * w;
            double Jh = J_ * w;
            
            if (bond.type == "nnn") {
                th = tp_ * w;
                Jh = Jp_ * w;
            }
            
            // Hopping terms (t-J uses projected operators)
            // For tJ site type, the operators already project
            if (std::abs(th) > 1e-15) {
                ampo += -th, "Cdagup", i, "Cup", j;
                ampo += -th, "Cdagup", j, "Cup", i;
                ampo += -th, "Cdagdn", i, "Cdn", j;
                ampo += -th, "Cdagdn", j, "Cdn", i;
            }
            
            // Exchange terms
            if (std::abs(Jh) > 1e-15) {
                // S_i · S_j = Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j)
                ampo += Jh, "Sz", i, "Sz", j;
                ampo += 0.5 * Jh, "S+", i, "S-", j;
                ampo += 0.5 * Jh, "S-", i, "S+", j;
                
                // -n_i n_j / 4 term
                ampo += -0.25 * Jh, "Ntot", i, "Ntot", j;
            }
        }
        
        return ampo;
    }
    
    MPO buildMPO() const {
        auto ampo = buildAutoMPO();
        auto H = toMPO(ampo);
        
        if (infinite_) {
            auto ll = leftLinkIndex(H, 1);
            H.ref(0) = ITensor(dag(ll));
            H.ref(0).set(dag(ll)(2), 1.0);
            
            auto rl = rightLinkIndex(H, N_);
            H.ref(N_ + 1) = ITensor(rl);
            H.ref(N_ + 1).set(rl(1), 1.0);
        }
        
        return H;
    }
};

inline MPO tJMPO(SiteSet const& sites,
                Lattice const& lattice,
                Args const& args = Args::global()) {
    return tJHamiltonian(sites, lattice, args);
}

} // namespace idmrg

#endif // IDMRG_MODELS_TJ_H
