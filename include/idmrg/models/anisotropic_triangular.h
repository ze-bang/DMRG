//
// Anisotropic Triangular Lattice Spin Model (XXZ + Bond-Dependent)
//
// This implements the Hamiltonian:
//
// H = H_XXZ + H_bd
//
// H_XXZ = J Σ_{<ij>} (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)
//
// H_bd = Σ_{<ij>} { 2J_±± (cos(φ̃_α)[x,y]_ij - sin(φ̃_α){x,y}_ij)
//                 + J_z± (cos(φ̃_α){y,z}_ij - sin(φ̃_α){x,z}_ij) }
//
// where:
//   [a,b]_ij = S^a_i S^a_j - S^b_i S^b_j
//   {a,b}_ij = S^a_i S^b_j + S^b_i S^a_j
//   φ̃_α = {0, -2π/3, 2π/3} for bonds along primitive vectors δ_1, δ_2, δ_3
//
// Reference: Magnetization processes in spin-orbit coupled triangular antiferromagnets
//

#ifndef IDMRG_MODELS_ANISOTROPIC_TRIANGULAR_H
#define IDMRG_MODELS_ANISOTROPIC_TRIANGULAR_H

#include "itensor/all.h"
#include "idmrg/lattice/triangular.h"
#include "idmrg/config.h"
#include <cmath>
#include <complex>

namespace idmrg {

using namespace itensor;

//
// Anisotropic Triangular Lattice Hamiltonian
//
// Bond directions on triangular lattice (primitive vectors):
//   δ_1 (a1): horizontal                    -> φ̃ = 0
//   δ_2 (a2): 60° from horizontal           -> φ̃ = -2π/3  
//   δ_3 (a3): 120° from horizontal          -> φ̃ = 2π/3
//
// Parameters:
//   J:      Overall exchange scale for XXZ part
//   Delta:  Ising anisotropy (0 ≤ Δ ≤ 1 for easy-plane)
//   Jpmpm:  J_±± bond-dependent term
//   Jzpm:   J_z± mixed term
//   hz:     External magnetic field
//
class AnisotropicTriangularHamiltonian {
public:
    AnisotropicTriangularHamiltonian(SiteSet const& sites,
                                     TriangularLattice const& lattice,
                                     Args const& args = Args::global())
        : sites_(sites),
          lattice_(lattice),
          N_(sites.length()),
          infinite_(args.getBool("Infinite", false))
    {
        // XXZ parameters
        J_     = args.getReal("J", 1.0);       // Exchange coupling
        Delta_ = args.getReal("Delta", 1.0);   // Ising anisotropy (Δ)
        
        // Bond-dependent parameters  
        Jpmpm_ = args.getReal("Jpmpm", 0.0);   // J_±± coupling
        Jzpm_  = args.getReal("Jzpm", 0.0);    // J_z± coupling
        
        // External field
        hz_    = args.getReal("hz", 0.0);
        
        // Phases for each bond direction: φ̃_α = {0, -2π/3, 2π/3}
        phi_a1_ = 0.0;
        phi_a2_ = -2.0 * M_PI / 3.0;
        phi_a3_ = 2.0 * M_PI / 3.0;
    }
    
    // Convert to MPO
    operator MPO() { return buildMPO(); }
    
    // Get the AutoMPO for inspection or modification
    AutoMPO getAutoMPO() const {
        return buildAutoMPO();
    }
    
    // Build parameters
    double J() const { return J_; }
    double Delta() const { return Delta_; }
    double Jpmpm() const { return Jpmpm_; }
    double Jzpm() const { return Jzpm_; }
    double hz() const { return hz_; }
    bool isInfinite() const { return infinite_; }

private:
    SiteSet const& sites_;
    TriangularLattice const& lattice_;
    int N_;
    double J_, Delta_, Jpmpm_, Jzpm_, hz_;
    bool infinite_;
    
    // Phase angles for each bond direction
    double phi_a1_, phi_a2_, phi_a3_;
    
    // Get the phase angle φ̃_α based on bond direction
    double getPhase(std::string const& bondType) const {
        if (bondType == "a1") {
            return phi_a1_;  // φ̃ = 0
        } else if (bondType == "a2") {
            return phi_a2_;  // φ̃ = -2π/3
        } else if (bondType == "a1-a2") {
            // This is the a3 direction (diagonal)
            return phi_a3_;  // φ̃ = 2π/3
        }
        return 0.0;
    }
    
    AutoMPO buildAutoMPO() const {
        AutoMPO ampo(sites_);
        
        // Loop over all bonds in the lattice
        for (const auto& bond : lattice_.bonds()) {
            int i = bond.s1;
            int j = bond.s2;
            double w = bond.weight;
            
            // Get the phase angle for this bond direction
            double phi = getPhase(bond.type);
            double cos_phi = std::cos(phi);
            double sin_phi = std::sin(phi);
            
            //=================================================================
            // H_XXZ = J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)
            //       = J Σ [½(S⁺_i S⁻_j + S⁻_i S⁺_j) + Δ S^z_i S^z_j]
            //=================================================================
            
            if (std::abs(J_) > 1e-15) {
                // XY terms: J(S^x S^x + S^y S^y) = (J/2)(S⁺S⁻ + S⁻S⁺)
                ampo += 0.5 * J_ * w, "S+", i, "S-", j;
                ampo += 0.5 * J_ * w, "S-", i, "S+", j;
                
                // Ising term: JΔ S^z S^z
                ampo += J_ * Delta_ * w, "Sz", i, "Sz", j;
            }
            
            //=================================================================
            // Bond-dependent J_±± term:
            // 2J_±± (cos(φ̃)[x,y]_ij - sin(φ̃){x,y}_ij)
            //
            // where [x,y] = S^x S^x - S^y S^y = ½(S⁺S⁺ + S⁻S⁻)
            //       {x,y} = S^x S^y + S^y S^x = (1/2i)(S⁺S⁺ - S⁻S⁻)
            //
            // This gives:
            // 2J_±± [cos(φ̃)·½(S⁺S⁺ + S⁻S⁻) - sin(φ̃)·(1/2i)(S⁺S⁺ - S⁻S⁻)]
            // = J_±± [cos(φ̃)(S⁺S⁺ + S⁻S⁻) + i·sin(φ̃)(S⁺S⁺ - S⁻S⁻)]
            // = J_±± [(cos(φ̃) + i·sin(φ̃))S⁺S⁺ + (cos(φ̃) - i·sin(φ̃))S⁻S⁻]
            // = J_±± [e^{iφ̃} S⁺_i S⁺_j + e^{-iφ̃} S⁻_i S⁻_j]
            //=================================================================
            
            if (std::abs(Jpmpm_) > 1e-15) {
                // e^{iφ̃} S⁺_i S⁺_j
                ampo += Cplx(Jpmpm_ * w * cos_phi, Jpmpm_ * w * sin_phi), "S+", i, "S+", j;
                
                // e^{-iφ̃} S⁻_i S⁻_j
                ampo += Cplx(Jpmpm_ * w * cos_phi, -Jpmpm_ * w * sin_phi), "S-", i, "S-", j;
            }
            
            //=================================================================
            // Bond-dependent J_z± term:
            // J_z± (cos(φ̃){y,z}_ij - sin(φ̃){x,z}_ij)
            //
            // where {y,z} = S^y S^z + S^z S^y = (1/2i)[(S⁺-S⁻)S^z + S^z(S⁺-S⁻)]
            //       {x,z} = S^x S^z + S^z S^x = (1/2)[(S⁺+S⁻)S^z + S^z(S⁺+S⁻)]
            //
            // Expanding:
            // J_z± [cos(φ̃)·(1/2i)(S⁺S^z - S⁻S^z + S^zS⁺ - S^zS⁻)
            //      - sin(φ̃)·(1/2)(S⁺S^z + S⁻S^z + S^zS⁺ + S^zS⁻)]
            //
            // Coefficients for each term:
            // S⁺_i S^z_j: (cos(φ̃)/2i - sin(φ̃)/2) = -(i/2)(cos(φ̃) - i·sin(φ̃)) = -(i/2)e^{-iφ̃}
            // S⁻_i S^z_j: (-cos(φ̃)/2i - sin(φ̃)/2) = (i/2)(cos(φ̃) + i·sin(φ̃)) = (i/2)e^{iφ̃}
            // S^z_i S⁺_j: (cos(φ̃)/2i - sin(φ̃)/2) = -(i/2)e^{-iφ̃}
            // S^z_i S⁻_j: (-cos(φ̃)/2i - sin(φ̃)/2) = (i/2)e^{iφ̃}
            //
            // So the full term is:
            // J_z± · (-i/2) [e^{-iφ̃}(S⁺_iS^z_j + S^z_iS⁺_j) - e^{iφ̃}(S⁻_iS^z_j + S^z_iS⁻_j)]
            //=================================================================
            
            if (std::abs(Jzpm_) > 1e-15) {
                // Coefficient: -i/2 * e^{-iφ̃} = -i/2 * (cos(φ̃) - i·sin(φ̃))
                //            = (-i·cos(φ̃) - sin(φ̃))/2 = -(sin(φ̃) + i·cos(φ̃))/2
                Cplx coeff_plus = Cplx(-sin_phi, -cos_phi) * (0.5 * Jzpm_ * w);
                
                // Coefficient: i/2 * e^{iφ̃} = i/2 * (cos(φ̃) + i·sin(φ̃))
                //            = (i·cos(φ̃) - sin(φ̃))/2 = (-sin(φ̃) + i·cos(φ̃))/2
                Cplx coeff_minus = Cplx(-sin_phi, cos_phi) * (0.5 * Jzpm_ * w);
                
                // e^{-iφ̃}(S⁺_iS^z_j + S^z_iS⁺_j) terms with coefficient -i/2
                ampo += coeff_plus, "S+", i, "Sz", j;
                ampo += coeff_plus, "Sz", i, "S+", j;
                
                // -e^{iφ̃}(S⁻_iS^z_j + S^z_iS⁻_j) terms with coefficient i/2
                // Note the minus sign is absorbed: we use -coeff_minus
                ampo += -coeff_minus, "S-", i, "Sz", j;
                ampo += -coeff_minus, "Sz", i, "S-", j;
            }
        }
        
        // External magnetic field term: -h_z Σ_i S^z_i
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
            setupInfiniteBoundary(H);
        }
        
        return H;
    }
    
    void setupInfiniteBoundary(MPO& H) const {
        if (H.length() >= 1) {
            // Left edge
            auto ll = leftLinkIndex(H, 1);
            H.ref(0) = ITensor(dag(ll));
            H.ref(0).set(dag(ll)(2), 1.0);
            
            // Right edge
            auto rl = rightLinkIndex(H, N_);
            H.ref(N_ + 1) = ITensor(rl);
            H.ref(N_ + 1).set(rl(1), 1.0);
        }
    }
};

//
// Convenience function to create Anisotropic Triangular Lattice MPO
//
// Parameters:
//   J:      Exchange coupling (default: 1.0)
//   Delta:  Ising anisotropy Δ, 0 ≤ Δ ≤ 1 for easy-plane (default: 1.0)
//   Jpmpm:  J_±± bond-dependent coupling (default: 0.0)
//   Jzpm:   J_z± mixed coupling (default: 0.0)
//   hz:     External field (default: 0.0)
//
inline MPO AnisotropicTriangularMPO(SiteSet const& sites, 
                                    TriangularLattice const& lattice,
                                    Args const& args = Args::global()) {
    return AnisotropicTriangularHamiltonian(sites, lattice, args);
}

//
// Special cases
//

// Standard Heisenberg on triangular lattice (Δ = 1, J_±± = J_z± = 0)
inline MPO HeisenbergTriangularMPO(SiteSet const& sites,
                                   TriangularLattice const& lattice,
                                   Args const& args = Args::global()) {
    auto newArgs = args;
    newArgs.add("Delta", 1.0);
    newArgs.add("Jpmpm", 0.0);
    newArgs.add("Jzpm", 0.0);
    return AnisotropicTriangularMPO(sites, lattice, newArgs);
}

// XXZ model on triangular lattice (no bond-dependent terms)
inline MPO XXZTriangularMPO(SiteSet const& sites,
                            TriangularLattice const& lattice,
                            Args const& args = Args::global()) {
    auto newArgs = args;
    newArgs.add("Jpmpm", 0.0);
    newArgs.add("Jzpm", 0.0);
    return AnisotropicTriangularMPO(sites, lattice, newArgs);
}

// Easy-plane XXZ + bond-dependent (typical for layered systems)
inline MPO EasyPlaneTriangularMPO(SiteSet const& sites,
                                  TriangularLattice const& lattice,
                                  Args const& args = Args::global()) {
    // For easy-plane systems, typically 0 < Δ < 1
    return AnisotropicTriangularMPO(sites, lattice, args);
}

} // namespace idmrg

#endif // IDMRG_MODELS_ANISOTROPIC_TRIANGULAR_H
