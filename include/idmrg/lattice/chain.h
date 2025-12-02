//
// 1D Chain Lattice
//

#ifndef IDMRG_LATTICE_CHAIN_H
#define IDMRG_LATTICE_CHAIN_H

#include "lattice.h"

namespace idmrg {

//
// 1D Chain lattice with optional next-nearest neighbor bonds
//
class Chain : public Lattice {
public:
    //
    // Constructor
    // @param N: Number of sites
    // @param include_nnn: Include next-nearest neighbor bonds
    //
    Chain(int N, bool include_nnn = false)
        : N_(N), include_nnn_(include_nnn) 
    {
        buildLattice();
    }
    
    int numSites() const override { return N_; }
    LatticeType type() const override { return LatticeType::Chain; }
    std::string name() const override { return "1D Chain"; }
    int dimension() const override { return 1; }
    
    std::vector<int> dimensions() const override {
        return {N_};
    }
    
    int unitCellSize() const override {
        // For iDMRG with translational invariance
        return isInfinite() ? N_ / 2 : N_;
    }
    
    bool hasNNN() const { return include_nnn_; }

private:
    int N_;
    bool include_nnn_;
    
    void buildLattice() {
        // Create site coordinates
        for (int i = 1; i <= N_; ++i) {
            addSite(i, static_cast<double>(i-1), 0.0);
        }
        
        // Nearest-neighbor bonds
        for (int i = 1; i < N_; ++i) {
            addBond(i, i+1, 1.0, "nn");
        }
        
        // Periodic boundary
        if (xbc_ == BoundaryCondition::Periodic && N_ > 2) {
            addBond(N_, 1, 1.0, "nn");
        }
        
        // Next-nearest neighbor bonds
        if (include_nnn_) {
            for (int i = 1; i < N_ - 1; ++i) {
                addBond(i, i+2, 1.0, "nnn");
            }
            if (xbc_ == BoundaryCondition::Periodic && N_ > 3) {
                addBond(N_-1, 1, 1.0, "nnn");
                addBond(N_, 2, 1.0, "nnn");
            }
        }
    }
};

//
// Two-leg ladder lattice
//
class Ladder : public Lattice {
public:
    //
    // Constructor
    // @param Lx: Length along the ladder (number of rungs)
    // @param include_diagonal: Include diagonal bonds
    //
    Ladder(int Lx, bool include_diagonal = false)
        : Lx_(Lx), include_diagonal_(include_diagonal)
    {
        buildLattice();
    }
    
    int numSites() const override { return 2 * Lx_; }
    LatticeType type() const override { return LatticeType::Ladder; }
    std::string name() const override { return "2-Leg Ladder"; }
    int dimension() const override { return 2; }
    
    std::vector<int> dimensions() const override {
        return {Lx_, 2};
    }
    
    int unitCellSize() const override {
        return isInfinite() ? Lx_ : 2 * Lx_;
    }
    
    // Site indexing: (x, leg) -> index (1-based)
    // Using snake ordering for efficient MPS representation
    int siteIndex(int x, int leg) const {
        // Snake ordering: alternate direction on each rung
        return 2 * x + leg + 1;  // Simple ordering: all leg 0 first, then leg 1
    }
    
    std::pair<int, int> siteCoords(int index) const {
        int x = (index - 1) / 2;
        int leg = (index - 1) % 2;
        return {x, leg};
    }

private:
    int Lx_;
    bool include_diagonal_;
    
    void buildLattice() {
        // Create site coordinates
        // Using snake ordering for better MPS representation
        for (int x = 0; x < Lx_; ++x) {
            for (int leg = 0; leg < 2; ++leg) {
                int idx = siteIndex(x, leg);
                addSite(idx, static_cast<double>(x), static_cast<double>(leg));
            }
        }
        
        // Leg bonds (along the ladder)
        for (int x = 0; x < Lx_ - 1; ++x) {
            for (int leg = 0; leg < 2; ++leg) {
                addBond(siteIndex(x, leg), siteIndex(x+1, leg), 1.0, "leg");
            }
        }
        
        // Rung bonds
        for (int x = 0; x < Lx_; ++x) {
            addBond(siteIndex(x, 0), siteIndex(x, 1), 1.0, "rung");
        }
        
        // Diagonal bonds
        if (include_diagonal_) {
            for (int x = 0; x < Lx_ - 1; ++x) {
                addBond(siteIndex(x, 0), siteIndex(x+1, 1), 1.0, "diag");
                addBond(siteIndex(x, 1), siteIndex(x+1, 0), 1.0, "diag");
            }
        }
        
        // Periodic boundary along ladder direction
        if (xbc_ == BoundaryCondition::Periodic && Lx_ > 2) {
            for (int leg = 0; leg < 2; ++leg) {
                addBond(siteIndex(Lx_-1, leg), siteIndex(0, leg), 1.0, "leg");
            }
            if (include_diagonal_) {
                addBond(siteIndex(Lx_-1, 0), siteIndex(0, 1), 1.0, "diag");
                addBond(siteIndex(Lx_-1, 1), siteIndex(0, 0), 1.0, "diag");
            }
        }
    }
};

} // namespace idmrg

#endif // IDMRG_LATTICE_CHAIN_H
