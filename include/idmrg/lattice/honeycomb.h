//
// 2D Honeycomb Lattice
//

#ifndef IDMRG_LATTICE_HONEYCOMB_H
#define IDMRG_LATTICE_HONEYCOMB_H

#include "lattice.h"
#include <cmath>

namespace idmrg {

//
// 2D Honeycomb (graphene) lattice
// Two-atom basis (A and B sublattices)
// Each site has 3 nearest neighbors
//
class HoneycombLattice : public Lattice {
public:
    //
    // Constructor
    // @param Lx: Number of unit cells in x-direction
    // @param Ly: Number of unit cells in y-direction
    //
    HoneycombLattice(int Lx, int Ly)
        : Lx_(Lx), Ly_(Ly)
    {
        buildLattice();
    }
    
    int numSites() const override { return 2 * Lx_ * Ly_; }
    LatticeType type() const override { return LatticeType::Honeycomb; }
    std::string name() const override { return "2D Honeycomb Lattice"; }
    int dimension() const override { return 2; }
    
    std::vector<int> dimensions() const override {
        return {Lx_, Ly_, 2};  // 2 for sublattice
    }
    
    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    
    // Site indexing: (x, y, sublattice) -> 1D index
    // sublattice: 0 = A, 1 = B
    int siteIndex(int x, int y, int sub) const {
        // Optimized ordering for MPS
        int cell = x * Ly_ + y;
        return 2 * cell + sub + 1;
    }
    
    std::tuple<int, int, int> siteCoords(int index) const {
        int idx = index - 1;
        int sub = idx % 2;
        int cell = idx / 2;
        int y = cell % Ly_;
        int x = cell / Ly_;
        return {x, y, sub};
    }

private:
    int Lx_, Ly_;
    
    void buildLattice() {
        // Honeycomb lattice vectors
        // a1 = (sqrt(3), 0)
        // a2 = (sqrt(3)/2, 3/2)
        // Basis: A at (0, 0), B at (sqrt(3)/2, 1/2)
        const double sqrt3 = std::sqrt(3.0);
        
        // Create sites
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                // A sublattice
                int idxA = siteIndex(x, y, 0);
                double rxA = sqrt3 * x + sqrt3/2 * y;
                double ryA = 1.5 * y;
                SiteCoord sA(idxA, rxA, ryA);
                sA.sublattice = 0;
                sA.ux = x;
                sA.uy = y;
                sites_.push_back(sA);
                
                // B sublattice
                int idxB = siteIndex(x, y, 1);
                double rxB = rxA + sqrt3/2;
                double ryB = ryA + 0.5;
                SiteCoord sB(idxB, rxB, ryB);
                sB.sublattice = 1;
                sB.ux = x;
                sB.uy = y;
                sites_.push_back(sB);
            }
        }
        
        // Sort by index
        std::sort(sites_.begin(), sites_.end(), 
                  [](const SiteCoord& a, const SiteCoord& b) {
                      return a.index < b.index;
                  });
        
        // Build bonds - each A connects to 3 B sites
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                int sA = siteIndex(x, y, 0);
                int sB = siteIndex(x, y, 1);
                
                // A-B bond within unit cell
                addBond(sA, sB, 1.0, "intra");
                
                // A connects to B in cell (x-1, y) if exists
                if (x > 0) {
                    addBond(sA, siteIndex(x-1, y, 1), 1.0, "inter-x");
                } else if (xbc_ == BoundaryCondition::Periodic) {
                    addBond(sA, siteIndex(Lx_-1, y, 1), 1.0, "inter-x");
                }
                
                // A connects to B in cell (x, y-1) if exists
                if (y > 0) {
                    addBond(sA, siteIndex(x, y-1, 1), 1.0, "inter-y");
                } else if (ybc_ == BoundaryCondition::Periodic) {
                    addBond(sA, siteIndex(x, Ly_-1, 1), 1.0, "inter-y");
                }
            }
        }
    }
};

//
// Kagome lattice (corner-sharing triangles)
// Three-atom basis
//
class KagomeLattice : public Lattice {
public:
    KagomeLattice(int Lx, int Ly)
        : Lx_(Lx), Ly_(Ly)
    {
        buildLattice();
    }
    
    int numSites() const override { return 3 * Lx_ * Ly_; }
    LatticeType type() const override { return LatticeType::Kagome; }
    std::string name() const override { return "Kagome Lattice"; }
    int dimension() const override { return 2; }
    
    std::vector<int> dimensions() const override {
        return {Lx_, Ly_, 3};
    }
    
    // Site indexing: (x, y, sublattice) -> 1D index
    // sublattice: 0, 1, 2
    int siteIndex(int x, int y, int sub) const {
        int cell = x * Ly_ + y;
        return 3 * cell + sub + 1;
    }

private:
    int Lx_, Ly_;
    
    void buildLattice() {
        const double sqrt3 = std::sqrt(3.0);
        
        // Create sites with 3-site basis
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                double ox = 2.0 * x + y;  // Origin x
                double oy = sqrt3 * y;     // Origin y
                
                // Three sublattices
                addSite(siteIndex(x, y, 0), ox, oy);
                addSite(siteIndex(x, y, 1), ox + 1.0, oy);
                addSite(siteIndex(x, y, 2), ox + 0.5, oy + sqrt3/2);
            }
        }
        
        std::sort(sites_.begin(), sites_.end(), 
                  [](const SiteCoord& a, const SiteCoord& b) {
                      return a.index < b.index;
                  });
        
        // Build bonds within and between unit cells
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                // Bonds within unit cell (forming one triangle)
                addBond(siteIndex(x, y, 0), siteIndex(x, y, 1), 1.0, "intra");
                addBond(siteIndex(x, y, 1), siteIndex(x, y, 2), 1.0, "intra");
                addBond(siteIndex(x, y, 2), siteIndex(x, y, 0), 1.0, "intra");
                
                // Bonds to neighboring cells
                if (x < Lx_ - 1) {
                    addBond(siteIndex(x, y, 1), siteIndex(x+1, y, 0), 1.0, "inter");
                }
                if (y < Ly_ - 1) {
                    addBond(siteIndex(x, y, 2), siteIndex(x, y+1, 0), 1.0, "inter");
                }
                if (x < Lx_ - 1 && y > 0) {
                    addBond(siteIndex(x, y, 1), siteIndex(x+1, y-1, 2), 1.0, "inter");
                }
            }
        }
    }
};

} // namespace idmrg

#endif // IDMRG_LATTICE_HONEYCOMB_H
