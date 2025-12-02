//
// 2D Triangular Lattice
//

#ifndef IDMRG_LATTICE_TRIANGULAR_H
#define IDMRG_LATTICE_TRIANGULAR_H

#include "lattice.h"
#include <cmath>

namespace idmrg {

//
// 2D Triangular lattice
// Each site has 6 nearest neighbors
// Mapped to 1D using snake ordering
//
class TriangularLattice : public Lattice {
public:
    //
    // Constructor
    // @param Lx: Length in x-direction
    // @param Ly: Width in y-direction
    //
    TriangularLattice(int Lx, int Ly)
        : Lx_(Lx), Ly_(Ly)
    {
        buildLattice();
    }
    
    int numSites() const override { return Lx_ * Ly_; }
    LatticeType type() const override { return LatticeType::Triangular; }
    std::string name() const override { return "2D Triangular Lattice"; }
    int dimension() const override { return 2; }
    
    std::vector<int> dimensions() const override {
        return {Lx_, Ly_};
    }
    
    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    
    // Snake ordering for optimal MPS
    int siteIndex(int x, int y) const {
        if (x % 2 == 0) {
            return x * Ly_ + y + 1;
        } else {
            return x * Ly_ + (Ly_ - 1 - y) + 1;
        }
    }
    
    std::pair<int, int> siteCoords(int index) const {
        int idx = index - 1;
        int x = idx / Ly_;
        int y = idx % Ly_;
        if (x % 2 == 1) y = Ly_ - 1 - y;
        return {x, y};
    }

private:
    int Lx_, Ly_;
    
    void buildLattice() {
        // Site coordinates in real space
        // Triangular lattice: a1 = (1, 0), a2 = (1/2, sqrt(3)/2)
        const double sqrt3_2 = std::sqrt(3.0) / 2.0;
        
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                int idx = siteIndex(x, y);
                double rx = x + 0.5 * y;
                double ry = sqrt3_2 * y;
                addSite(idx, rx, ry);
            }
        }
        
        // Sort sites by index
        std::sort(sites_.begin(), sites_.end(), 
                  [](const SiteCoord& a, const SiteCoord& b) {
                      return a.index < b.index;
                  });
        
        // Build bonds - 6 neighbors per site
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                int s = siteIndex(x, y);
                
                // +x direction
                if (x < Lx_ - 1) {
                    addBond(s, siteIndex(x+1, y), 1.0, "a1");
                } else if (xbc_ == BoundaryCondition::Periodic) {
                    addBond(s, siteIndex(0, y), 1.0, "a1");
                }
                
                // +y direction
                if (y < Ly_ - 1) {
                    addBond(s, siteIndex(x, y+1), 1.0, "a2");
                } else if (ybc_ == BoundaryCondition::Periodic) {
                    addBond(s, siteIndex(x, 0), 1.0, "a2");
                }
                
                // +x-y direction (diagonal)
                if (x < Lx_ - 1 && y > 0) {
                    addBond(s, siteIndex(x+1, y-1), 1.0, "a1-a2");
                } else if (xbc_ == BoundaryCondition::Periodic && 
                          ybc_ == BoundaryCondition::Periodic) {
                    if (x < Lx_ - 1) {
                        addBond(s, siteIndex(x+1, Ly_-1), 1.0, "a1-a2");
                    } else if (y > 0) {
                        addBond(s, siteIndex(0, y-1), 1.0, "a1-a2");
                    } else {
                        addBond(s, siteIndex(0, Ly_-1), 1.0, "a1-a2");
                    }
                }
            }
        }
    }
};

} // namespace idmrg

#endif // IDMRG_LATTICE_TRIANGULAR_H
