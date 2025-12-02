//
// 2D Square Lattice
// Supports cylinder and torus geometries for iDMRG
//

#ifndef IDMRG_LATTICE_SQUARE_H
#define IDMRG_LATTICE_SQUARE_H

#include "lattice.h"
#include <cassert>

namespace idmrg {

//
// Site ordering strategies for 2D -> 1D mapping
//
enum class SiteOrdering {
    RowMajor,    // Row by row (natural for OBC)
    ColumnMajor, // Column by column
    Snake,       // Alternating direction (reduces bond dimension for cylinders)
    Hilbert      // Hilbert curve (experimental)
};

//
// 2D Square lattice mapped to 1D for MPS
// Supports cylinder (periodic in Y) and torus (periodic in X and Y) geometries
//
class SquareLattice : public Lattice {
public:
    //
    // Constructor
    // @param Lx: Length in x-direction (along cylinder axis)
    // @param Ly: Length in y-direction (cylinder circumference)
    // @param ordering: Site ordering for 1D mapping
    //
    SquareLattice(int Lx, int Ly, 
                  SiteOrdering ordering = SiteOrdering::Snake)
        : Lx_(Lx), Ly_(Ly), ordering_(ordering)
    {
        buildLattice();
    }
    
    int numSites() const override { return Lx_ * Ly_; }
    LatticeType type() const override { return LatticeType::Square; }
    std::string name() const override { return "2D Square Lattice"; }
    int dimension() const override { return 2; }
    
    std::vector<int> dimensions() const override {
        return {Lx_, Ly_};
    }
    
    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    
    int unitCellSize() const override {
        // For cylinder iDMRG, unit cell is one column (Ly sites)
        return isInfinite() ? Ly_ : Lx_ * Ly_;
    }
    
    // Convert (x, y) coordinates to 1D site index (1-based)
    int siteIndex(int x, int y) const {
        assert(x >= 0 && x < Lx_ && y >= 0 && y < Ly_);
        switch (ordering_) {
            case SiteOrdering::RowMajor:
                return y * Lx_ + x + 1;
            case SiteOrdering::ColumnMajor:
                return x * Ly_ + y + 1;
            case SiteOrdering::Snake:
            default:
                // Snake ordering: alternate direction in each column
                if (x % 2 == 0) {
                    return x * Ly_ + y + 1;
                } else {
                    return x * Ly_ + (Ly_ - 1 - y) + 1;
                }
        }
    }
    
    // Convert 1D index back to (x, y) coordinates
    std::pair<int, int> siteCoords(int index) const {
        assert(index >= 1 && index <= numSites());
        int idx = index - 1;
        switch (ordering_) {
            case SiteOrdering::RowMajor:
                return {idx % Lx_, idx / Lx_};
            case SiteOrdering::ColumnMajor:
                return {idx / Ly_, idx % Ly_};
            case SiteOrdering::Snake:
            default:
                int x = idx / Ly_;
                int y = idx % Ly_;
                if (x % 2 == 1) y = Ly_ - 1 - y;
                return {x, y};
        }
    }
    
    // Check if geometry is a cylinder (periodic in y, open in x)
    bool isCylinder() const {
        return ybc_ == BoundaryCondition::Periodic && 
               xbc_ == BoundaryCondition::Open;
    }
    
    // Check if geometry is a torus (periodic in both)
    bool isTorus() const {
        return ybc_ == BoundaryCondition::Periodic && 
               xbc_ == BoundaryCondition::Periodic;
    }
    
    // Geometry string
    std::string geometry() const {
        if (isTorus()) return "Torus";
        if (isCylinder()) return "Cylinder";
        if (xbc_ == BoundaryCondition::Periodic) return "X-Cylinder";
        return "Open";
    }
    
    std::string info() const override {
        std::ostringstream oss;
        oss << Lattice::info();
        oss << "  Dimensions: " << Lx_ << " x " << Ly_ << "\n";
        oss << "  Geometry: " << geometry() << "\n";
        return oss.str();
    }

private:
    int Lx_, Ly_;
    SiteOrdering ordering_;
    
    void buildLattice() {
        // Create site coordinates based on ordering
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                int idx = siteIndex(x, y);
                sites_.push_back(SiteCoord(idx, x, y));
            }
        }
        // Sort sites by index for consistent access
        std::sort(sites_.begin(), sites_.end(), 
                  [](const SiteCoord& a, const SiteCoord& b) {
                      return a.index < b.index;
                  });
        
        // Build bonds
        for (int x = 0; x < Lx_; ++x) {
            for (int y = 0; y < Ly_; ++y) {
                int s = siteIndex(x, y);
                
                // X-direction bond (along cylinder axis)
                if (x < Lx_ - 1) {
                    addBond(s, siteIndex(x+1, y), 1.0, "x");
                } else if (xbc_ == BoundaryCondition::Periodic && Lx_ > 2) {
                    addBond(s, siteIndex(0, y), 1.0, "x");
                }
                
                // Y-direction bond (around cylinder)
                if (y < Ly_ - 1) {
                    addBond(s, siteIndex(x, y+1), 1.0, "y");
                } else if (ybc_ == BoundaryCondition::Periodic && Ly_ > 2) {
                    addBond(s, siteIndex(x, 0), 1.0, "y");
                }
            }
        }
    }
};

//
// Square lattice with next-nearest neighbor bonds (J1-J2 model)
//
class SquareLatticeJ1J2 : public SquareLattice {
public:
    SquareLatticeJ1J2(int Lx, int Ly, 
                      SiteOrdering ordering = SiteOrdering::Snake)
        : SquareLattice(Lx, Ly, ordering)
    {
        addNNNBonds();
    }
    
    std::string name() const override { return "2D Square Lattice (J1-J2)"; }

private:
    void addNNNBonds() {
        int Lx = dimensions()[0];
        int Ly = dimensions()[1];
        
        for (int x = 0; x < Lx; ++x) {
            for (int y = 0; y < Ly; ++y) {
                int s = siteIndex(x, y);
                
                // Diagonal bonds: (+1, +1) and (+1, -1)
                if (x < Lx - 1) {
                    if (y < Ly - 1) {
                        addBond(s, siteIndex(x+1, y+1), 1.0, "nnn");
                    } else if (yBC() == BoundaryCondition::Periodic) {
                        addBond(s, siteIndex(x+1, 0), 1.0, "nnn");
                    }
                    
                    if (y > 0) {
                        addBond(s, siteIndex(x+1, y-1), 1.0, "nnn");
                    } else if (yBC() == BoundaryCondition::Periodic) {
                        addBond(s, siteIndex(x+1, Ly-1), 1.0, "nnn");
                    }
                }
            }
        }
    }
};

} // namespace idmrg

#endif // IDMRG_LATTICE_SQUARE_H
