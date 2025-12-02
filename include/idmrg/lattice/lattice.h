//
// Base Lattice Class
// Provides abstract interface for various lattice geometries
//

#ifndef IDMRG_LATTICE_LATTICE_H
#define IDMRG_LATTICE_LATTICE_H

#include <vector>
#include <utility>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <sstream>
#include "idmrg/config.h"

namespace idmrg {

//
// Bond structure representing a connection between two sites
//
struct Bond {
    int s1;         // First site index (1-based for ITensor compatibility)
    int s2;         // Second site index
    double weight;  // Bond weight/strength multiplier
    std::string type;  // Bond type identifier (e.g., "x", "y", "nn", "nnn")
    
    Bond(int site1, int site2, double w = 1.0, std::string t = "nn")
        : s1(site1), s2(site2), weight(w), type(std::move(t)) {}
    
    // Ensure s1 < s2 for consistent ordering
    void normalize() {
        if (s1 > s2) std::swap(s1, s2);
    }
    
    bool operator==(const Bond& other) const {
        return (s1 == other.s1 && s2 == other.s2) ||
               (s1 == other.s2 && s2 == other.s1);
    }
    
    std::string toString() const {
        std::ostringstream oss;
        oss << "Bond(" << s1 << ", " << s2 << ", w=" << weight << ", type=" << type << ")";
        return oss.str();
    }
};

//
// Site coordinate in real space
//
struct SiteCoord {
    int index;      // 1-based site index
    double x, y, z; // Real space coordinates
    int ux, uy;     // Unit cell coordinates (for periodic systems)
    int sublattice; // Sublattice index (for non-Bravais lattices)
    
    SiteCoord() : index(0), x(0), y(0), z(0), ux(0), uy(0), sublattice(0) {}
    SiteCoord(int idx, double px, double py, double pz = 0.0)
        : index(idx), x(px), y(py), z(pz), ux(0), uy(0), sublattice(0) {}
    
    double distance(const SiteCoord& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

//
// Abstract base class for lattice geometries
//
class Lattice {
public:
    virtual ~Lattice() = default;
    
    // Core interface
    virtual int numSites() const = 0;
    virtual int numBonds() const { return static_cast<int>(bonds_.size()); }
    virtual LatticeType type() const = 0;
    virtual std::string name() const = 0;
    
    // Geometry information
    virtual int dimension() const = 0;
    virtual std::vector<int> dimensions() const = 0;
    
    // Unit cell for iDMRG
    virtual int unitCellSize() const { return numSites(); }
    virtual bool isInfinite() const { return infinite_; }
    void setInfinite(bool inf) { infinite_ = inf; }
    
    // Boundary conditions
    BoundaryCondition xBC() const { return xbc_; }
    BoundaryCondition yBC() const { return ybc_; }
    void setXBC(BoundaryCondition bc) { xbc_ = bc; }
    void setYBC(BoundaryCondition bc) { ybc_ = bc; }
    
    // Bond access
    const std::vector<Bond>& bonds() const { return bonds_; }
    const Bond& bond(int i) const { return bonds_.at(i); }
    
    // Site coordinate access
    const std::vector<SiteCoord>& sites() const { return sites_; }
    const SiteCoord& site(int i) const { return sites_.at(i-1); } // 1-based
    
    // Find bonds involving a specific site
    std::vector<Bond> bondsAt(int site) const {
        std::vector<Bond> result;
        for (const auto& b : bonds_) {
            if (b.s1 == site || b.s2 == site) {
                result.push_back(b);
            }
        }
        return result;
    }
    
    // Find nearest neighbors of a site
    std::vector<int> neighbors(int site) const {
        std::vector<int> result;
        for (const auto& b : bonds_) {
            if (b.s1 == site) result.push_back(b.s2);
            else if (b.s2 == site) result.push_back(b.s1);
        }
        return result;
    }
    
    // Maximum bond range (for MPO construction)
    int maxBondRange() const {
        int maxRange = 0;
        for (const auto& b : bonds_) {
            maxRange = std::max(maxRange, std::abs(b.s2 - b.s1));
        }
        return maxRange;
    }
    
    // Filter bonds by type
    std::vector<Bond> bondsByType(const std::string& btype) const {
        std::vector<Bond> result;
        for (const auto& b : bonds_) {
            if (b.type == btype) result.push_back(b);
        }
        return result;
    }
    
    // Print lattice information
    virtual std::string info() const {
        std::ostringstream oss;
        oss << "Lattice: " << name() << "\n";
        oss << "  Sites: " << numSites() << "\n";
        oss << "  Bonds: " << numBonds() << "\n";
        oss << "  Dimension: " << dimension() << "\n";
        oss << "  Infinite: " << (infinite_ ? "Yes" : "No") << "\n";
        oss << "  X boundary: " << to_string(xbc_) << "\n";
        if (dimension() > 1) {
            oss << "  Y boundary: " << to_string(ybc_) << "\n";
        }
        oss << "  Max bond range: " << maxBondRange() << "\n";
        return oss.str();
    }

protected:
    std::vector<Bond> bonds_;
    std::vector<SiteCoord> sites_;
    bool infinite_ = false;
    BoundaryCondition xbc_ = BoundaryCondition::Open;
    BoundaryCondition ybc_ = BoundaryCondition::Open;
    
    // Helper to add a bond (checks for duplicates)
    void addBond(int s1, int s2, double w = 1.0, const std::string& type = "nn") {
        Bond b(s1, s2, w, type);
        b.normalize();
        // Check for duplicate
        for (const auto& existing : bonds_) {
            if (existing == b) return;
        }
        bonds_.push_back(b);
    }
    
    // Helper to add site coordinate
    void addSite(int idx, double x, double y, double z = 0.0) {
        SiteCoord s(idx, x, y, z);
        sites_.push_back(s);
    }
};

//
// Bond iterator for range-based for loops
//
class BondIterator {
public:
    using iterator = std::vector<Bond>::const_iterator;
    
    BondIterator(const Lattice& lat) : lattice_(lat) {}
    
    iterator begin() const { return lattice_.bonds().begin(); }
    iterator end() const { return lattice_.bonds().end(); }
    
private:
    const Lattice& lattice_;
};

inline BondIterator allBonds(const Lattice& lat) {
    return BondIterator(lat);
}

} // namespace idmrg

#endif // IDMRG_LATTICE_LATTICE_H
