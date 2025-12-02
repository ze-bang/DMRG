//
// Example: Generic Lattice with Custom Hamiltonian
//
// Demonstrates how to:
// 1. Create a custom lattice geometry
// 2. Build a custom Hamiltonian using AutoMPO
// 3. Run iDMRG and measure observables
//

#include "itensor/all.h"
#include "idmrg/idmrg.h"
#include "idmrg/lattice/lattice.h"
#include "idmrg/lattice/triangular.h"
#include "idmrg/lattice/honeycomb.h"
#include "idmrg/util/timer.h"
#include "idmrg/util/io.h"

using namespace itensor;
using namespace idmrg;

//
// Custom lattice example: Frustrated ladder with cross-links
//
class FrustratedLadder : public Lattice {
public:
    FrustratedLadder(int Lx, double J_leg, double J_rung, double J_diag)
        : Lx_(Lx), J_leg_(J_leg), J_rung_(J_rung), J_diag_(J_diag)
    {
        buildLattice();
    }
    
    int numSites() const override { return 2 * Lx_; }
    LatticeType type() const override { return LatticeType::Custom; }
    std::string name() const override { return "Frustrated 2-Leg Ladder"; }
    int dimension() const override { return 2; }
    std::vector<int> dimensions() const override { return {Lx_, 2}; }
    
    // (rung, leg) -> site index
    int siteIndex(int rung, int leg) const {
        return 2 * rung + leg + 1;
    }

private:
    int Lx_;
    double J_leg_, J_rung_, J_diag_;
    
    void buildLattice() {
        // Create sites
        for (int x = 0; x < Lx_; ++x) {
            for (int leg = 0; leg < 2; ++leg) {
                addSite(siteIndex(x, leg), x, leg);
            }
        }
        
        // Leg bonds (along the ladder)
        for (int x = 0; x < Lx_ - 1; ++x) {
            addBond(siteIndex(x, 0), siteIndex(x+1, 0), J_leg_, "leg");
            addBond(siteIndex(x, 1), siteIndex(x+1, 1), J_leg_, "leg");
        }
        
        // Rung bonds
        for (int x = 0; x < Lx_; ++x) {
            addBond(siteIndex(x, 0), siteIndex(x, 1), J_rung_, "rung");
        }
        
        // Diagonal bonds (frustration)
        for (int x = 0; x < Lx_ - 1; ++x) {
            addBond(siteIndex(x, 0), siteIndex(x+1, 1), J_diag_, "diag");
            addBond(siteIndex(x, 1), siteIndex(x+1, 0), J_diag_, "diag");
        }
    }
};

//
// Build custom Hamiltonian from lattice
//
MPO buildCustomHamiltonian(SiteSet const& sites, 
                          Lattice const& lattice,
                          bool infinite = false) {
    int N = sites.length();
    auto ampo = AutoMPO(sites);
    
    // Add interactions based on bond types
    for (const auto& bond : lattice.bonds()) {
        int i = bond.s1;
        int j = bond.s2;
        double J = bond.weight;
        
        // Heisenberg interaction
        ampo += J, "Sz", i, "Sz", j;
        ampo += 0.5 * J, "S+", i, "S-", j;
        ampo += 0.5 * J, "S-", i, "S+", j;
    }
    
    auto H = toMPO(ampo);
    
    if (infinite) {
        // Set up boundary tensors for iDMRG
        auto ll = leftLinkIndex(H, 1);
        H.ref(0) = ITensor(dag(ll));
        H.ref(0).set(dag(ll)(2), 1.0);
        
        auto rl = rightLinkIndex(H, N);
        H.ref(N + 1) = ITensor(rl);
        H.ref(N + 1).set(rl(1), 1.0);
    }
    
    return H;
}

int main(int argc, char* argv[]) {
    printfln("\n=============================================");
    printfln("  Custom Lattice Example - iDMRG");
    printfln("=============================================\n");
    
    // Example 1: Frustrated ladder
    println("=== Frustrated 2-Leg Ladder ===\n");
    
    int Lx = 4;  // Length of unit cell (will use 2 for iDMRG)
    double J_leg = 1.0;
    double J_rung = 1.0;
    double J_diag = 0.5;  // Frustration strength
    
    int N = 2 * Lx;  // Total sites (2 unit cells)
    
    printfln("Lattice: Lx = %d, J_leg = %.2f, J_rung = %.2f, J_diag = %.2f",
            Lx, J_leg, J_rung, J_diag);
    
    auto ladder = FrustratedLadder(Lx, J_leg, J_rung, J_diag);
    ladder.setInfinite(true);
    
    println("\n" + ladder.info());
    
    // Create spin-1/2 sites
    auto sites = SpinHalf(N, {"ConserveQNs=", true});
    
    // Build Hamiltonian
    MPO H = buildCustomHamiltonian(sites, ladder, true);
    
    // Sweeps
    auto sweeps = Sweeps(15);
    sweeps.maxdim() = 50, 100, 200, 300, 400;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 3, 2;
    sweeps.noise() = 1E-7, 1E-8, 0;
    
    // Initialize Néel state
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        if (i % 2 == 1) state.set(i, "Up");
        else state.set(i, "Dn");
    }
    auto psi = MPS(state);
    
    // Run iDMRG
    Timer timer;
    timer.start();
    
    auto result = idmrg(psi, H, sweeps, {"OutputLevel", 1});
    
    timer.stop();
    
    printfln("\nFrustrated Ladder Results:");
    printfln("  Energy per site: %.14f", result.energy_per_site);
    printfln("  Entanglement entropy: %.10f", result.entropy);
    printfln("  Time: %s\n", timer.elapsedStr().c_str());
    
    // Example 2: Triangular strip
    println("\n=== Triangular Lattice Strip ===\n");
    
    int Lx_tri = 4;
    int Ly_tri = 3;
    int N_tri = Lx_tri * Ly_tri;
    
    printfln("Triangular strip: %d x %d = %d sites per unit cell", 
            Lx_tri, Ly_tri, N_tri);
    
    auto tri_lattice = TriangularLattice(Lx_tri, Ly_tri);
    tri_lattice.setYBC(BoundaryCondition::Periodic);  // Cylinder
    tri_lattice.setInfinite(true);
    
    println("\n" + tri_lattice.info());
    
    auto sites_tri = SpinHalf(2 * N_tri, {"ConserveQNs=", true});
    
    // Use generic Heisenberg interaction
    auto ampo_tri = AutoMPO(sites_tri);
    for (const auto& bond : tri_lattice.bonds()) {
        ampo_tri += 1.0, "Sz", bond.s1, "Sz", bond.s2;
        ampo_tri += 0.5, "S+", bond.s1, "S-", bond.s2;
        ampo_tri += 0.5, "S-", bond.s1, "S+", bond.s2;
    }
    auto H_tri = toMPO(ampo_tri);
    
    // Setup infinite boundaries
    {
        auto ll = leftLinkIndex(H_tri, 1);
        H_tri.ref(0) = ITensor(dag(ll));
        H_tri.ref(0).set(dag(ll)(2), 1.0);
        
        auto rl = rightLinkIndex(H_tri, 2 * N_tri);
        H_tri.ref(2 * N_tri + 1) = ITensor(rl);
        H_tri.ref(2 * N_tri + 1).set(rl(1), 1.0);
    }
    
    // Initialize
    auto state_tri = InitState(sites_tri);
    for (int i = 1; i <= 2 * N_tri; ++i) {
        if (i % 2 == 1) state_tri.set(i, "Up");
        else state_tri.set(i, "Dn");
    }
    auto psi_tri = MPS(state_tri);
    
    // Run
    timer.reset();
    timer.start();
    
    auto result_tri = idmrg(psi_tri, H_tri, sweeps, {"OutputLevel", 1});
    
    timer.stop();
    
    printfln("\nTriangular Strip Results:");
    printfln("  Energy per site: %.14f", result_tri.energy_per_site);
    printfln("  Entanglement entropy: %.10f", result_tri.entropy);
    printfln("  Time: %s\n", timer.elapsedStr().c_str());
    
    // Example 3: Honeycomb lattice
    println("\n=== Honeycomb Lattice Strip ===\n");
    
    int Lx_hc = 2;
    int Ly_hc = 3;
    int N_hc = 2 * Lx_hc * Ly_hc;  // 2 atoms per unit cell
    
    printfln("Honeycomb: %d x %d unit cells = %d sites per unit cell", 
            Lx_hc, Ly_hc, N_hc);
    
    auto hc_lattice = HoneycombLattice(Lx_hc, Ly_hc);
    hc_lattice.setYBC(BoundaryCondition::Periodic);
    hc_lattice.setInfinite(true);
    
    println("\n" + hc_lattice.info());
    
    auto sites_hc = SpinHalf(2 * N_hc, {"ConserveQNs=", true});
    
    auto ampo_hc = AutoMPO(sites_hc);
    for (const auto& bond : hc_lattice.bonds()) {
        ampo_hc += 1.0, "Sz", bond.s1, "Sz", bond.s2;
        ampo_hc += 0.5, "S+", bond.s1, "S-", bond.s2;
        ampo_hc += 0.5, "S-", bond.s1, "S+", bond.s2;
    }
    auto H_hc = toMPO(ampo_hc);
    
    {
        auto ll = leftLinkIndex(H_hc, 1);
        H_hc.ref(0) = ITensor(dag(ll));
        H_hc.ref(0).set(dag(ll)(2), 1.0);
        
        auto rl = rightLinkIndex(H_hc, 2 * N_hc);
        H_hc.ref(2 * N_hc + 1) = ITensor(rl);
        H_hc.ref(2 * N_hc + 1).set(rl(1), 1.0);
    }
    
    auto state_hc = InitState(sites_hc);
    for (int i = 1; i <= 2 * N_hc; ++i) {
        if (i % 2 == 1) state_hc.set(i, "Up");
        else state_hc.set(i, "Dn");
    }
    auto psi_hc = MPS(state_hc);
    
    timer.reset();
    timer.start();
    
    auto result_hc = idmrg(psi_hc, H_hc, sweeps, {"OutputLevel", 1});
    
    timer.stop();
    
    printfln("\nHoneycomb Results:");
    printfln("  Energy per site: %.14f", result_hc.energy_per_site);
    printfln("  Entanglement entropy: %.10f", result_hc.entropy);
    printfln("  Time: %s\n", timer.elapsedStr().c_str());
    
    println("\n=== Summary ===");
    println("Lattice              E/site          S_entangle");
    println("------------------------------------------------");
    printfln("Frustrated Ladder    %.10f    %.6f", result.energy_per_site, result.entropy);
    printfln("Triangular Strip     %.10f    %.6f", result_tri.energy_per_site, result_tri.entropy);
    printfln("Honeycomb Strip      %.10f    %.6f", result_hc.energy_per_site, result_hc.entropy);
    println("------------------------------------------------\n");
    
    println("Done!");
    
    return 0;
}
