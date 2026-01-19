//
// Example: 2D Heisenberg Model with GPU Acceleration
//
// This benchmark demonstrates GPU-accelerated finite DMRG for the 
// 2D Heisenberg model on a square lattice cylinder.
//
// Features:
//   - GPU-accelerated tensor contractions (cuBLAS)
//   - GPU-accelerated SVD (cuSOLVER)
//   - Automatic CPU/GPU selection based on tensor size
//   - Performance profiling
//

#include "idmrg/all.h"
#include "idmrg/gpu.h"
#include <random>

using namespace itensor;
using namespace idmrg;

// Custom DMRG observer with GPU timing
class GPUDMRGObserver : public DMRGObserver {
public:
    GPUDMRGObserver(MPS const& psi, Args const& args = Args::global())
        : DMRGObserver(psi, args) {}
    
    void measure(Args const& args = Args::global()) override {
        DMRGObserver::measure(args);
        
        // Print GPU stats periodically
        if (args.getBool("AtCenter", false)) {
            int sweep = args.getInt("Sweep", 0);
            if (sweep > 0 && sweep % 5 == 0) {
#if IDMRG_GPU_ENABLED
                auto& timers = gpu::GPUTimers::instance();
                printfln("  [GPU] SVD: %d calls (%.1f ms), Contraction: %d calls (%.1f ms)",
                        timers.svd_gpu.count(), timers.svd_gpu.totalMs(),
                        timers.contraction_gpu.count(), timers.contraction_gpu.totalMs());
#endif
            }
        }
    }
};

int main(int argc, char* argv[]) {
    // Default parameters
    int Ly = 4;          // Cylinder circumference
    int Lx = 8;          // Length in X direction
    double J = 1.0;      // Exchange coupling
    int maxDim = 500;    // Maximum bond dimension
    int numSweeps = 15;  // Number of DMRG sweeps
    bool useGPU = true;  // Enable GPU acceleration
    int gpuDevice = 0;   // GPU device ID
    
    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--Ly" && i+1 < argc) Ly = std::stoi(argv[++i]);
        else if (arg == "--Lx" && i+1 < argc) Lx = std::stoi(argv[++i]);
        else if (arg == "--J" && i+1 < argc) J = std::stod(argv[++i]);
        else if (arg == "--maxdim" && i+1 < argc) maxDim = std::stoi(argv[++i]);
        else if (arg == "--sweeps" && i+1 < argc) numSweeps = std::stoi(argv[++i]);
        else if (arg == "--no-gpu") useGPU = false;
        else if (arg == "--gpu" && i+1 < argc) gpuDevice = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            println("Usage: heisenberg_gpu [options]");
            println("  --Lx N      Cylinder length (default: 8)");
            println("  --Ly N      Cylinder circumference (default: 4)");
            println("  --J val     Exchange coupling (default: 1.0)");
            println("  --maxdim N  Max bond dimension (default: 500)");
            println("  --sweeps N  Number of sweeps (default: 15)");
            println("  --no-gpu    Disable GPU acceleration");
            println("  --gpu N     GPU device ID (default: 0)");
            return 0;
        }
    }
    
    int N = Lx * Ly;
    
    println("\n==============================================");
    println("  2D Heisenberg Model - GPU Accelerated DMRG");
    println("==============================================\n");
    
    // Print build configuration
    printBuildConfig();
    
    // Initialize GPU if enabled
    GPUAccelerator gpuAccel(gpuDevice);
    if (useGPU && gpuAccel.isEnabled()) {
        println("GPU Acceleration: ENABLED");
        gpuAccel.printStatus();
    } else if (useGPU) {
        println("GPU Acceleration: REQUESTED but NOT AVAILABLE");
        println("Falling back to CPU.\n");
        useGPU = false;
    } else {
        println("GPU Acceleration: DISABLED by user\n");
    }
    
    printfln("System: %d x %d cylinder (%d sites)", Lx, Ly, N);
    printfln("  X direction: Open (length %d)", Lx);
    printfln("  Y direction: Periodic (circumference %d)", Ly);
    printfln("Exchange coupling J = %.4f", J);
    println("");
    
    printfln("Max bond dimension: %d", maxDim);
    printfln("Number of sweeps: %d\n", numSweeps);
    
    printfln("Reference (thermodynamic limit): E/N ≈ -0.6694\n");
    
    // Snake pattern site ordering
    auto coords = [Lx, Ly](int i) -> std::pair<int,int> {
        int row = (i - 1) / Lx;
        int col = (i - 1) % Lx;
        if (row % 2 == 1) col = Lx - 1 - col;
        return {col, row};
    };
    
    auto siteIndex = [Lx, Ly, &coords](int x, int y) -> int {
        for (int i = 1; i <= Lx*Ly; ++i) {
            auto [cx, cy] = coords(i);
            if (cx == x && cy == y) return i;
        }
        return -1;
    };
    
    // Create sites with QN conservation
    auto sites = SpinHalf(N, {"ConserveQNs=", true});
    
    // Build Hamiltonian
    println("Building 2D Heisenberg Hamiltonian...");
    auto ampo = AutoMPO(sites);
    
    int bondCount = 0;
    for (int i = 1; i <= N; ++i) {
        auto [x, y] = coords(i);
        
        // Horizontal bond
        if (x < Lx - 1) {
            int j = siteIndex(x + 1, y);
            if (j > 0 && j != i) {
                ampo += J,     "Sz", i, "Sz", j;
                ampo += 0.5*J, "S+", i, "S-", j;
                ampo += 0.5*J, "S-", i, "S+", j;
                bondCount++;
            }
        }
        
        // Vertical bond (periodic)
        int y2 = (y + 1) % Ly;
        int j = siteIndex(x, y2);
        if (j > 0 && j != i) {
            ampo += J,     "Sz", i, "Sz", j;
            ampo += 0.5*J, "S+", i, "S-", j;
            ampo += 0.5*J, "S-", i, "S+", j;
            bondCount++;
        }
    }
    
    auto H = toMPO(ampo);
    printfln("Number of bonds: %d", bondCount);
    printfln("MPO bond dimension: %d\n", maxLinkDim(H));
    
    // Sweeps schedule
    auto sweeps = Sweeps(numSweeps);
    sweeps.maxdim() = 50, 100, 200, 300, 400, maxDim;
    sweeps.cutoff() = 1E-8, 1E-10, 1E-12;
    sweeps.niter() = 4, 3, 2;
    sweeps.noise() = 1E-6, 1E-7, 1E-8, 1E-10, 0;
    
    println("Sweep schedule:");
    println(sweeps);
    println("");
    
    // Initialize Néel state
    println("Initializing MPS with Néel pattern...");
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        auto [x, y] = coords(i);
        state.set(i, ((x + y) % 2 == 0) ? "Up" : "Dn");
    }
    auto psi = MPS(state);
    printfln("Initial MPS bond dimension: %d\n", maxLinkDim(psi));
    
    // Reset GPU timers
#if IDMRG_GPU_ENABLED
    gpu::GPUTimers::instance().reset();
#endif
    
    // Run DMRG
    println("Starting DMRG optimization...\n");
    
    Timer timer;
    timer.start();
    
    // Use custom observer for GPU stats
    auto obs = GPUDMRGObserver(psi);
    
    auto [energy, psi_gs] = dmrg(H, psi, sweeps, obs, {"Quiet", false});
    
    timer.stop();
    
    // Results
    println("\n==============================================");
    println("                  Results                    ");
    println("==============================================");
    printfln("  Ground state energy: %.14f", energy);
    printfln("  Energy per site: %.14f", energy / N);
    printfln("  Final bond dimension: %d", maxLinkDim(psi_gs));
    printfln("  Time: %s", timer.elapsedStr().c_str());
    println("");
    
    // Compare to reference
    double E_ref = -0.6694;
    double E_per_site = energy / N;
    double error_pct = 100.0 * std::abs(E_per_site - E_ref) / std::abs(E_ref);
    
    printfln("Comparison to thermodynamic limit (E/N ≈ %.4f):", E_ref);
    printfln("  Finite-size E/N = %.6f", E_per_site);
    printfln("  Difference: %.4f%%", error_pct);
    println("");
    
    // Print GPU performance summary
#if IDMRG_GPU_ENABLED
    if (useGPU) {
        gpu::GPUTimers::instance().printSummary();
    }
#endif
    
    // Local magnetization
    println("Local <Sz> per site:");
    for (int y = 0; y < Ly; ++y) {
        print("  ");
        for (int x = 0; x < Lx; ++x) {
            int i = siteIndex(x, y);
            psi_gs.position(i);
            auto Sz_op = op(sites, "Sz", i);
            auto val = eltC(dag(prime(psi_gs(i), "Site")) * Sz_op * psi_gs(i));
            printf("%+.4f ", real(val));
        }
        println("");
    }
    println("");
    
    println("(See 'vN Entropy at center bond' in sweep output above)");
    println("");
    
    return 0;
}
