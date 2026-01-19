//
// GPU SVD Benchmark
// Demonstrates GPU-accelerated SVD vs CPU
//

#include "idmrg/gpu.h"
#include <iostream>
#include <random>
#include <chrono>
#include <vector>

using namespace idmrg;
using namespace idmrg::gpu;

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double stopMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Generate random matrix
std::vector<double> randomMatrix(int m, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    
    std::vector<double> mat(m * n);
    for (auto& x : mat) {
        x = dis(gen);
    }
    return mat;
}

int main(int argc, char* argv[]) {
    int size = 1000;  // Matrix dimension
    int repeats = 10;  // Number of repetitions
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i+1 < argc) size = std::stoi(argv[++i]);
        else if (arg == "--repeats" && i+1 < argc) repeats = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: benchmark_gpu_svd [options]\n";
            std::cout << "  --size N      Matrix dimension (default: 1000)\n";
            std::cout << "  --repeats N   Number of repetitions (default: 10)\n";
            return 0;
        }
    }
    
    std::cout << "\n============================================\n";
    std::cout << "       GPU SVD Benchmark\n";
    std::cout << "============================================\n\n";
    
    // Print build config
    printBuildConfig();
    
    // Initialize GPU
    GPUAccelerator gpuAccel(0);
    gpuAccel.printStatus();
    
    if (!gpuAccel.isEnabled()) {
        std::cout << "\nGPU not available. Exiting.\n";
        return 1;
    }
    
    std::cout << "\nMatrix size: " << size << " x " << size << "\n";
    std::cout << "Repetitions: " << repeats << "\n\n";
    
#if IDMRG_GPU_ENABLED
    // Ensure GPU is initialized
    auto& ctx = GPUContext::instance();
    if (!ctx.isInitialized()) {
        ctx.initialize(0);
    }
    
    Timer timer;
    double total_gpu_time = 0;
    
    std::cout << "Running GPU SVD benchmark...\n";
    
    for (int r = 0; r < repeats; ++r) {
        // Create random matrix
        auto host_data = randomMatrix(size, size);
        
        // Create GPU matrix
        GPUMatrix<double> gpu_mat(size, size);
        copyToDevice(host_data.data(), gpu_mat.data(), size * size);
        
        // Warm up (first iteration)
        if (r == 0) {
            auto result = svd(gpu_mat, 1e-14, size);
            cudaDeviceSynchronize();
        }
        
        // Timed run
        timer.start();
        auto result = svd(gpu_mat, 1e-14, size);
        cudaDeviceSynchronize();
        double elapsed = timer.stopMs();
        
        total_gpu_time += elapsed;
        
        std::cout << "  Run " << (r+1) << ": " << elapsed << " ms, rank = " << result.rank << "\n";
    }
    
    double avg_gpu = total_gpu_time / repeats;
    
    std::cout << "\n============================================\n";
    std::cout << "Results:\n";
    std::cout << "  GPU SVD average: " << avg_gpu << " ms\n";
    std::cout << "  Throughput: " << (size * size * sizeof(double) / (avg_gpu / 1000.0)) / (1024*1024*1024) << " GB/s\n";
    std::cout << "============================================\n\n";
    
    // Memory info
    std::cout << "GPU Memory after benchmark:\n";
    std::cout << "  Free: " << ctx.freeMemory() / (1024.0*1024*1024) << " GB\n";
    std::cout << "  Total: " << ctx.totalMemory() / (1024.0*1024*1024) << " GB\n\n";
    
#else
    std::cout << "GPU support not compiled in.\n";
#endif
    
    return 0;
}
