//
// GPU Support Header
// Main include file for GPU acceleration
//

#ifndef IDMRG_GPU_H
#define IDMRG_GPU_H

#include "gpu/gpu_backend.h"
#include "gpu/gpu_tensor.h"
#include "gpu/gpu_svd.h"
#include "gpu/gpu_contraction.h"
#include "gpu/itensor_gpu.h"

namespace idmrg {

//=============================================================================
// GPU Accelerator Interface
// Provides high-level API for GPU-accelerated DMRG operations
//=============================================================================

class GPUAccelerator {
public:
    GPUAccelerator(int device_id = 0) : device_id_(device_id) {
#if IDMRG_GPU_ENABLED
        gpu::GPUContext::instance().initialize(device_id);
        enabled_ = true;
#else
        enabled_ = false;
#endif
    }
    
    ~GPUAccelerator() = default;
    
    // Check if GPU is available and enabled
    bool isEnabled() const { return enabled_; }
    
    // Get GPU info
    std::string deviceName() const {
#if IDMRG_GPU_ENABLED
        if (enabled_) {
            return gpu::GPUContext::instance().deviceName();
        }
#endif
        return "No GPU";
    }
    
    size_t totalMemory() const {
#if IDMRG_GPU_ENABLED
        if (enabled_) {
            return gpu::GPUContext::instance().totalMemory();
        }
#endif
        return 0;
    }
    
    size_t freeMemory() const {
#if IDMRG_GPU_ENABLED
        if (enabled_) {
            return gpu::GPUContext::instance().freeMemory();
        }
#endif
        return 0;
    }
    
    // Print GPU status
    void printStatus() const {
        std::cout << "GPU Accelerator Status:\n";
        std::cout << "  Backend: " << gpu::gpu_backend_name() << "\n";
        std::cout << "  Enabled: " << (enabled_ ? "Yes" : "No") << "\n";
        if (enabled_) {
            std::cout << "  Device: " << deviceName() << "\n";
            std::cout << "  Total Memory: " << totalMemory() / (1024*1024*1024.0) << " GB\n";
            std::cout << "  Free Memory: " << freeMemory() / (1024*1024*1024.0) << " GB\n";
        }
    }
    
    // Decision helper: should we use GPU for given tensor size?
    bool shouldUseGPU(size_t tensor_elements) const {
        if (!enabled_) return false;
        
        // Heuristic: GPU is beneficial for large tensors
        // Threshold depends on GPU architecture
        constexpr size_t GPU_THRESHOLD = 10000;  // ~80KB for doubles
        return tensor_elements > GPU_THRESHOLD;
    }

private:
    int device_id_;
    bool enabled_;
};

//=============================================================================
// Compile-time checks
//=============================================================================

// Helper to check GPU support at compile time
constexpr bool hasGPUSupport() {
    return gpu::gpu_enabled();
}

// Print build configuration
inline void printBuildConfig() {
    std::cout << "\n=== iDMRG Build Configuration ===\n";
    
    // BLAS backend
#if defined(IDMRG_USE_AOCL)
    std::cout << "BLAS Backend: AMD AOCL (BLIS + libFLAME)\n";
#elif defined(IDMRG_USE_MKL)
    std::cout << "BLAS Backend: Intel MKL\n";
#else
    std::cout << "BLAS Backend: System BLAS/LAPACK\n";
#endif
    
    // GPU backend
    std::cout << "GPU Backend: " << gpu::gpu_backend_name() << "\n";
    
#if defined(IDMRG_USE_CUTENSOR)
    std::cout << "cuTENSOR: Enabled\n";
#endif
    
    std::cout << "================================\n\n";
}

} // namespace idmrg

#endif // IDMRG_GPU_H
