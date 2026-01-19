//
// ITensor GPU Acceleration Layer
// Provides GPU-accelerated operations for ITensor tensors
//

#ifndef IDMRG_ITENSOR_GPU_H
#define IDMRG_ITENSOR_GPU_H

#include "gpu_backend.h"
#include "gpu_svd.h"
#include "gpu_contraction.h"
#include "itensor/all.h"
#include <vector>
#include <chrono>

namespace idmrg {
namespace gpu {

using namespace itensor;

//=============================================================================
// Configuration for GPU offloading decisions
//=============================================================================

struct GPUConfig {
    // Minimum tensor size (total elements) to consider GPU offloading
    size_t min_tensor_size = 10000;
    
    // Minimum matrix dimension for SVD to use GPU
    int min_svd_dim = 256;
    
    // Minimum matrix dimension for GEMM to use GPU
    int min_gemm_dim = 128;
    
    // Enable timing output
    bool timing = false;
    
    // Singleton access
    static GPUConfig& instance() {
        static GPUConfig config;
        return config;
    }
};

//=============================================================================
// Timer for performance measurements
//=============================================================================

class GPUTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        total_ms_ += ms;
        count_++;
        return ms;
    }
    
    double totalMs() const { return total_ms_; }
    int count() const { return count_; }
    double avgMs() const { return count_ > 0 ? total_ms_ / count_ : 0; }
    
    void reset() { total_ms_ = 0; count_ = 0; }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
    double total_ms_ = 0;
    int count_ = 0;
};

// Global timers for profiling
struct GPUTimers {
    GPUTimer svd_total;
    GPUTimer svd_gpu;
    GPUTimer svd_cpu;
    GPUTimer contraction_total;
    GPUTimer contraction_gpu;
    GPUTimer contraction_cpu;
    GPUTimer data_transfer;
    
    static GPUTimers& instance() {
        static GPUTimers timers;
        return timers;
    }
    
    void printSummary() const {
        std::cout << "\n=== GPU Performance Summary ===\n";
        std::cout << "SVD Operations:\n";
        std::cout << "  Total: " << svd_total.count() << " calls, " 
                  << svd_total.totalMs() << " ms\n";
        std::cout << "  GPU: " << svd_gpu.count() << " calls, "
                  << svd_gpu.totalMs() << " ms\n";
        std::cout << "  CPU: " << svd_cpu.count() << " calls, "
                  << svd_cpu.totalMs() << " ms\n";
        std::cout << "Contractions:\n";
        std::cout << "  Total: " << contraction_total.count() << " calls, "
                  << contraction_total.totalMs() << " ms\n";
        std::cout << "  GPU: " << contraction_gpu.count() << " calls, "
                  << contraction_gpu.totalMs() << " ms\n";
        std::cout << "  CPU: " << contraction_cpu.count() << " calls, "
                  << contraction_cpu.totalMs() << " ms\n";
        std::cout << "Data Transfer: " << data_transfer.totalMs() << " ms\n";
        std::cout << "================================\n\n";
    }
    
    void reset() {
        svd_total.reset();
        svd_gpu.reset();
        svd_cpu.reset();
        contraction_total.reset();
        contraction_gpu.reset();
        contraction_cpu.reset();
        data_transfer.reset();
    }
};

#if IDMRG_GPU_ENABLED

//=============================================================================
// ITensor to/from GPU Matrix Conversion
//=============================================================================

// Extract data from an ITensor into a GPU matrix
// Assumes ITensor has been reshaped/combined to have exactly 2 indices
template<typename T = Real>
GPUMatrix<T> itensorToGPUMatrix(ITensor const& A, 
                                  Index const& row_ind,
                                  Index const& col_ind) {
    auto& timers = GPUTimers::instance();
    timers.data_transfer.start();
    
    size_t m = dim(row_ind);
    size_t n = dim(col_ind);
    
    // Create host vector and fill from ITensor
    std::vector<T> host_data(m * n);
    
    // ITensor stores in column-major by default
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            host_data[j * m + i] = elt(A, static_cast<int>(i+1), static_cast<int>(j+1));
        }
    }
    
    // Create GPU matrix and copy
    GPUMatrix<T> gpu_mat(m, n);
    copyToDevice(host_data.data(), gpu_mat.data(), m * n);
    
    timers.data_transfer.stop();
    return gpu_mat;
}

// Convert GPU matrix back to ITensor
template<typename T = Real>
ITensor gpuMatrixToITensor(GPUMatrix<T> const& mat,
                           Index const& row_ind,
                           Index const& col_ind) {
    auto& timers = GPUTimers::instance();
    timers.data_transfer.start();
    
    size_t m = mat.rows();
    size_t n = mat.cols();
    
    // Copy to host
    std::vector<T> host_data(m * n);
    copyToHost(host_data.data(), mat.data(), m * n);
    
    // Create ITensor
    auto result = ITensor(row_ind, col_ind);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            result.set(static_cast<int>(i+1), static_cast<int>(j+1), 
                      host_data[j * m + i]);
        }
    }
    
    timers.data_transfer.stop();
    return result;
}

//=============================================================================
// GPU-Accelerated SVD for ITensor
//=============================================================================

// Wrapper for SVD that uses GPU when beneficial
// Returns (U, S, V) decomposition: T = U * S * V
// Where S is diagonal matrix of singular values
inline std::tuple<ITensor, ITensor, ITensor>
svdGPU(ITensor const& T,
       Index const& Uis,   // Index going to U
       Index const& Vis,   // Index going to V
       Args const& args = Args::global()) {
    
    auto& config = GPUConfig::instance();
    auto& timers = GPUTimers::instance();
    auto& ctx = GPUContext::instance();
    
    timers.svd_total.start();
    
    // Get dimensions
    int m = dim(Uis);
    int n = dim(Vis);
    int k = std::min(m, n);
    
    // Check if GPU is beneficial
    bool use_gpu = ctx.isInitialized() && 
                   (m >= config.min_svd_dim || n >= config.min_svd_dim);
    
    if (!use_gpu) {
        // Fall back to ITensor SVD
        timers.svd_cpu.start();
        ITensor U, S, V;
        auto [u_comb, v_comb] = factor(T, {Uis}, {Vis}, args);
        svd(T, U, S, V, args);
        timers.svd_cpu.stop();
        timers.svd_total.stop();
        return {U, S, V};
    }
    
    timers.svd_gpu.start();
    
    // Parse args
    Real cutoff = args.getReal("Cutoff", 1E-14);
    int maxdim = args.getInt("MaxDim", k);
    
    // Convert to GPU matrix
    auto gpu_mat = itensorToGPUMatrix(T, Uis, Vis);
    
    // Perform GPU SVD
    auto result = svd(gpu_mat, cutoff, maxdim);
    
    // Create output index for truncated dimension
    int kept = result.rank;
    if (kept > maxdim) kept = maxdim;
    if (kept < 1) kept = 1;
    
    auto s_ind = Index(kept, "Link,SVD");
    
    // Convert U matrix back
    GPUMatrix<Real> U_trunc(m, kept);
    // Copy first 'kept' columns
    GPU_CHECK(cudaMemcpy2D(
        U_trunc.data(), m * sizeof(double),
        result.U.data(), m * sizeof(double),
        m * sizeof(double), kept,
        cudaMemcpyDeviceToDevice));
    ITensor U = gpuMatrixToITensor(U_trunc, Uis, s_ind);
    
    // Create diagonal S matrix
    ITensor S(s_ind, prime(s_ind));
    for (int i = 0; i < kept; ++i) {
        S.set(i+1, i+1, result.S[i]);
    }
    
    // Convert Vt matrix back (need to transpose)
    GPUMatrix<Real> Vt_trunc(kept, n);
    // Copy first 'kept' rows
    GPU_CHECK(cudaMemcpy2D(
        Vt_trunc.data(), kept * sizeof(double),
        result.Vt.data(), result.Vt.rows() * sizeof(double),
        kept * sizeof(double), n,
        cudaMemcpyDeviceToDevice));
    
    // V = Vt^T
    ITensor V = gpuMatrixToITensor(Vt_trunc, prime(s_ind), Vis);
    V = swapInds(V, {prime(s_ind)}, {dag(s_ind)});
    
    timers.svd_gpu.stop();
    timers.svd_total.stop();
    
    return {U, S, V};
}

//=============================================================================
// GPU-Accelerated Tensor Contraction
//=============================================================================

// Contract two ITensors using GPU GEMM when beneficial
// This is a simplified version that handles the most common case:
// Matrix-matrix multiplication after combining indices
inline ITensor
contractGPU(ITensor const& A, ITensor const& B,
            IndexSet const& common_inds) {
    
    auto& config = GPUConfig::instance();
    auto& timers = GPUTimers::instance();
    auto& ctx = GPUContext::instance();
    
    timers.contraction_total.start();
    
    // Get the indices
    auto Ainds = inds(A);
    auto Binds = inds(B);
    
    // Find contracted and uncontracted indices
    IndexSet A_only, B_only, contracted;
    for (auto const& i : Ainds) {
        bool found = false;
        for (auto const& j : Binds) {
            if (i == j || i == dag(j)) {
                contracted = IndexSet(contracted, i);
                found = true;
                break;
            }
        }
        if (!found) A_only = IndexSet(A_only, i);
    }
    for (auto const& i : Binds) {
        bool found = false;
        for (auto const& j : contracted) {
            if (i == j || i == dag(j)) {
                found = true;
                break;
            }
        }
        if (!found) B_only = IndexSet(B_only, i);
    }
    
    // Calculate dimensions
    long m = 1, k = 1, n = 1;
    for (auto const& i : A_only) m *= dim(i);
    for (auto const& i : contracted) k *= dim(i);
    for (auto const& i : B_only) n *= dim(i);
    
    // Check if GPU is beneficial
    bool use_gpu = ctx.isInitialized() &&
                   (m >= config.min_gemm_dim || 
                    n >= config.min_gemm_dim || 
                    k >= config.min_gemm_dim);
    
    if (!use_gpu || length(A_only) == 0 || length(B_only) == 0 || 
        length(contracted) == 0) {
        // Fall back to ITensor contraction
        timers.contraction_cpu.start();
        auto result = A * B;
        timers.contraction_cpu.stop();
        timers.contraction_total.stop();
        return result;
    }
    
    timers.contraction_gpu.start();
    
    // Combine indices for matrix multiplication
    auto [Ac, Acomb] = combiner(A_only);
    auto [Bc, Bcomb] = combiner(B_only);
    auto [Cc, Ccomb] = combiner(contracted);
    
    auto A_mat = A * Ac * Cc;
    auto B_mat = dag(Cc) * B * Bc;
    
    // Get combined index
    auto row_ind = commonIndex(A_mat, Ac);
    auto mid_ind = commonIndex(A_mat, Cc);
    auto col_ind = commonIndex(B_mat, Bc);
    
    // Convert to GPU matrices
    auto gpu_A = itensorToGPUMatrix(A_mat, row_ind, dag(mid_ind));
    auto gpu_B = itensorToGPUMatrix(B_mat, mid_ind, col_ind);
    
    // Allocate result
    GPUMatrix<Real> gpu_C(m, n);
    
    // Perform GPU GEMM: C = A * B
    gemm(Transpose::NoTrans, Transpose::NoTrans,
         static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
         1.0, gpu_A.data(), static_cast<int>(m),
         gpu_B.data(), static_cast<int>(k),
         0.0, gpu_C.data(), static_cast<int>(m));
    
    // Convert back
    auto result_mat = gpuMatrixToITensor(gpu_C, row_ind, col_ind);
    
    // Uncombine indices
    auto result = result_mat * dag(Ac) * dag(Bc);
    
    timers.contraction_gpu.stop();
    timers.contraction_total.stop();
    
    return result;
}

//=============================================================================
// Batched Operations for Sweep Optimizations
//=============================================================================

// Batch multiple SVD operations for better GPU utilization
class BatchedSVD {
public:
    struct SVDTask {
        ITensor tensor;
        Index row_ind;
        Index col_ind;
        Args args;
    };
    
    void add(ITensor const& T, Index const& Uis, Index const& Vis,
             Args const& args = Args::global()) {
        tasks_.push_back({T, Uis, Vis, args});
    }
    
    std::vector<std::tuple<ITensor, ITensor, ITensor>> execute() {
        std::vector<std::tuple<ITensor, ITensor, ITensor>> results;
        results.reserve(tasks_.size());
        
        // For now, execute sequentially
        // Future: batch small SVDs together
        for (auto const& task : tasks_) {
            results.push_back(svdGPU(task.tensor, task.row_ind, 
                                     task.col_ind, task.args));
        }
        
        tasks_.clear();
        return results;
    }
    
private:
    std::vector<SVDTask> tasks_;
};

#else // !IDMRG_GPU_ENABLED

// Stub implementations when GPU is disabled
template<typename T = Real>
GPUMatrix<T> itensorToGPUMatrix(ITensor const&, Index const&, Index const&) {
    throw std::runtime_error("GPU support not enabled");
}

template<typename T = Real>
ITensor gpuMatrixToITensor(GPUMatrix<T> const&, Index const&, Index const&) {
    throw std::runtime_error("GPU support not enabled");
}

inline std::tuple<ITensor, ITensor, ITensor>
svdGPU(ITensor const& T, Index const&, Index const&, Args const& args = Args::global()) {
    // Fall back to CPU
    ITensor U, S, V;
    svd(T, U, S, V, args);
    return {U, S, V};
}

inline ITensor contractGPU(ITensor const& A, ITensor const& B, IndexSet const&) {
    return A * B;
}

#endif // IDMRG_GPU_ENABLED

//=============================================================================
// Hybrid CPU/GPU Manager
//=============================================================================

class HybridCompute {
public:
    HybridCompute() {
#if IDMRG_GPU_ENABLED
        gpu_available_ = GPUContext::instance().isInitialized();
#else
        gpu_available_ = false;
#endif
    }
    
    bool gpuAvailable() const { return gpu_available_; }
    
    // Adaptive SVD: chooses best backend based on size
    std::tuple<ITensor, ITensor, ITensor>
    svd(ITensor const& T, Index const& Uis, Index const& Vis,
        Args const& args = Args::global()) {
        
        int m = dim(Uis);
        int n = dim(Vis);
        
        if (gpu_available_ && (m > 256 || n > 256)) {
            return svdGPU(T, Uis, Vis, args);
        } else {
            ITensor U, S, V;
            itensor::svd(T, U, S, V, args);
            return {U, S, V};
        }
    }
    
    // Adaptive contraction
    ITensor contract(ITensor const& A, ITensor const& B) {
        auto Ainds = inds(A);
        auto Binds = inds(B);
        
        // Find common indices
        IndexSet common;
        for (auto const& i : Ainds) {
            for (auto const& j : Binds) {
                if (i == j || i == dag(j)) {
                    common = IndexSet(common, i);
                }
            }
        }
        
        // Estimate size
        long total_size = 1;
        for (auto const& i : Ainds) total_size *= dim(i);
        for (auto const& i : Binds) total_size *= dim(i);
        
        if (gpu_available_ && total_size > 100000) {
            return contractGPU(A, B, common);
        } else {
            return A * B;
        }
    }
    
    void printStats() const {
#if IDMRG_GPU_ENABLED
        GPUTimers::instance().printSummary();
#endif
    }
    
    void resetStats() {
#if IDMRG_GPU_ENABLED
        GPUTimers::instance().reset();
#endif
    }
    
private:
    bool gpu_available_;
};

// Global hybrid compute instance
inline HybridCompute& hybridCompute() {
    static HybridCompute hc;
    return hc;
}

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_ITENSOR_GPU_H
