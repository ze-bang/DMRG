//
// GPU SVD Implementation
// Singular Value Decomposition using cuSOLVER (CUDA) or rocSOLVER (HIP)
//

#ifndef IDMRG_GPU_SVD_H
#define IDMRG_GPU_SVD_H

#include "gpu_backend.h"
#include "gpu_tensor.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace idmrg {
namespace gpu {

#if IDMRG_GPU_ENABLED

//=============================================================================
// SVD Result Structure
//=============================================================================

template<typename T>
struct SVDResult {
    GPUMatrix<T> U;      // Left singular vectors (m x k)
    std::vector<T> S;    // Singular values (k)
    GPUMatrix<T> Vt;     // Right singular vectors transposed (k x n)
    int rank;            // Effective rank
    T truncation_error;  // Frobenius norm of discarded part
    
    SVDResult() : rank(0), truncation_error(0) {}
};

//=============================================================================
// GPU SVD Implementation
//=============================================================================

#ifdef IDMRG_GPU_BACKEND_CUDA

// Full SVD: A = U * S * Vt
template<typename T>
SVDResult<T> svd(const GPUMatrix<T>& A, 
                 T cutoff = 1e-14, 
                 int max_rank = -1);

// Specialization for double
template<>
inline SVDResult<double> svd(const GPUMatrix<double>& A, 
                             double cutoff, 
                             int max_rank) 
{
    auto& ctx = GPUContext::instance();
    if (!ctx.isInitialized()) {
        throw std::runtime_error("GPU context not initialized");
    }
    
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());
    int k = std::min(m, n);
    
    if (max_rank > 0 && max_rank < k) {
        k = max_rank;
    }
    
    SVDResult<double> result;
    
    // Allocate device memory for SVD
    DevicePtr<double> d_A = allocateDevice<double>(m * n);
    DevicePtr<double> d_S = allocateDevice<double>(k);
    DevicePtr<double> d_U = allocateDevice<double>(m * k);
    DevicePtr<double> d_Vt = allocateDevice<double>(k * n);
    DevicePtr<int> d_info = allocateDevice<int>(1);
    
    // Copy input matrix
    GPU_CHECK(cudaMemcpy(d_A.get(), A.data(), m * n * sizeof(double), 
                         cudaMemcpyDeviceToDevice));
    
    // Query workspace size
    int lwork;
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(
        ctx.cusolverHandle(), m, n, &lwork));
    
    DevicePtr<double> d_work = allocateDevice<double>(lwork);
    DevicePtr<double> d_rwork = allocateDevice<double>(std::min(m, n) - 1);
    
    // Compute SVD
    // 'S' = compute min(m,n) singular values
    // 'S' for U means compute first min(m,n) columns of U
    signed char jobu = 'S';
    signed char jobvt = 'S';
    
    CUSOLVER_CHECK(cusolverDnDgesvd(
        ctx.cusolverHandle(),
        jobu, jobvt,
        m, n,
        d_A.get(), m,      // Input matrix
        d_S.get(),          // Singular values
        d_U.get(), m,       // Left singular vectors
        d_Vt.get(), k,      // Right singular vectors (transposed)
        d_work.get(), lwork,
        d_rwork.get(),
        d_info.get()
    ));
    
    synchronize();
    
    // Check for errors
    int info;
    copyToHost(&info, d_info.get(), 1);
    if (info != 0) {
        throw std::runtime_error("cusolverDnDgesvd failed with info = " + 
                                 std::to_string(info));
    }
    
    // Copy singular values to host for truncation decision
    std::vector<double> S(k);
    copyToHost(S.data(), d_S.get(), k);
    
    // Determine truncation
    double total_weight = 0.0;
    for (int i = 0; i < k; ++i) {
        total_weight += S[i] * S[i];
    }
    
    // Find truncation rank based on cutoff
    int new_rank = k;
    double truncation_weight = 0.0;
    
    for (int i = k - 1; i >= 0; --i) {
        double relative_error = std::sqrt(truncation_weight / total_weight);
        if (S[i] > cutoff && relative_error < cutoff) {
            new_rank = i + 1;
            break;
        }
        truncation_weight += S[i] * S[i];
        new_rank = i;
    }
    
    if (new_rank == 0) new_rank = 1;  // Keep at least one
    
    // Apply max_rank constraint
    if (max_rank > 0 && new_rank > max_rank) {
        new_rank = max_rank;
    }
    
    // Prepare result
    result.rank = new_rank;
    result.S.assign(S.begin(), S.begin() + new_rank);
    result.truncation_error = 0.0;
    for (int i = new_rank; i < k; ++i) {
        result.truncation_error += S[i] * S[i];
    }
    result.truncation_error = std::sqrt(result.truncation_error);
    
    // Copy truncated U and Vt
    result.U = GPUMatrix<double>(m, new_rank);
    result.Vt = GPUMatrix<double>(new_rank, n);
    
    // Copy column-major submatrices
    for (int j = 0; j < new_rank; ++j) {
        GPU_CHECK(cudaMemcpy(result.U.data() + j * m,
                             d_U.get() + j * m,
                             m * sizeof(double),
                             cudaMemcpyDeviceToDevice));
    }
    
    for (int i = 0; i < new_rank; ++i) {
        GPU_CHECK(cudaMemcpy(result.Vt.data() + i * n,
                             d_Vt.get() + i * n,
                             n * sizeof(double),
                             cudaMemcpyDeviceToDevice));
    }
    
    return result;
}

// Specialization for float
template<>
inline SVDResult<float> svd(const GPUMatrix<float>& A, 
                            float cutoff, 
                            int max_rank) 
{
    auto& ctx = GPUContext::instance();
    if (!ctx.isInitialized()) {
        throw std::runtime_error("GPU context not initialized");
    }
    
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());
    int k = std::min(m, n);
    
    if (max_rank > 0 && max_rank < k) {
        k = max_rank;
    }
    
    SVDResult<float> result;
    
    // Allocate device memory
    DevicePtr<float> d_A = allocateDevice<float>(m * n);
    DevicePtr<float> d_S = allocateDevice<float>(k);
    DevicePtr<float> d_U = allocateDevice<float>(m * k);
    DevicePtr<float> d_Vt = allocateDevice<float>(k * n);
    DevicePtr<int> d_info = allocateDevice<int>(1);
    
    GPU_CHECK(cudaMemcpy(d_A.get(), A.data(), m * n * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    int lwork;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(
        ctx.cusolverHandle(), m, n, &lwork));
    
    DevicePtr<float> d_work = allocateDevice<float>(lwork);
    DevicePtr<float> d_rwork = allocateDevice<float>(std::min(m, n) - 1);
    
    signed char jobu = 'S';
    signed char jobvt = 'S';
    
    CUSOLVER_CHECK(cusolverDnSgesvd(
        ctx.cusolverHandle(),
        jobu, jobvt,
        m, n,
        d_A.get(), m,
        d_S.get(),
        d_U.get(), m,
        d_Vt.get(), k,
        d_work.get(), lwork,
        d_rwork.get(),
        d_info.get()
    ));
    
    synchronize();
    
    int info;
    copyToHost(&info, d_info.get(), 1);
    if (info != 0) {
        throw std::runtime_error("cusolverDnSgesvd failed");
    }
    
    // Copy and truncate (similar to double version)
    std::vector<float> S(k);
    copyToHost(S.data(), d_S.get(), k);
    
    float total_weight = 0.0f;
    for (int i = 0; i < k; ++i) {
        total_weight += S[i] * S[i];
    }
    
    int new_rank = k;
    float truncation_weight = 0.0f;
    
    for (int i = k - 1; i >= 0; --i) {
        float relative_error = std::sqrt(truncation_weight / total_weight);
        if (S[i] > cutoff && relative_error < cutoff) {
            new_rank = i + 1;
            break;
        }
        truncation_weight += S[i] * S[i];
        new_rank = i;
    }
    
    if (new_rank == 0) new_rank = 1;
    if (max_rank > 0 && new_rank > max_rank) new_rank = max_rank;
    
    result.rank = new_rank;
    result.S.assign(S.begin(), S.begin() + new_rank);
    result.truncation_error = 0.0f;
    for (int i = new_rank; i < k; ++i) {
        result.truncation_error += S[i] * S[i];
    }
    result.truncation_error = std::sqrt(result.truncation_error);
    
    result.U = GPUMatrix<float>(m, new_rank);
    result.Vt = GPUMatrix<float>(new_rank, n);
    
    for (int j = 0; j < new_rank; ++j) {
        GPU_CHECK(cudaMemcpy(result.U.data() + j * m,
                             d_U.get() + j * m,
                             m * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
    
    for (int i = 0; i < new_rank; ++i) {
        GPU_CHECK(cudaMemcpy(result.Vt.data() + i * n,
                             d_Vt.get() + i * n,
                             n * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
    
    return result;
}

#endif // IDMRG_GPU_BACKEND_CUDA

#ifdef IDMRG_GPU_BACKEND_HIP

// HIP/rocSOLVER implementation
template<typename T>
SVDResult<T> svd(const GPUMatrix<T>& A, 
                 T cutoff = 1e-14, 
                 int max_rank = -1);

template<>
inline SVDResult<double> svd(const GPUMatrix<double>& A,
                             double cutoff,
                             int max_rank)
{
    auto& ctx = GPUContext::instance();
    if (!ctx.isInitialized()) {
        throw std::runtime_error("GPU context not initialized");
    }
    
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());
    int k = std::min(m, n);
    
    if (max_rank > 0 && max_rank < k) k = max_rank;
    
    SVDResult<double> result;
    
    DevicePtr<double> d_A = allocateDevice<double>(m * n);
    DevicePtr<double> d_S = allocateDevice<double>(k);
    DevicePtr<double> d_U = allocateDevice<double>(m * k);
    DevicePtr<double> d_Vt = allocateDevice<double>(k * n);
    DevicePtr<double> d_E = allocateDevice<double>(k - 1);  // For bidiagonal
    DevicePtr<int> d_info = allocateDevice<int>(1);
    
    hipMemcpy(d_A.get(), A.data(), m * n * sizeof(double), hipMemcpyDeviceToDevice);
    
    // rocSOLVER SVD
    rocsolver_dgesvd(
        ctx.rocsolverHandle(),
        rocblas_svect_all,  // Compute all of U
        rocblas_svect_all,  // Compute all of Vt
        m, n,
        d_A.get(), m,
        d_S.get(),
        d_U.get(), m,
        d_Vt.get(), k,
        d_E.get(),
        rocblas_outofplace,
        d_info.get()
    );
    
    synchronize();
    
    // Similar truncation logic as CUDA version...
    std::vector<double> S(k);
    copyToHost(S.data(), d_S.get(), k);
    
    double total_weight = 0.0;
    for (int i = 0; i < k; ++i) total_weight += S[i] * S[i];
    
    int new_rank = k;
    double truncation_weight = 0.0;
    
    for (int i = k - 1; i >= 0; --i) {
        double rel_err = std::sqrt(truncation_weight / total_weight);
        if (S[i] > cutoff && rel_err < cutoff) {
            new_rank = i + 1;
            break;
        }
        truncation_weight += S[i] * S[i];
        new_rank = i;
    }
    
    if (new_rank == 0) new_rank = 1;
    if (max_rank > 0 && new_rank > max_rank) new_rank = max_rank;
    
    result.rank = new_rank;
    result.S.assign(S.begin(), S.begin() + new_rank);
    result.truncation_error = std::sqrt(truncation_weight);
    
    result.U = GPUMatrix<double>(m, new_rank);
    result.Vt = GPUMatrix<double>(new_rank, n);
    
    for (int j = 0; j < new_rank; ++j) {
        hipMemcpy(result.U.data() + j * m, d_U.get() + j * m,
                  m * sizeof(double), hipMemcpyDeviceToDevice);
    }
    for (int i = 0; i < new_rank; ++i) {
        hipMemcpy(result.Vt.data() + i * n, d_Vt.get() + i * n,
                  n * sizeof(double), hipMemcpyDeviceToDevice);
    }
    
    return result;
}

#endif // IDMRG_GPU_BACKEND_HIP

#endif // IDMRG_GPU_ENABLED

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_GPU_SVD_H
