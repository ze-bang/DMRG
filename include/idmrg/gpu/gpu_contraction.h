//
// GPU Tensor Contraction
// Implements tensor contractions using cuBLAS (CUDA) or rocBLAS (HIP)
//
// For DMRG, the main operations are:
// 1. Matrix-matrix multiplication (GEMM) for tensor contractions
// 2. Matrix-vector multiplication (GEMV) for applying operators
//

#ifndef IDMRG_GPU_CONTRACTION_H
#define IDMRG_GPU_CONTRACTION_H

#include "gpu_backend.h"
#include "gpu_tensor.h"
#include <vector>
#include <stdexcept>

namespace idmrg {
namespace gpu {

#if IDMRG_GPU_ENABLED

//=============================================================================
// BLAS Operations on GPU
//=============================================================================

// Matrix multiplication: C = alpha * op(A) * op(B) + beta * C
// op(X) = X, X^T, or X^H depending on trans parameter

enum class Transpose { NoTrans, Trans, ConjTrans };

#ifdef IDMRG_GPU_BACKEND_CUDA

inline cublasOperation_t toCublasOp(Transpose t) {
    switch (t) {
        case Transpose::NoTrans: return CUBLAS_OP_N;
        case Transpose::Trans: return CUBLAS_OP_T;
        case Transpose::ConjTrans: return CUBLAS_OP_C;
    }
    return CUBLAS_OP_N;
}

// Double precision GEMM
inline void gemm(Transpose transA, Transpose transB,
                 int m, int n, int k,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    CUBLAS_CHECK(cublasDgemm(
        ctx.cublasHandle(),
        toCublasOp(transA), toCublasOp(transB),
        m, n, k,
        &alpha,
        A, lda,
        B, ldb,
        &beta,
        C, ldc
    ));
}

// Single precision GEMM
inline void gemm(Transpose transA, Transpose transB,
                 int m, int n, int k,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    CUBLAS_CHECK(cublasSgemm(
        ctx.cublasHandle(),
        toCublasOp(transA), toCublasOp(transB),
        m, n, k,
        &alpha,
        A, lda,
        B, ldb,
        &beta,
        C, ldc
    ));
}

// Double precision GEMV: y = alpha * op(A) * x + beta * y
inline void gemv(Transpose transA,
                 int m, int n,
                 double alpha,
                 const double* A, int lda,
                 const double* x, int incx,
                 double beta,
                 double* y, int incy)
{
    auto& ctx = GPUContext::instance();
    CUBLAS_CHECK(cublasDgemv(
        ctx.cublasHandle(),
        toCublasOp(transA),
        m, n,
        &alpha,
        A, lda,
        x, incx,
        &beta,
        y, incy
    ));
}

// Batched GEMM for multiple small matrix multiplications
inline void gemmBatched(Transpose transA, Transpose transB,
                        int m, int n, int k,
                        double alpha,
                        const double** A_array, int lda,
                        const double** B_array, int ldb,
                        double beta,
                        double** C_array, int ldc,
                        int batchCount)
{
    auto& ctx = GPUContext::instance();
    CUBLAS_CHECK(cublasDgemmBatched(
        ctx.cublasHandle(),
        toCublasOp(transA), toCublasOp(transB),
        m, n, k,
        &alpha,
        A_array, lda,
        B_array, ldb,
        &beta,
        C_array, ldc,
        batchCount
    ));
}

// Strided batched GEMM (more efficient for uniform strides)
inline void gemmStridedBatched(Transpose transA, Transpose transB,
                               int m, int n, int k,
                               double alpha,
                               const double* A, int lda, long long strideA,
                               const double* B, int ldb, long long strideB,
                               double beta,
                               double* C, int ldc, long long strideC,
                               int batchCount)
{
    auto& ctx = GPUContext::instance();
    CUBLAS_CHECK(cublasDgemmStridedBatched(
        ctx.cublasHandle(),
        toCublasOp(transA), toCublasOp(transB),
        m, n, k,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C, ldc, strideC,
        batchCount
    ));
}

#endif // IDMRG_GPU_BACKEND_CUDA

#ifdef IDMRG_GPU_BACKEND_HIP

inline rocblas_operation toRocblasOp(Transpose t) {
    switch (t) {
        case Transpose::NoTrans: return rocblas_operation_none;
        case Transpose::Trans: return rocblas_operation_transpose;
        case Transpose::ConjTrans: return rocblas_operation_conjugate_transpose;
    }
    return rocblas_operation_none;
}

inline void gemm(Transpose transA, Transpose transB,
                 int m, int n, int k,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    rocblas_dgemm(
        ctx.rocblasHandle(),
        toRocblasOp(transA), toRocblasOp(transB),
        m, n, k,
        &alpha,
        A, lda,
        B, ldb,
        &beta,
        C, ldc
    );
}

inline void gemm(Transpose transA, Transpose transB,
                 int m, int n, int k,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    rocblas_sgemm(
        ctx.rocblasHandle(),
        toRocblasOp(transA), toRocblasOp(transB),
        m, n, k,
        &alpha,
        A, lda,
        B, ldb,
        &beta,
        C, ldc
    );
}

inline void gemmStridedBatched(Transpose transA, Transpose transB,
                               int m, int n, int k,
                               double alpha,
                               const double* A, int lda, long long strideA,
                               const double* B, int ldb, long long strideB,
                               double beta,
                               double* C, int ldc, long long strideC,
                               int batchCount)
{
    auto& ctx = GPUContext::instance();
    rocblas_dgemm_strided_batched(
        ctx.rocblasHandle(),
        toRocblasOp(transA), toRocblasOp(transB),
        m, n, k,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C, ldc, strideC,
        batchCount
    );
}

#endif // IDMRG_GPU_BACKEND_HIP

//=============================================================================
// High-level Tensor Contraction Interface
//=============================================================================

// Contract two matrices: C = A * B
template<typename T>
GPUMatrix<T> contract(const GPUMatrix<T>& A, const GPUMatrix<T>& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix dimension mismatch in contraction");
    }
    
    GPUMatrix<T> C(A.rows(), B.cols());
    
    gemm(Transpose::NoTrans, Transpose::NoTrans,
         static_cast<int>(A.rows()), 
         static_cast<int>(B.cols()), 
         static_cast<int>(A.cols()),
         T(1.0),
         A.data(), static_cast<int>(A.lda()),
         B.data(), static_cast<int>(B.lda()),
         T(0.0),
         C.data(), static_cast<int>(C.lda()));
    
    return C;
}

// Contract with transpose: C = A^T * B
template<typename T>
GPUMatrix<T> contractTransA(const GPUMatrix<T>& A, const GPUMatrix<T>& B) {
    if (A.rows() != B.rows()) {
        throw std::runtime_error("Matrix dimension mismatch in contraction");
    }
    
    GPUMatrix<T> C(A.cols(), B.cols());
    
    gemm(Transpose::Trans, Transpose::NoTrans,
         static_cast<int>(A.cols()),
         static_cast<int>(B.cols()),
         static_cast<int>(A.rows()),
         T(1.0),
         A.data(), static_cast<int>(A.lda()),
         B.data(), static_cast<int>(B.lda()),
         T(0.0),
         C.data(), static_cast<int>(C.lda()));
    
    return C;
}

// Contract: C = A * B^T
template<typename T>
GPUMatrix<T> contractTransB(const GPUMatrix<T>& A, const GPUMatrix<T>& B) {
    if (A.cols() != B.cols()) {
        throw std::runtime_error("Matrix dimension mismatch in contraction");
    }
    
    GPUMatrix<T> C(A.rows(), B.rows());
    
    gemm(Transpose::NoTrans, Transpose::Trans,
         static_cast<int>(A.rows()),
         static_cast<int>(B.rows()),
         static_cast<int>(A.cols()),
         T(1.0),
         A.data(), static_cast<int>(A.lda()),
         B.data(), static_cast<int>(B.lda()),
         T(0.0),
         C.data(), static_cast<int>(C.lda()));
    
    return C;
}

//=============================================================================
// Tensor Network Specific Operations
//=============================================================================

// MPS-MPO contraction helper
// Contracts: result[a',s',b'] = sum_s A[a,s,b] * W[a',s',s,b'] 
// This is reshaped to matrix multiplication

template<typename T>
class TensorContractor {
public:
    TensorContractor() = default;
    
    // Contract MPS tensor with MPO tensor
    // A: (chi_l, d, chi_r) - MPS tensor
    // W: (w_l, d', d, w_r) - MPO tensor
    // Result: (chi_l * w_l, d', chi_r * w_r)
    GPUTensor<T> contractMPSMPO(const GPUTensor<T>& A,
                                 const GPUTensor<T>& W) {
        // Reshape for efficient GEMM
        // This is a simplified version; real implementation would handle
        // index permutations and fusions
        
        if (A.rank() != 3 || W.rank() != 4) {
            throw std::runtime_error("Invalid tensor ranks for MPS-MPO contraction");
        }
        
        size_t chi_l = A.dim(0);
        size_t d = A.dim(1);
        size_t chi_r = A.dim(2);
        
        size_t w_l = W.dim(0);
        size_t d_prime = W.dim(1);
        size_t d_w = W.dim(2);
        size_t w_r = W.dim(3);
        
        if (d != d_w) {
            throw std::runtime_error("Physical dimension mismatch");
        }
        
        // Result dimensions
        std::vector<size_t> result_dims = {chi_l * w_l, d_prime, chi_r * w_r};
        GPUTensor<T> result(result_dims);
        
        // Implementation would involve:
        // 1. Reshape A to (chi_l * d, chi_r)
        // 2. Reshape W to (w_l * d_prime, d * w_r)
        // 3. Perform contractions
        // 4. Reshape result
        
        // For now, placeholder
        return result;
    }
    
    // Environment update: L' = sum_s L[a,a'] * A[a,s,b] * conj(A[a',s,b'])
    // This is the expensive step in DMRG
    GPUMatrix<T> updateLeftEnvironment(const GPUMatrix<T>& L,
                                        const GPUTensor<T>& A) {
        // L: (chi, chi) - left environment
        // A: (chi, d, chi') - MPS tensor
        
        size_t chi = A.dim(0);
        size_t d = A.dim(1);
        size_t chi_new = A.dim(2);
        
        // Reshape A to (chi, d * chi')
        GPUMatrix<T> A_mat(chi, d * chi_new, A.data());
        
        // Compute L * A
        auto LA = contract(L, A_mat);  // (chi, d * chi')
        
        // Need to contract with A^*
        // Full implementation would handle complex conjugation
        
        auto result = contractTransB(LA, A_mat);  // (chi', chi')
        
        return result;
    }
};

#endif // IDMRG_GPU_ENABLED

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_GPU_CONTRACTION_H
