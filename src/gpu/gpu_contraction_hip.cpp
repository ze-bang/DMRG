//
// GPU Tensor Contraction Implementation (HIP/ROCm)
//

#include "idmrg/gpu/gpu_contraction.h"

#ifdef IDMRG_USE_HIP

#include <hip/hip_runtime.h>

namespace idmrg {
namespace gpu {

//=============================================================================
// Element-wise operations
//=============================================================================

__global__ void scaleKernel(double* A, int n, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] *= alpha;
    }
}

__global__ void addKernel(
    const double* A, 
    const double* B, 
    double* C, 
    int n, 
    double alpha, 
    double beta) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = alpha * A[idx] + beta * B[idx];
    }
}

__global__ void hadamardKernel(
    const double* A, 
    const double* B, 
    double* C, 
    int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void permute3Kernel(
    const double* A,
    double* B,
    int dim0, int dim1, int dim2,
    int perm0, int perm1, int perm2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim0 * dim1 * dim2;
    
    if (idx < total) {
        int i = idx / (dim1 * dim2);
        int j = (idx / dim2) % dim1;
        int k = idx % dim2;
        
        int dims[3] = {dim0, dim1, dim2};
        int indices[3] = {i, j, k};
        int perm[3] = {perm0, perm1, perm2};
        
        int new_idx = 0;
        int stride = 1;
        for (int p = 2; p >= 0; --p) {
            new_idx += indices[perm[p]] * stride;
            stride *= dims[perm[p]];
        }
        
        B[new_idx] = A[idx];
    }
}

//=============================================================================
// Wrapper functions
//=============================================================================

void scaleTensor(double* A, int n, double alpha) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(scaleKernel, dim3(numBlocks), dim3(blockSize), 
                       0, 0, A, n, alpha);
}

void addTensors(const double* A, const double* B, double* C, 
                int n, double alpha, double beta) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(addKernel, dim3(numBlocks), dim3(blockSize), 
                       0, 0, A, B, C, n, alpha, beta);
}

void hadamardProduct(const double* A, const double* B, double* C, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hadamardKernel, dim3(numBlocks), dim3(blockSize), 
                       0, 0, A, B, C, n);
}

void permuteTensor3(const double* A, double* B,
                    int dim0, int dim1, int dim2,
                    int perm0, int perm1, int perm2) {
    int total = dim0 * dim1 * dim2;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(permute3Kernel, dim3(numBlocks), dim3(blockSize), 
                       0, 0, A, B, dim0, dim1, dim2, perm0, perm1, perm2);
}

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_USE_HIP
