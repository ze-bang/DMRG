//
// GPU Backend Selection and Common Definitions
// Supports CUDA (NVIDIA) and HIP (AMD ROCm)
//

#ifndef IDMRG_GPU_BACKEND_H
#define IDMRG_GPU_BACKEND_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <complex>

//=============================================================================
// Backend Detection - Include GPU headers BEFORE namespace
//=============================================================================

#if defined(IDMRG_USE_CUDA)
    #define IDMRG_GPU_ENABLED 1
    #define IDMRG_GPU_BACKEND_CUDA 1
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cusolverDn.h>
#elif defined(IDMRG_USE_HIP)
    #define IDMRG_GPU_ENABLED 1
    #define IDMRG_GPU_BACKEND_HIP 1
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
    #include <rocsolver/rocsolver.h>
#else
    #define IDMRG_GPU_ENABLED 0
#endif

namespace idmrg {
namespace gpu {

//=============================================================================
// Error Checking Macros
//=============================================================================

#if defined(IDMRG_GPU_BACKEND_CUDA)
    #define GPU_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error(std::string("CUDA error: ") + \
                    cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                    std::to_string(__LINE__)); \
            } \
        } while(0)
    
    #define CUBLAS_CHECK(call) \
        do { \
            cublasStatus_t status = call; \
            if (status != CUBLAS_STATUS_SUCCESS) { \
                throw std::runtime_error("cuBLAS error at " + \
                    std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
            } \
        } while(0)
    
    #define CUSOLVER_CHECK(call) \
        do { \
            cusolverStatus_t status = call; \
            if (status != CUSOLVER_STATUS_SUCCESS) { \
                throw std::runtime_error("cuSOLVER error at " + \
                    std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
            } \
        } while(0)

#elif defined(IDMRG_GPU_BACKEND_HIP)
    #define GPU_CHECK(call) \
        do { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                throw std::runtime_error(std::string("HIP error: ") + \
                    hipGetErrorString(err) + " at " + __FILE__ + ":" + \
                    std::to_string(__LINE__)); \
            } \
        } while(0)
#endif

//=============================================================================
// GPU Memory Management
//=============================================================================

#if IDMRG_GPU_ENABLED

// Device memory deleter for smart pointers
struct DeviceDeleter {
    void operator()(void* ptr) const {
#if defined(IDMRG_GPU_BACKEND_CUDA)
        if (ptr) cudaFree(ptr);
#elif defined(IDMRG_GPU_BACKEND_HIP)
        if (ptr) hipFree(ptr);
#endif
    }
};

template<typename T>
using DevicePtr = std::unique_ptr<T[], DeviceDeleter>;

// Allocate device memory
template<typename T>
DevicePtr<T> allocateDevice(size_t count) {
    T* ptr = nullptr;
#if defined(IDMRG_GPU_BACKEND_CUDA)
    GPU_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
#elif defined(IDMRG_GPU_BACKEND_HIP)
    GPU_CHECK(hipMalloc(&ptr, count * sizeof(T)));
#endif
    return DevicePtr<T>(ptr);
}

// Copy host to device
template<typename T>
void copyToDevice(const T* h_ptr, T* d_ptr, size_t count) {
#if defined(IDMRG_GPU_BACKEND_CUDA)
    GPU_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
#elif defined(IDMRG_GPU_BACKEND_HIP)
    GPU_CHECK(hipMemcpy(d_ptr, h_ptr, count * sizeof(T), hipMemcpyHostToDevice));
#endif
}

// Copy device to host
template<typename T>
void copyToHost(T* h_ptr, const T* d_ptr, size_t count) {
#if defined(IDMRG_GPU_BACKEND_CUDA)
    GPU_CHECK(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
#elif defined(IDMRG_GPU_BACKEND_HIP)
    GPU_CHECK(hipMemcpy(h_ptr, d_ptr, count * sizeof(T), hipMemcpyDeviceToHost));
#endif
}

// Synchronize device
inline void synchronize() {
#if defined(IDMRG_GPU_BACKEND_CUDA)
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(IDMRG_GPU_BACKEND_HIP)
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

#endif // IDMRG_GPU_ENABLED

//=============================================================================
// GPU Context Singleton
//=============================================================================

#if IDMRG_GPU_ENABLED

class GPUContext {
public:
    static GPUContext& instance() {
        static GPUContext ctx;
        return ctx;
    }
    
    ~GPUContext() {
        cleanup();
    }
    
    bool isInitialized() const { return initialized_; }
    
    void initialize(int device_id = 0) {
        if (initialized_) return;
        
#ifdef IDMRG_GPU_BACKEND_CUDA
        GPU_CHECK(cudaSetDevice(device_id));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
#elif defined(IDMRG_GPU_BACKEND_HIP)
        GPU_CHECK(hipSetDevice(device_id));
        rocblas_create_handle(&rocblas_handle_);
        rocsolver_create_handle(&rocsolver_handle_);
#endif
        device_id_ = device_id;
        initialized_ = true;
    }
    
    void cleanup() {
        if (!initialized_) return;
        
#ifdef IDMRG_GPU_BACKEND_CUDA
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        if (cusolver_handle_) {
            cusolverDnDestroy(cusolver_handle_);
            cusolver_handle_ = nullptr;
        }
#elif defined(IDMRG_GPU_BACKEND_HIP)
        if (rocblas_handle_) {
            rocblas_destroy_handle(rocblas_handle_);
            rocblas_handle_ = nullptr;
        }
        if (rocsolver_handle_) {
            rocsolver_destroy_handle(rocsolver_handle_);
            rocsolver_handle_ = nullptr;
        }
#endif
        initialized_ = false;
    }
    
#ifdef IDMRG_GPU_BACKEND_CUDA
    cublasHandle_t cublasHandle() { return cublas_handle_; }
    cusolverDnHandle_t cusolverHandle() { return cusolver_handle_; }
#elif defined(IDMRG_GPU_BACKEND_HIP)
    rocblas_handle rocblasHandle() { return rocblas_handle_; }
    rocsolver_handle rocsolverHandle() { return rocsolver_handle_; }
#endif
    
    int deviceId() const { return device_id_; }
    
    // Get device properties
    std::string deviceName() const {
#ifdef IDMRG_GPU_BACKEND_CUDA
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id_);
        return prop.name;
#elif defined(IDMRG_GPU_BACKEND_HIP)
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device_id_);
        return prop.name;
#endif
    }
    
    size_t totalMemory() const {
#ifdef IDMRG_GPU_BACKEND_CUDA
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id_);
        return prop.totalGlobalMem;
#elif defined(IDMRG_GPU_BACKEND_HIP)
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device_id_);
        return prop.totalGlobalMem;
#endif
    }
    
    size_t freeMemory() const {
        size_t free_mem, total_mem;
#ifdef IDMRG_GPU_BACKEND_CUDA
        cudaMemGetInfo(&free_mem, &total_mem);
#elif defined(IDMRG_GPU_BACKEND_HIP)
        hipMemGetInfo(&free_mem, &total_mem);
#endif
        return free_mem;
    }

private:
    GPUContext() = default;
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;
    
    bool initialized_ = false;
    int device_id_ = 0;
    
#ifdef IDMRG_GPU_BACKEND_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
#elif defined(IDMRG_GPU_BACKEND_HIP)
    rocblas_handle rocblas_handle_ = nullptr;
    rocsolver_handle rocsolver_handle_ = nullptr;
#endif
};

// RAII initializer
class GPUInitializer {
public:
    GPUInitializer(int device_id = 0) {
        GPUContext::instance().initialize(device_id);
    }
};

#endif // IDMRG_GPU_ENABLED

//=============================================================================
// Compile-time GPU availability check
//=============================================================================

constexpr bool gpu_enabled() {
#if IDMRG_GPU_ENABLED
    return true;
#else
    return false;
#endif
}

inline std::string gpu_backend_name() {
#if defined(IDMRG_GPU_BACKEND_CUDA)
    return "CUDA";
#elif defined(IDMRG_GPU_BACKEND_HIP)
    return "HIP/ROCm";
#else
    return "None";
#endif
}

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_GPU_BACKEND_H
