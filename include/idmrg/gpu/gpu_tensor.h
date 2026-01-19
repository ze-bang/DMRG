//
// GPU Tensor Class
// Wrapper for dense tensors on GPU with automatic memory management
//

#ifndef IDMRG_GPU_TENSOR_H
#define IDMRG_GPU_TENSOR_H

#include "gpu_backend.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace idmrg {
namespace gpu {

#if IDMRG_GPU_ENABLED

//=============================================================================
// GPU Dense Tensor
//=============================================================================

template<typename T = double>
class GPUTensor {
public:
    using value_type = T;
    
    // Default constructor (empty tensor)
    GPUTensor() : data_(nullptr), size_(0) {}
    
    // Constructor with dimensions
    explicit GPUTensor(const std::vector<size_t>& dims) 
        : dims_(dims) 
    {
        size_ = std::accumulate(dims.begin(), dims.end(), 
                                size_t(1), std::multiplies<size_t>());
        if (size_ > 0) {
            data_ = allocateDevice<T>(size_);
        }
    }
    
    // Constructor from host data
    GPUTensor(const std::vector<size_t>& dims, const T* host_data)
        : GPUTensor(dims)
    {
        if (size_ > 0 && host_data) {
            copyToDevice(host_data, data_.get(), size_);
        }
    }
    
    // Constructor from std::vector
    GPUTensor(const std::vector<size_t>& dims, const std::vector<T>& host_data)
        : GPUTensor(dims, host_data.data())
    {
        if (host_data.size() != size_) {
            throw std::runtime_error("Data size mismatch");
        }
    }
    
    // Move constructor
    GPUTensor(GPUTensor&& other) noexcept
        : data_(std::move(other.data_))
        , dims_(std::move(other.dims_))
        , size_(other.size_)
    {
        other.size_ = 0;
    }
    
    // Move assignment
    GPUTensor& operator=(GPUTensor&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            dims_ = std::move(other.dims_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Copy constructor
    GPUTensor(const GPUTensor& other) 
        : dims_(other.dims_), size_(other.size_)
    {
        if (size_ > 0) {
            data_ = allocateDevice<T>(size_);
#ifdef IDMRG_GPU_BACKEND_CUDA
            GPU_CHECK(cudaMemcpy(data_.get(), other.data_.get(), 
                                 size_ * sizeof(T), cudaMemcpyDeviceToDevice));
#elif defined(IDMRG_GPU_BACKEND_HIP)
            GPU_CHECK(hipMemcpy(data_.get(), other.data_.get(), 
                               size_ * sizeof(T), hipMemcpyDeviceToDevice));
#endif
        }
    }
    
    // Copy assignment
    GPUTensor& operator=(const GPUTensor& other) {
        if (this != &other) {
            GPUTensor tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }
    
    // Accessors
    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }
    
    size_t size() const { return size_; }
    size_t rank() const { return dims_.size(); }
    const std::vector<size_t>& dims() const { return dims_; }
    size_t dim(size_t i) const { return dims_.at(i); }
    
    bool empty() const { return size_ == 0; }
    
    // Copy to host
    std::vector<T> toHost() const {
        std::vector<T> result(size_);
        if (size_ > 0) {
            copyToHost(result.data(), data_.get(), size_);
        }
        return result;
    }
    
    // Copy from host
    void fromHost(const T* host_data) {
        if (size_ > 0) {
            copyToDevice(host_data, data_.get(), size_);
        }
    }
    
    void fromHost(const std::vector<T>& host_data) {
        if (host_data.size() != size_) {
            throw std::runtime_error("Size mismatch in fromHost");
        }
        fromHost(host_data.data());
    }
    
    // Reshape (no data copy, just reinterpret dimensions)
    void reshape(const std::vector<size_t>& new_dims) {
        size_t new_size = std::accumulate(new_dims.begin(), new_dims.end(),
                                          size_t(1), std::multiplies<size_t>());
        if (new_size != size_) {
            throw std::runtime_error("Cannot reshape: size mismatch");
        }
        dims_ = new_dims;
    }
    
    // Memory info
    size_t memoryBytes() const { return size_ * sizeof(T); }
    
    // Print info
    void printInfo(const std::string& name = "GPUTensor") const {
        std::cout << name << ": [";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0) std::cout << " x ";
            std::cout << dims_[i];
        }
        std::cout << "] = " << size_ << " elements, "
                  << memoryBytes() / 1024.0 / 1024.0 << " MB" << std::endl;
    }

private:
    DevicePtr<T> data_;
    std::vector<size_t> dims_;
    size_t size_;
};

// Type aliases
using GPUTensorD = GPUTensor<double>;
using GPUTensorF = GPUTensor<float>;
using GPUTensorZ = GPUTensor<std::complex<double>>;
using GPUTensorC = GPUTensor<std::complex<float>>;

//=============================================================================
// GPU Matrix (2D Tensor) - Convenience wrapper for BLAS operations
//=============================================================================

template<typename T = double>
class GPUMatrix {
public:
    using value_type = T;
    
    GPUMatrix() : rows_(0), cols_(0), lda_(0) {}
    
    GPUMatrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), lda_(rows)
        , tensor_({rows, cols})
    {}
    
    GPUMatrix(size_t rows, size_t cols, const T* host_data)
        : rows_(rows), cols_(cols), lda_(rows)
        , tensor_({rows, cols}, host_data)
    {}
    
    // Accessors
    T* data() { return tensor_.data(); }
    const T* data() const { return tensor_.data(); }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t lda() const { return lda_; }  // Leading dimension (for BLAS)
    size_t size() const { return tensor_.size(); }
    
    std::vector<T> toHost() const { return tensor_.toHost(); }
    void fromHost(const T* data) { tensor_.fromHost(data); }
    
private:
    size_t rows_, cols_, lda_;
    GPUTensor<T> tensor_;
};

using GPUMatrixD = GPUMatrix<double>;
using GPUMatrixF = GPUMatrix<float>;

#endif // IDMRG_GPU_ENABLED

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_GPU_TENSOR_H
