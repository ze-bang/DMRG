cmake_minimum_required(VERSION 3.14)
project(iDMRG VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# Options
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_MKL "Use Intel MKL for BLAS/LAPACK" OFF)
option(USE_AOCL "Use AMD AOCL (BLIS/libFLAME) for BLAS/LAPACK" OFF)
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
option(USE_HIP "Enable AMD ROCm/HIP GPU acceleration" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" OFF)

# AOCL paths (AMD Optimizing CPU Libraries)
set(AOCL_ROOT "$ENV{AOCL_ROOT}" CACHE PATH "Path to AOCL installation")
if(NOT AOCL_ROOT)
    set(AOCL_ROOT "/opt/AMD/aocl" CACHE PATH "Path to AOCL installation" FORCE)
endif()

# Find ITensor
# Users should set ITENSOR_DIR to point to their ITensor installation
set(ITENSOR_DIR "$ENV{HOME}/software/itensor" CACHE PATH "Path to ITensor library")

if(NOT EXISTS "${ITENSOR_DIR}/itensor")
    message(WARNING "ITensor not found at ${ITENSOR_DIR}. Please set ITENSOR_DIR correctly.")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ITENSOR_DIR}
)

# Link directories
link_directories(${ITENSOR_DIR}/lib)

# Find OpenMP
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
    endif()
endif()

#=============================================================================
# BLAS/LAPACK Backend Selection
#=============================================================================

# Priority: AOCL > MKL > System BLAS
if(USE_AOCL)
    message(STATUS "Looking for AMD AOCL at: ${AOCL_ROOT}")
    
    # Find AOCL BLIS (BLAS) - handle various AOCL directory structures
    find_library(BLIS_LIBRARY
        NAMES blis blis-mt
        PATHS 
            ${AOCL_ROOT}/lib 
            ${AOCL_ROOT}/lib64
            ${AOCL_ROOT}/aocc/lib
            ${AOCL_ROOT}/aocc/lib64
            ${AOCL_ROOT}/aocc/lib_LP64
            ${AOCL_ROOT}/gcc/lib_LP64
        PATH_SUFFIXES LP64 ILP64
    )
    
    # Find AOCL libFLAME (LAPACK)
    find_library(FLAME_LIBRARY
        NAMES flame
        PATHS 
            ${AOCL_ROOT}/lib 
            ${AOCL_ROOT}/lib64
            ${AOCL_ROOT}/aocc/lib
            ${AOCL_ROOT}/aocc/lib64
            ${AOCL_ROOT}/aocc/lib_LP64
            ${AOCL_ROOT}/gcc/lib_LP64
        PATH_SUFFIXES LP64 ILP64
    )
    
    # Find AOCL includes
    find_path(AOCL_INCLUDE_DIR
        NAMES blis.h blis/blis.h
        PATHS 
            ${AOCL_ROOT}/include 
            ${AOCL_ROOT}/include/blis
            ${AOCL_ROOT}/aocc/include
            ${AOCL_ROOT}/aocc/include/blis
            ${AOCL_ROOT}/aocc/include_LP64
            ${AOCL_ROOT}/gcc/include_LP64
    )
    
    if(BLIS_LIBRARY AND FLAME_LIBRARY)
        message(STATUS "Found AOCL BLIS: ${BLIS_LIBRARY}")
        message(STATUS "Found AOCL libFLAME: ${FLAME_LIBRARY}")
        
        set(BLAS_LIBRARIES ${BLIS_LIBRARY})
        set(LAPACK_LIBRARIES ${FLAME_LIBRARY})
        
        if(AOCL_INCLUDE_DIR)
            include_directories(${AOCL_INCLUDE_DIR})
        endif()
        
        # AOCL requires additional libraries
        find_library(AOCL_UTILS_LIBRARY
            NAMES aoclutils
            PATHS 
                ${AOCL_ROOT}/lib 
                ${AOCL_ROOT}/lib64
                ${AOCL_ROOT}/aocc/lib
                ${AOCL_ROOT}/aocc/lib64
                ${AOCL_ROOT}/aocc/lib_LP64
                ${AOCL_ROOT}/gcc/lib_LP64
        )
        if(AOCL_UTILS_LIBRARY)
            list(APPEND BLAS_LIBRARIES ${AOCL_UTILS_LIBRARY})
        endif()
        
        add_compile_definitions(IDMRG_USE_AOCL)
        set(BLAS_VENDOR "AOCL")
    else()
        message(WARNING "AOCL not found at ${AOCL_ROOT}, falling back to system BLAS")
        set(USE_AOCL OFF)
    endif()
endif()

if(USE_MKL AND NOT USE_AOCL)
    set(BLA_VENDOR Intel10_64lp)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_compile_definitions(IDMRG_USE_MKL)
    set(BLAS_VENDOR "MKL")
endif()

if(NOT USE_AOCL AND NOT USE_MKL)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLAS_VENDOR "System")
endif()

#=============================================================================
# GPU Backend Selection
#=============================================================================

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
        
        add_compile_definitions(IDMRG_USE_CUDA)
        set(GPU_BACKEND "CUDA")
        
        # Set CUDA architecture (adjust for your GPU)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")
        endif()
        
        # CUDA libraries we need
        set(GPU_LIBRARIES 
            CUDA::cudart
            CUDA::cublas
            CUDA::cusolver
        )
        
        # Optional: cuTENSOR for optimized tensor contractions
        find_library(CUTENSOR_LIBRARY
            NAMES cutensor
            PATHS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64
        )
        if(CUTENSOR_LIBRARY)
            message(STATUS "Found cuTENSOR: ${CUTENSOR_LIBRARY}")
            list(APPEND GPU_LIBRARIES ${CUTENSOR_LIBRARY})
            add_compile_definitions(IDMRG_USE_CUTENSOR)
        endif()
    else()
        message(WARNING "CUDA requested but no CUDA compiler found")
        set(USE_CUDA OFF)
    endif()
endif()

if(USE_HIP)
    # AMD ROCm/HIP support
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip QUIET)
    find_package(rocblas QUIET)
    find_package(rocsolver QUIET)
    
    if(hip_FOUND AND rocblas_FOUND AND rocsolver_FOUND)
        message(STATUS "ROCm/HIP found")
        add_compile_definitions(IDMRG_USE_HIP)
        set(GPU_BACKEND "HIP")
        
        set(GPU_LIBRARIES 
            hip::host
            roc::rocblas
            roc::rocsolver
        )
    else()
        message(WARNING "HIP requested but ROCm not found")
        set(USE_HIP OFF)
    endif()
endif()

# Header-only library
add_library(idmrg INTERFACE)
target_include_directories(idmrg INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# GPU library (compiled if CUDA or HIP enabled)
if(USE_CUDA OR USE_HIP)
    if(USE_CUDA)
        add_library(idmrg_gpu 
            src/gpu/gpu_tensor.cu
            src/gpu/gpu_svd.cu
            src/gpu/gpu_contraction.cu
        )
        set_target_properties(idmrg_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_include_directories(idmrg_gpu PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(idmrg_gpu PUBLIC ${GPU_LIBRARIES})
    endif()
    
    if(USE_HIP)
        add_library(idmrg_gpu 
            src/gpu/gpu_tensor_hip.cpp
            src/gpu/gpu_svd_hip.cpp
            src/gpu/gpu_contraction_hip.cpp
        )
        target_include_directories(idmrg_gpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(idmrg_gpu PUBLIC ${GPU_LIBRARIES})
    endif()
endif()

# Helper function to link common libraries
function(link_idmrg_libraries target)
    target_link_libraries(${target} 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} OpenMP::OpenMP_CXX)
    endif()
    if(TARGET idmrg_gpu)
        target_link_libraries(${target} idmrg_gpu)
    endif()
endfunction()

# Create examples
if(BUILD_EXAMPLES)
    # 1D Heisenberg chain example
    add_executable(heisenberg_chain examples/heisenberg_chain.cpp)
    link_idmrg_libraries(heisenberg_chain)

    # 2D Hubbard model example
    add_executable(hubbard_2d examples/hubbard_2d.cpp)
    link_idmrg_libraries(hubbard_2d)

    # Generic lattice example
    add_executable(generic_lattice examples/generic_lattice.cpp)
    link_idmrg_libraries(generic_lattice)

    # Anisotropic triangular lattice example
    add_executable(anisotropic_triangular examples/anisotropic_triangular.cpp)
    link_idmrg_libraries(anisotropic_triangular)

    # Structure factor calculation example
    add_executable(structure_factor examples/structure_factor.cpp)
    link_idmrg_libraries(structure_factor)

    # Benchmark SSSF example
    add_executable(benchmark_sssf examples/benchmark_sssf.cpp)
    link_idmrg_libraries(benchmark_sssf)

    # J1-J2 Heisenberg model on square lattice benchmark
    add_executable(heisenberg_square_j1j2 examples/heisenberg_square_j1j2.cpp)
    link_idmrg_libraries(heisenberg_square_j1j2)

    # 2D Heisenberg on square lattice cylinder (finite DMRG)
    add_executable(heisenberg_square_finite examples/heisenberg_square_finite.cpp)
    link_idmrg_libraries(heisenberg_square_finite)
endif()

# Install
install(DIRECTORY include/ DESTINATION include)
install(TARGETS idmrg EXPORT idmrgTargets)
install(EXPORT idmrgTargets
    FILE idmrgTargets.cmake
    NAMESPACE idmrg::
    DESTINATION lib/cmake/idmrg
)

# Print configuration summary
message(STATUS "")
message(STATUS "============== iDMRG Configuration ==============")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "ITensor directory: ${ITENSOR_DIR}")
message(STATUS "")
message(STATUS "--- CPU Backend ---")
message(STATUS "OpenMP:            ${USE_OPENMP}")
message(STATUS "BLAS vendor:       ${BLAS_VENDOR}")
if(USE_AOCL)
message(STATUS "AOCL root:         ${AOCL_ROOT}")
message(STATUS "BLIS library:      ${BLIS_LIBRARY}")
message(STATUS "libFLAME library:  ${FLAME_LIBRARY}")
endif()
if(USE_MKL)
message(STATUS "Intel MKL:         ${USE_MKL}")
endif()
message(STATUS "")
message(STATUS "--- GPU Backend ---")
if(USE_CUDA)
message(STATUS "CUDA:              ON (${CUDAToolkit_VERSION})")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(CUTENSOR_LIBRARY)
message(STATUS "cuTENSOR:          ${CUTENSOR_LIBRARY}")
endif()
elseif(USE_HIP)
message(STATUS "ROCm/HIP:          ON")
else()
message(STATUS "GPU acceleration:  OFF")
endif()
message(STATUS "")
message(STATUS "Build examples:    ${BUILD_EXAMPLES}")
message(STATUS "=================================================")
message(STATUS "")
