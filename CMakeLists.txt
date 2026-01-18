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
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" OFF)

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

# Find BLAS/LAPACK
if(USE_MKL)
    set(BLA_VENDOR Intel10_64lp)
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Header-only library
add_library(idmrg INTERFACE)
target_include_directories(idmrg INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create examples
if(BUILD_EXAMPLES)
    # 1D Heisenberg chain example
    add_executable(heisenberg_chain examples/heisenberg_chain.cpp)
    target_link_libraries(heisenberg_chain 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(heisenberg_chain OpenMP::OpenMP_CXX)
    endif()

    # 2D Hubbard model example
    add_executable(hubbard_2d examples/hubbard_2d.cpp)
    target_link_libraries(hubbard_2d 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(hubbard_2d OpenMP::OpenMP_CXX)
    endif()

    # Generic lattice example
    add_executable(generic_lattice examples/generic_lattice.cpp)
    target_link_libraries(generic_lattice 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(generic_lattice OpenMP::OpenMP_CXX)
    endif()

    # Anisotropic triangular lattice example
    add_executable(anisotropic_triangular examples/anisotropic_triangular.cpp)
    target_link_libraries(anisotropic_triangular 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(anisotropic_triangular OpenMP::OpenMP_CXX)
    endif()

    # Structure factor calculation example
    add_executable(structure_factor examples/structure_factor.cpp)
    target_link_libraries(structure_factor 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(structure_factor OpenMP::OpenMP_CXX)
    endif()

    # Benchmark SSSF example
    add_executable(benchmark_sssf examples/benchmark_sssf.cpp)
    target_link_libraries(benchmark_sssf 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_sssf OpenMP::OpenMP_CXX)
    endif()

    # J1-J2 Heisenberg model on square lattice benchmark
    add_executable(heisenberg_square_j1j2 examples/heisenberg_square_j1j2.cpp)
    target_link_libraries(heisenberg_square_j1j2 
        idmrg 
        itensor 
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        pthread
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(heisenberg_square_j1j2 OpenMP::OpenMP_CXX)
    endif()
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
message(STATUS "========== iDMRG Configuration ==========")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "ITensor directory: ${ITENSOR_DIR}")
message(STATUS "OpenMP:            ${USE_OPENMP}")
message(STATUS "Intel MKL:         ${USE_MKL}")
message(STATUS "Build examples:    ${BUILD_EXAMPLES}")
message(STATUS "==========================================")
message(STATUS "")
