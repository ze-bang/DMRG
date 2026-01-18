#!/bin/bash
#
# Build script for iDMRG
# Supports AOCL (AMD), MKL (Intel), CUDA (NVIDIA GPU), and HIP (AMD GPU)
#

set -e

# Default ITensor directory
ITENSOR_DIR="${ITENSOR_DIR:-$HOME/software/itensor}"

# Parse arguments
BUILD_TYPE="Release"
BUILD_EXAMPLES="ON"
USE_MKL="OFF"
USE_AOCL="OFF"
USE_CUDA="OFF"
USE_HIP="OFF"
AOCL_ROOT="${AOCL_ROOT:-/opt/AMD/aocl}"
CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --mkl)
            USE_MKL="ON"
            USE_AOCL="OFF"
            shift
            ;;
        --aocl)
            USE_AOCL="ON"
            USE_MKL="OFF"
            shift
            ;;
        --aocl-root)
            AOCL_ROOT="$2"
            shift 2
            ;;
        --cuda)
            USE_CUDA="ON"
            USE_HIP="OFF"
            shift
            ;;
        --hip|--rocm)
            USE_HIP="ON"
            USE_CUDA="OFF"
            shift
            ;;
        --itensor)
            ITENSOR_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "CPU Backend Options:"
            echo "  --aocl          Use AMD AOCL (BLIS/libFLAME) for BLAS/LAPACK"
            echo "  --aocl-root DIR Set AOCL installation directory (default: /opt/AMD/aocl)"
            echo "  --mkl           Use Intel MKL for BLAS/LAPACK"
            echo ""
            echo "GPU Options:"
            echo "  --cuda          Enable NVIDIA CUDA GPU acceleration"
            echo "  --hip, --rocm   Enable AMD ROCm/HIP GPU acceleration"
            echo ""
            echo "General Options:"
            echo "  --debug         Build with debug symbols"
            echo "  --no-examples   Don't build example programs"
            echo "  --itensor DIR   Set ITensor directory"
            echo "  --clean         Clean build directory before building"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  ITENSOR_DIR     Path to ITensor library"
            echo "  AOCL_ROOT       Path to AOCL installation"
            echo ""
            echo "Examples:"
            echo "  $0 --aocl                    # Build with AMD AOCL"
            echo "  $0 --aocl --cuda             # AOCL + NVIDIA GPU"
            echo "  $0 --aocl --hip              # AOCL + AMD GPU"
            echo "  $0 --mkl --cuda              # Intel MKL + NVIDIA GPU"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "  iDMRG Build Configuration"
echo "================================================"
echo "  Build type:     $BUILD_TYPE"
echo "  ITensor:        $ITENSOR_DIR"
echo "  Examples:       $BUILD_EXAMPLES"
echo ""
echo "  --- CPU Backend ---"
if [ "$USE_AOCL" = "ON" ]; then
    echo "  BLAS:           AMD AOCL"
    echo "  AOCL root:      $AOCL_ROOT"
elif [ "$USE_MKL" = "ON" ]; then
    echo "  BLAS:           Intel MKL"
else
    echo "  BLAS:           System BLAS/LAPACK"
fi
echo ""
echo "  --- GPU Backend ---"
if [ "$USE_CUDA" = "ON" ]; then
    echo "  GPU:            NVIDIA CUDA"
elif [ "$USE_HIP" = "ON" ]; then
    echo "  GPU:            AMD ROCm/HIP"
else
    echo "  GPU:            Disabled"
fi
echo "================================================"
echo ""

# Check ITensor exists
if [ ! -d "$ITENSOR_DIR/itensor" ]; then
    echo "Warning: ITensor not found at $ITENSOR_DIR"
    echo "Please install ITensor and set ITENSOR_DIR environment variable"
    echo "or use: $0 --itensor /path/to/itensor"
    echo ""
fi

# Check AOCL if requested
if [ "$USE_AOCL" = "ON" ] && [ ! -d "$AOCL_ROOT" ]; then
    echo "Warning: AOCL not found at $AOCL_ROOT"
    echo "Please install AOCL or set correct path with --aocl-root"
    echo ""
    echo "To install AOCL:"
    echo "  1. Download from https://developer.amd.com/amd-aocl/"
    echo "  2. Extract to /opt/AMD/aocl or custom location"
    echo "  3. Run: source /opt/AMD/aocl/setenv_AOCL.sh"
    echo ""
fi

# Create build directory
BUILD_DIR="build"

if [ $CLEAN -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DITENSOR_DIR="$ITENSOR_DIR" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DUSE_MKL="$USE_MKL" \
    -DUSE_AOCL="$USE_AOCL" \
    -DAOCL_ROOT="$AOCL_ROOT" \
    -DUSE_CUDA="$USE_CUDA" \
    -DUSE_HIP="$USE_HIP"

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "================================================"
echo "  Build complete!"
echo "================================================"
echo ""

if [ "$BUILD_EXAMPLES" = "ON" ]; then
    echo "Example executables:"
    ls -la heisenberg_chain hubbard_2d generic_lattice 2>/dev/null || true
    echo ""
    echo "Run examples:"
    echo "  ./build/heisenberg_chain"
    echo "  ./build/hubbard_2d"
    echo "  ./build/generic_lattice"
    echo ""
    if [ "$USE_CUDA" = "ON" ] || [ "$USE_HIP" = "ON" ]; then
        echo "GPU acceleration is enabled!"
        echo "Monitor GPU usage with:"
        if [ "$USE_CUDA" = "ON" ]; then
            echo "  nvidia-smi"
        else
            echo "  rocm-smi"
        fi
    fi
fi
