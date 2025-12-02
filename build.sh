#!/bin/bash
#
# Build script for iDMRG
#

set -e

# Default ITensor directory
ITENSOR_DIR="${ITENSOR_DIR:-$HOME/software/itensor}"

# Parse arguments
BUILD_TYPE="Release"
BUILD_EXAMPLES="ON"
USE_MKL="OFF"
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
            echo "Options:"
            echo "  --debug         Build with debug symbols"
            echo "  --no-examples   Don't build example programs"
            echo "  --mkl           Use Intel MKL for BLAS/LAPACK"
            echo "  --itensor DIR   Set ITensor directory"
            echo "  --clean         Clean build directory before building"
            echo "  -h, --help      Show this help"
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
echo "  Intel MKL:      $USE_MKL"
echo "================================================"
echo ""

# Check ITensor exists
if [ ! -d "$ITENSOR_DIR/itensor" ]; then
    echo "Warning: ITensor not found at $ITENSOR_DIR"
    echo "Please install ITensor and set ITENSOR_DIR environment variable"
    echo "or use: $0 --itensor /path/to/itensor"
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
    -DUSE_MKL="$USE_MKL"

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
fi
