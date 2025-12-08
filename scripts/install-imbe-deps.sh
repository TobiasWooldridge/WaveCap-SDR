#!/bin/bash
# Install IMBE codec dependencies for P25 voice decoding
# Requires: cmake, build tools (make, gcc/clang)
#
# This script installs:
#   1. mbelib - IMBE/AMBE vocoder library
#   2. dsd-fme - Digital Speech Decoder (Florida Man Edition)
#
# Usage:
#   ./scripts/install-imbe-deps.sh
#
# After installation, WaveCap-SDR will automatically detect DSD-FME
# and enable P25 voice decoding.

set -e

echo "=== IMBE Codec Dependencies Installer ==="
echo ""

# Check for required tools
check_deps() {
    local missing=""
    for cmd in cmake make git; do
        if ! command -v $cmd &> /dev/null; then
            missing="$missing $cmd"
        fi
    done

    if [ -n "$missing" ]; then
        echo "ERROR: Missing required tools:$missing"
        echo ""
        echo "Install them first:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  brew install cmake git"
        else
            echo "  sudo apt-get install cmake build-essential git"
        fi
        exit 1
    fi
}

# Create build directory
BUILD_DIR="${TMPDIR:-/tmp}/imbe-build-$$"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "Building in: $BUILD_DIR"
echo ""

# Check dependencies
echo "Checking dependencies..."
check_deps
echo "OK"
echo ""

# Build and install mbelib
echo "=== Building mbelib (IMBE/AMBE vocoder) ==="
if command -v dsd-fme &> /dev/null; then
    echo "mbelib appears to be installed (dsd-fme found)"
    echo "Skipping mbelib build..."
else
    git clone https://github.com/szechyjs/mbelib.git
    cd mbelib
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    echo ""
    echo "Installing mbelib (requires sudo)..."
    sudo make install
    cd "$BUILD_DIR"
    echo "mbelib installed successfully"
fi
echo ""

# Build and install DSD-FME
echo "=== Building DSD-FME (Digital Speech Decoder) ==="
if command -v dsd-fme &> /dev/null; then
    echo "dsd-fme is already installed:"
    which dsd-fme
    dsd-fme --version 2>/dev/null || true
else
    git clone https://github.com/lwvmobile/dsd-fme.git
    cd dsd-fme
    mkdir -p build && cd build

    # On macOS, we may need to specify library paths
    if [[ "$OSTYPE" == "darwin"* ]]; then
        cmake -DCMAKE_PREFIX_PATH=/usr/local ..
    else
        cmake ..
    fi

    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    echo ""
    echo "Installing DSD-FME (requires sudo)..."
    sudo make install
    cd "$BUILD_DIR"
fi
echo ""

# Update library cache on Linux
if [[ "$OSTYPE" == "linux"* ]]; then
    echo "Updating library cache..."
    sudo ldconfig
fi

# Verify installation
echo "=== Verifying Installation ==="
if command -v dsd-fme &> /dev/null; then
    echo "SUCCESS: dsd-fme is available"
    echo "  Location: $(which dsd-fme)"
    dsd-fme --version 2>/dev/null || true
else
    echo "WARNING: dsd-fme not found in PATH"
    echo "You may need to add /usr/local/bin to your PATH"
fi
echo ""

# Cleanup
echo "Cleaning up build directory..."
rm -rf "$BUILD_DIR"
echo ""

echo "=== Installation Complete ==="
echo ""
echo "WaveCap-SDR will now use DSD-FME for P25 IMBE voice decoding."
echo "Create a P25 channel and tune to a P25 system to test."
echo ""
echo "Note: Patent notice from mbelib applies - see:"
echo "  https://github.com/szechyjs/mbelib/blob/master/README.md"
