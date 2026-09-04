#!/usr/bin/env bash
# ==============================================================================
# NTRO-SRM: Automated Virtual Environment & Dependency Setup Script
# ==============================================================================
# Usage:
#   ./setup.sh              # Standard automated installation (creates venv + installs all)
#   ./setup.sh --no-test    # Fast installation (skips unit tests)
#   ./setup.sh --run        # Install and immediately launch the web application
#   ./setup.sh --clean      # Remove existing venv and perform fresh setup
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/venv"
RUN_APP=false
RUN_TESTS=true
CLEAN_SETUP=false

for arg in "$@"; do
    case $arg in
        --no-test)
            RUN_TESTS=false
            shift
            ;;
        --run)
            RUN_APP=true
            shift
            ;;
        --clean)
            CLEAN_SETUP=true
            shift
            ;;
        --help|-h)
            echo "NTRO-SRM Setup Utility"
            echo "Usage: ./setup.sh [OPTIONS]"
            echo "Options:"
            echo "  --no-test    Skip running unit & integration tests after setup"
            echo "  --run        Automatically launch the web app after installation"
            echo "  --clean      Remove existing venv and reinstall fresh"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
    esac
done

echo ""
echo "================================================================================"
echo "    _   _ _____ ____   ___        ____  ____  __  __ "
echo "   | \ | |_   _|  _ \ / _ \      / ___||  _ \|  \/  |"
echo "   |  \| | | | | |_) | | | |_____\___ \| |_) | |\/| |"
echo "   | |\  | | | |  _ <| |_| |_____|___) |  _ <| |  | |"
echo "   |_| \_| |_| |_| \_\\___/      |____/|_| \_\_|  |_|"
echo "   Sentinel-2 Multi-Spectral Super-Resolution Framework"
echo "================================================================================"
echo ""

if [ "$CLEAN_SETUP" = true ] && [ -d "${VENV_DIR}" ]; then
    echo "[!] Removing existing virtual environment (--clean specified)..."
    rm -rf "${VENV_DIR}"
fi

# 1. Check Python installation and version
echo "[1/7] Checking Python Environment..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd >/dev/null 2>&1; then
        PY_MAJOR=$($cmd -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
        PY_MINOR=$($cmd -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[!] Error: Python 3.10 or higher is required." >&2
    echo "    Please install Python 3.10+ (e.g., sudo apt install python3 python3-venv python3-pip)" >&2
    exit 1
fi
echo "      Found Python $($PYTHON_CMD --version 2>&1) at $(which $PYTHON_CMD)"

# 2. Hardware and GPU Inspection
echo "[2/7] Inspecting Compute Hardware..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    echo "      Detected NVIDIA GPU: ${GPU_NAME} (${GPU_MEM})"
    echo "      CUDA hardware acceleration will be enabled."
else
    echo "      No NVIDIA GPU detected. Framework will run in CPU fallback mode."
fi

# 3. Create or verify virtual environment
echo "[3/7] Setting up Virtual Environment..."
USE_SYSTEM_FLAG=""
if $PYTHON_CMD -c "import torch" >/dev/null 2>&1; then
    echo "      Host environment contains PyTorch. Enabling fast site-package reuse."
    USE_SYSTEM_FLAG="--system-site-packages"
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "      Creating virtual environment at ${VENV_DIR}..."
    $PYTHON_CMD -m venv ${USE_SYSTEM_FLAG} "${VENV_DIR}"
else
    echo "      Existing virtual environment detected at ${VENV_DIR}."
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"
echo "      Active Environment: ${VIRTUAL_ENV}"

# 4. Upgrade pip and build tools
echo "[4/7] Upgrading build tools..."
$VENV_PIP install --upgrade pip "setuptools<81.0.0" wheel --quiet

# 5. Install Dependencies and Package
echo "[5/7] Installing Dependencies & NTRO-SRM Package..."
$VENV_PIP install -r requirements.txt --quiet
$VENV_PIP install -e . --quiet

# Ensure third-party modules are initialized
echo "      Configuring third-party ESAOpenSR SEN2SR dependency..."
mkdir -p third_party

# If in a git repository and submodule is not yet checked out, initialize it
if [ -d ".git" ] && [ ! -d "third_party/SEN2SR/sen2sr" ]; then
    echo "      Initializing git submodule third_party/SEN2SR..."
    git submodule update --init --recursive third_party/SEN2SR 2>/dev/null || true
fi

# Fallback: if third_party/SEN2SR is still missing (e.g. downloaded as ZIP archive without .git)
if [ ! -d "third_party/SEN2SR/sen2sr" ]; then
    echo "      Cloning third-party ESAOpenSR SEN2SR repository..."
    rm -rf third_party/SEN2SR
    git clone --depth 1 https://github.com/ESAOpenSR/SEN2SR.git third_party/SEN2SR
fi

# Install SEN2SR into virtual environment in editable mode
if [ -d "third_party/SEN2SR" ]; then
    echo "      Installing SEN2SR package into virtual environment..."
    $VENV_PIP install -e third_party/SEN2SR --no-deps --quiet
fi

# Create required directories
mkdir -p outputs/web_jobs outputs/comparisons datasets/cache checkpoints/SEN2SRLite checkpoints/SEN2SR

# Copy .env.example to .env if not present
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "      Created default .env configuration from .env.example"
fi

# 6. Verify / Download Checkpoints
echo "[6/7] Verifying Pretrained Neural Weights..."
$VENV_PY scripts/download_checkpoints.py --model all

# 7. Verification Tests
if [ "$RUN_TESTS" = true ]; then
    echo "[7/7] Running Unit & Integration Tests..."
    if $VENV_PY -m pytest -q; then
        echo "      [✓] All tests passed successfully."
    else
        echo "      [!] Warning: Some tests had issues. Check logs above."
    fi
else
    echo "[7/7] Skipping test suite (--no-test specified)."
fi

echo ""
echo "================================================================================"
echo "    NTRO-SRM SETUP COMPLETED SUCCESSFULLY! "
echo "================================================================================"
echo ""
echo "To activate the environment in your shell:"
echo "    source venv/bin/activate"
echo ""
echo "To launch the Interactive Web Application:"
echo "    python scripts/run_web.py --port 8000"
echo "    -> Then open http://127.0.0.1:8000 in your browser"
echo ""
echo "To run super-resolution on a GeoTIFF via Command Line:"
echo "    python scripts/sr_sentinel2.py --input datasets/sample_s2/sample_s2_l2a.tif --output outputs/sr_2.5m.tif"
echo ""
echo "Available Models in UI and CLI:"
echo "    1. SEN2SR-Lite    : Fast CNN baseline (~0.47M params, ~2s)"
echo "    2. SEN2SR-Swin2SR : High-Quality Vision Transformer (~12.9M params)"
echo "================================================================================"
echo ""

if [ "$RUN_APP" = true ]; then
    echo "Launching Web Application on http://127.0.0.1:8000 ..."
    $VENV_PY scripts/run_web.py --port 8000
fi
