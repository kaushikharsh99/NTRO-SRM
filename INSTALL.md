# NTRO-SRM Installation Guide

This document provides complete instructions for installing and configuring the **NTRO-SRM** (Sentinel-2 Multi-Spectral Super-Resolution Mapping) framework across different operating systems and hardware configurations.

---

## 1. System Requirements

### Hardware Requirements
| Component | Minimum | Recommended (Tested) |
| :--- | :--- | :--- |
| **Processor** | 4-Core x86_64 / ARM64 CPU | 8-Core Intel Core i5/i7 or AMD Ryzen |
| **RAM** | 8 GB System Memory | 16 GB+ System Memory |
| **GPU** | CPU execution supported | **NVIDIA GeForce RTX 3050 6GB** or higher with CUDA 11.8–13.x |
| **VRAM** | N/A (CPU mode) | 4 GB+ for SEN2SR-Lite; **6 GB+ for SEN2SR-Swin2SR** |
| **Disk Space** | 3 GB free disk space | 8 GB free disk space (for venv, caches, and models) |

### Software Requirements
- **Python:** Version `3.10`, `3.11`, `3.12`, or `3.13`
- **Operating System:**
  - **Linux:** Ubuntu 20.04+, Debian 11+, Fedora 38+, Arch Linux
  - **Windows:** Windows 10 / 11 via **WSL2** (Ubuntu recommended for full CUDA support)
  - **macOS:** macOS Monterey+ (Apple Silicon supported via CPU fallback)
- **Git:** Version 2.25+

---

## 2. Quickstart: 1-Command Automated Setup (Recommended)

The repository provides an automated installation script that checks your environment, detects your GPU, creates a dedicated Python virtual environment, installs all dependencies, downloads required neural network weights, and verifies the installation:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/NTRO-SRM.git
cd NTRO-SRM

# 2. Run automated setup
./setup.sh
```

### Setup Script Options:
```bash
./setup.sh            # Standard installation + runs test suite verification
./setup.sh --no-test  # Fast installation (skips unit tests)
./setup.sh --run      # Installs everything and immediately launches the web app
```

Once setup completes, activate your environment anytime with:
```bash
source venv/bin/activate
```

---

## 3. Step-by-Step Manual Installation

If you prefer to configure your environment manually, follow the steps below:

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/NTRO-SRM.git
cd NTRO-SRM
```

### Step 2: Create and Activate Virtual Environment
```bash
# Linux / macOS / WSL2
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Upgrade Packaging Tools
```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch with CUDA (or CPU)
Depending on your hardware:

- **CUDA 12.x / 13.x (NVIDIA GPU - Recommended):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **CPU-Only (No NVIDIA GPU):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Install NTRO-SRM Package
Install the package in editable mode so changes to `src/` are immediately reflected:
```bash
pip install -e .
```

### Step 7: Download Pretrained Neural Weights
Download both `SEN2SR-Lite` (~0.47M parameters) and `SEN2SR-Swin2SR` (~12.9M parameters) checkpoints:
```bash
python scripts/download_checkpoints.py --model all
```

*Individual downloads:*
```bash
python scripts/download_checkpoints.py --model lite      # Fast baseline only
python scripts/download_checkpoints.py --model swin2sr   # Higher-capacity model only
```

### Step 8: Configure Environment (Optional)
Copy the template configuration file:
```bash
cp .env.example .env
```
To enable streaming directly from the European Space Agency Copernicus Data Space Ecosystem (CDSE), open `.env` and add your free CDSE credentials:
```ini
CDSE_CLIENT_ID="your_cdse_client_id"
CDSE_CLIENT_SECRET="your_cdse_client_secret"
```
*(If left blank, the application automatically falls back to AWS Earth Search public STAC catalog).*

### Step 9: Verify Installation
Run the automated test suite:
```bash
pytest -q
```
All 26 tests should pass.

---

## 4. Launching the Application

### Option A: Launch Interactive Web Application
```bash
source venv/bin/activate
python scripts/run_web.py --port 8000
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### Option B: Run Command-Line Inference on GeoTIFF
```bash
source venv/bin/activate

# Run SEN2SR-Lite (Fast baseline, ~2s)
python scripts/sr_sentinel2.py \
  --input datasets/sample_s2/sample_s2_l2a.tif \
  --output outputs/sr_lite_2.5m.tif \
  --model lite

# Run SEN2SR-Swin2SR (Higher-capacity Vision Transformer)
python scripts/sr_sentinel2.py \
  --input datasets/sample_s2/sample_s2_l2a.tif \
  --output outputs/sr_swin_2.5m.tif \
  --model swin2sr
```

---

## 5. Troubleshooting & FAQ

### 1. `CUDA out of memory (OOM)`
- **SEN2SR-Lite** requires less than **500 MB** VRAM and runs on almost any modern GPU or CPU.
- **SEN2SR-Swin2SR** requires **~2.5 GB** free VRAM during inference.
- If running Swin2SR on a 4GB or 6GB GPU, close GPU-intensive desktop applications (browsers with hardware acceleration, other neural network jobs).
- The framework includes a built-in memory guard in `SRService` that validates available VRAM prior to job execution.

### 2. `Failed to import rasterio` or GDAL issues
On Ubuntu / Debian systems, if `pip install rasterio` fails due to binary compilation issues:
```bash
sudo apt update
sudo apt install -y libgdal-dev gdal-bin python3-dev
pip install rasterio
```
In most standard environments, `pip install rasterio` downloads pre-compiled manylinux wheels automatically.

### 3. Mamba / Selective Scan C++ Extension Errors
Upstream ESAOpenSR MambaSR requires custom CUDA extensions that often fail to compile on newer compilers (GCC 14/15, Python 3.13, CUDA 13). **NTRO-SRM** includes a built-in, pure-PyTorch chunked selective scan (`src/ntro_srm/models/mamba_scan.py`) that operates on native PyTorch tensors without requiring any custom C++ compilation or NVRTC binaries.

### 4. Running Behind an Enterprise Proxy
If your machine is behind an HTTP/HTTPS proxy:
```bash
export HTTP_PROXY="http://proxy.yourcompany.com:8080"
export HTTPS_PROXY="http://proxy.yourcompany.com:8080"
./setup.sh
```
