# NTRO-SRM User & Developer Guide

Complete guide for operating the **NTRO-SRM** Sentinel-2 Multi-Spectral Super-Resolution platform via the Web Graphical User Interface, Command-Line Interface (CLI), and Python SDK.

---

## 1. Web Graphical User Interface (GUI)

The interactive web GIS interface provides a complete end-to-end workflow from satellite acquisition to super-resolved GeoTIFF export.

### Starting the Web Application
```bash
source venv/bin/activate
python scripts/run_web.py --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in any modern web browser.

---

### Key Web Features & Walkthrough

#### 1. Area of Interest (AOI) Selection
- **Interactive Draw Mode:** Click **"Draw Box"** and click-and-drag directly on the Leaflet map to define an AOI.
- **Quick Location Presets:** Click any floating preset pill at the top of the map:
  - 🌲 **Mountain Lake (VA):** Mixed alpine lake and dense deciduous forest.
  - 🌾 **Salinas Valley (CA):** High-frequency agricultural field boundaries.
  - 🏙️ **Frankfurt Airport (Germany):** Urban runways and building clusters.
  - ⚓ **Rotterdam Port (Netherlands):** Maritime channels and industrial infrastructure.
  - 🏖️ **Dubai Palm Jumeirah (UAE):** Coastal geometric marine landforms.
  - 🏔️ **Lake Tahoe (CA/NV):** Deep alpine water body and mountain ridges.
- **Manual Coordinates:** Expand the **"Manual Coordinates"** accordion in the sidebar to paste exact WGS84 bounding box coordinates.
- **Pre-downloaded Sample Scene:** Click **"Load Sample Scene"** in the top-right header to instantly test the system without downloading remote satellite tiles.

#### 2. Model Architecture Selection
Select your desired super-resolution neural network in the sidebar before clicking upscale:
- **SEN2SR-Lite (Default Baseline):**
  - Parameter Count: **~0.47 Million**
  - Architecture: Swift Parameter-free Attention Network (SPAN) CNN
  - Execution Time: **~2 seconds** on CUDA GPU
  - Best for: Rapid exploratory mapping, preliminary screening, real-time interactive panning.
- **SEN2SR-Swin2SR (Higher-Capacity Model):**
  - Parameter Count: **~12.9 Million** ($27.4\times$ capacity)
  - Architecture: Vision Transformer (Swin2SR) + State-Space (MambaSR)
  - Memory Footprint: ~2.46 GB VRAM on CUDA
  - Best for: Complex structural detail, fine edge reconstruction, and dense infrastructure.

#### 3. Real-Time Processing & Telemetry
Click **"UPSCALE PATCH TO 2.5m"**. The application tracks:
- Progress bar and pipeline stages (AOI extraction $\to$ radiometric normalization $\to$ neural inference $\to$ GeoTIFF generation $\to$ overlay synthesis).
- Real-time elapsed time in seconds and peak GPU VRAM allocation.

#### 4. Interactive Split-Slider & Visual Comparison
Once inference completes, Leaflet map overlays are dynamically rendered:
- **Swipe Split View (Default):** Drag the horizontal slider left/right to compare native $10\text{m}$ input against $2.5\text{m}$ super-resolved output.
- **Side Selection:** Switch the left comparison pane between **Native 10m S2** and **Bicubic 2.5m Baseline**.
- **Continuous Opacity Blend:** Switch to **Blend** mode and adjust the opacity slider from 0% to 100%.
- **Single Layer Mode:** Toggle **10m Native Only** or **2.5m SR Only**.

#### 5. Dual Multi-Spectral Color Modes
- **Natural RGB Mode:** True color representation combining Red (B04), Green (B03), and Blue (B02) calibrated using physical reflectance scaling to avoid neon saturation.
- **Color Infrared (CIR) Mode:** High-contrast false color infrared combining NIR (B08), Red (B04), and Green (B03) rendering healthy photosynthetic vegetation in rich crimson tones.

#### 6. Direct GeoTIFF Upload (Tab 2)
To upscale an existing Sentinel-2 GeoTIFF from your disk:
1. Navigate to **Tab 2: Upload GeoTIFF**.
2. Drag and drop any 10-band, 12-band, or RGB GeoTIFF file up to $512 \times 512$ pixels.
3. Click **"Upscale Uploaded Image"**.

#### 7. Downloading Super-Resolved Products
Download buttons appear automatically upon completion:
- **Download 10-Band 2.5m GeoTIFF:** Full scientific 32-bit floating-point multi-band raster with exact CRS and geotransform.
- **Download RGB (PNG):** Georeferenced true-color image.
- **Download CIR (PNG):** Georeferenced false-color infrared image.

---

## 2. Command-Line Interface (CLI)

The CLI allows automated batch processing of GeoTIFF files on headless servers.

### Basic Syntax
```bash
python scripts/sr_sentinel2.py \
  --input <path_to_input_geotiff> \
  --output <path_to_output_geotiff> \
  [--model {lite,swin2sr}] \
  [--device {cuda,cpu}] \
  [--overlap OVERLAP_PIXELS]
```

### CLI Examples

#### Example 1: Run Fast Baseline (SEN2SR-Lite) on Sample Scene
```bash
python scripts/sr_sentinel2.py \
  --input datasets/sample_s2/sample_s2_l2a.tif \
  --output outputs/sample_s2_lite_2.5m.tif \
  --model lite \
  --device cuda
```

#### Example 2: Run Higher-Capacity Vision Transformer (SEN2SR-Swin2SR)
```bash
python scripts/sr_sentinel2.py \
  --input datasets/sample_s2/sample_s2_l2a.tif \
  --output outputs/sample_s2_swin_2.5m.tif \
  --model swin2sr \
  --device cuda \
  --overlap 32
```

#### Example 3: CPU Fallback Execution
```bash
python scripts/sr_sentinel2.py \
  --input datasets/sample_s2/sample_s2_l2a.tif \
  --output outputs/sample_s2_cpu_2.5m.tif \
  --model lite \
  --device cpu
```

#### Example 4: Run Dual Model Benchmark Script
```bash
python scripts/compare_models.py
```
This generates side-by-side triptych comparisons saved in `outputs/comparisons/`.

---

## 3. Python SDK & Programmatic Usage

You can embed the NTRO-SRM pipeline directly into custom Python geospatial pipelines.

### End-to-End Pipeline
```python
from pathlib import Path
from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline

# 1. Initialize pipeline with requested model variant
pipeline = Sentinel2SRPipeline(
    model_variant="swin2sr",  # or "lite"
    device="cuda",            # or "cpu"
)

# 2. Execute 4x super-resolution and export to GeoTIFF
result = pipeline.predict(
    input_path=Path("datasets/sample_s2/sample_s2_l2a.tif"),
    output_path=Path("outputs/my_sr_result_2.5m.tif"),
    overlap=32,
)

print(f"Super-resolution completed in: {result.inference_time_ms:.1f} ms")
print(f"Output raster dimensions: {result.output_shape} (10 bands, 2.5m GSD)")
print(f"Coordinate Reference System: {result.crs}")
```

### Direct Tensor Inference
```python
import torch
from ntro_srm.models.sen2sr import SEN2SRModel

# Load neural model adapter
model = SEN2SRModel(model_variant="lite", device="cuda")

# Input: Normalized reflectance tensor of shape (Batch, 10 bands, Height, Width)
lr_tensor = torch.rand(1, 10, 128, 128, device="cuda")

# Predict 4x spatial resolution: shape (1, 10, 512, 512)
with torch.no_grad():
    sr_tensor = model.predict(lr_tensor)

print("Super-resolved tensor shape:", sr_tensor.shape)
```

---

## 4. REST API Endpoints Reference

The FastAPI backend exposes the following RESTful endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/system-info` | `GET` | Returns GPU name, available VRAM, CUDA status, and model metadata. |
| `/api/demo/info` | `GET` | Returns metadata of pre-installed local Sentinel-2 sample scene. |
| `/api/sentinel/search` | `POST` | Queries STAC catalog (CDSE or AWS Earth Search) for cloud-free Sentinel-2 scenes. |
| `/api/sr/process` | `POST` | Enqueues background super-resolution job for specified AOI and model. |
| `/api/sr/upload` | `POST` | Uploads custom GeoTIFF patch and enqueues super-resolution job. |
| `/api/sr/jobs/{job_id}` | `GET` | Polls real-time progress percentage, current step, and results. |
| `/api/sr/jobs/{job_id}/preview/{layer}` | `GET` | Serves georeferenced PNG preview overlay (`lr_rgb`, `sr_rgb`, `bicubic_rgb`, `lr_cir`, etc.). |
| `/api/sr/jobs/{job_id}/download/{type}` | `GET` | Downloads 10-band 2.5m GeoTIFF (`type=geotiff`), RGB (`type=rgb`), or CIR (`type=cir`). |

### Example cURL Request: Check System Status
```bash
curl -s http://127.0.0.1:8000/api/system-info | jq .
```
