"""Evaluation script to run and compare SEN2SR-Lite vs SEN2SR-Swin2SR on Sentinel-2."""

from __future__ import annotations

import gc
from pathlib import Path
import time
import numpy as np
from PIL import Image
import rasterio
import torch

from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline
from ntro_srm.web.services.sr_service import render_false_color_cir, render_true_color_rgb


def main():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "datasets" / "sample_s2" / "sample_s2_l2a.tif"
    out_dir = root / "outputs" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path("/home/harsh/.gemini/antigravity-cli/brain/ef21fa25-6741-4bb7-a4d7-bd9d1dafb7d9")

    print(f"[1/5] Input Scene: {input_path}")
    with rasterio.open(input_path) as src:
        print(f"      Input Dimensions: {src.width}x{src.height}")
        print(f"      Input Bands: {src.count}")
        print(f"      Input Resolution: {src.res}")
        print(f"      Input CRS: {src.crs}")

    # 1. Run SEN2SR-Lite
    lite_out_path = out_dir / "sample_s2_sen2sr_lite_2.5m.tif"
    print("\n[2/5] Running SEN2SR-Lite (Fast Baseline)...")
    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.perf_counter()
    pipeline_lite = Sentinel2SRPipeline(model_variant="lite", device="cuda")
    res_lite = pipeline_lite.predict(
        input_path=input_path,
        output_path=lite_out_path,
        overlap=32,
    )
    lite_time = time.perf_counter() - t0
    print(f"      SEN2SR-Lite completed in {lite_time:.2f}s")
    print(f"      Peak VRAM: {res_lite.peak_gpu_memory_mb:.1f} MB")
    print(f"      Output Shape: {res_lite.output_shape}")

    # Unload Lite pipeline
    del pipeline_lite
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Run SEN2SR-Swin2SR
    swin_out_path = out_dir / "sample_s2_sen2sr_swin_2.5m.tif"
    print("\n[3/5] Running SEN2SR-Swin2SR (Higher-Capacity Model)...")
    t0 = time.perf_counter()
    pipeline_swin = Sentinel2SRPipeline(model_variant="swin2sr", device="cuda")
    res_swin = pipeline_swin.predict(
        input_path=input_path,
        output_path=swin_out_path,
        overlap=32,
    )
    swin_time = time.perf_counter() - t0
    print(f"      SEN2SR-Swin2SR completed in {swin_time:.2f}s")
    print(f"      Peak VRAM: {res_swin.peak_gpu_memory_mb:.1f} MB")
    print(f"      Output Shape: {res_swin.output_shape}")

    # 3. Verify GeoTIFFs
    print("\n[4/5] Verifying Output GeoTIFF Metadata...")
    for label, path in [("SEN2SR-Lite", lite_out_path), ("SEN2SR-Swin2SR", swin_out_path)]:
        with rasterio.open(path) as ds:
            print(f"      [{label}] {path.name}:")
            print(f"         Dimensions: {ds.width}x{ds.height} (Expected: 1024x1024)")
            print(f"         Bands: {ds.count} (Expected: 10)")
            print(f"         Resolution: {ds.res} (Expected: (2.5, 2.5))")
            print(f"         CRS: {ds.crs}")
            assert ds.width == 1024 and ds.height == 1024, f"Invalid dimensions: {ds.width}x{ds.height}"
            assert ds.count == 10, f"Invalid band count: {ds.count}"
            assert abs(ds.res[0] - 2.5) < 1e-4 and abs(ds.res[1] - 2.5) < 1e-4, f"Invalid resolution: {ds.res}"

    # 4. Generate Visual Comparisons
    print("\n[5/5] Generating Visual Comparisons (RGB & CIR)...")
    with rasterio.open(lite_out_path) as ds:
        lite_data = ds.read().astype(np.float32)
    with rasterio.open(swin_out_path) as ds:
        swin_data = ds.read().astype(np.float32)

    lr_data = res_lite.lr_tensor.cpu().numpy()  # (10, 256, 256)

    # Standard Sentinel-2 band indices in 10-band stack:
    # 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red), 6: B08 (NIR)
    # RGB = [Red, Green, Blue] -> indices [2, 1, 0]
    # CIR = [NIR, Red, Green] -> indices [6, 2, 1]

    # Interpolate LR to 1024x1024 using nearest neighbor for authentic low-res comparison
    lr_t = torch.from_numpy(lr_data).unsqueeze(0)
    lr_1024 = torch.nn.functional.interpolate(lr_t, size=(1024, 1024), mode="nearest").squeeze(0).numpy()

    lr_rgb_raw = np.transpose(lr_1024[[2, 1, 0]], (1, 2, 0))
    lite_rgb_raw = np.transpose(lite_data[[2, 1, 0]], (1, 2, 0))
    swin_rgb_raw = np.transpose(swin_data[[2, 1, 0]], (1, 2, 0))

    lr_cir_raw = np.transpose(lr_1024[[6, 2, 1]], (1, 2, 0))
    lite_cir_raw = np.transpose(lite_data[[6, 2, 1]], (1, 2, 0))
    swin_cir_raw = np.transpose(swin_data[[6, 2, 1]], (1, 2, 0))

    # Shared radiometric calibration against reference
    lr_rgb_u8 = render_true_color_rgb(lr_rgb_raw, ref_rgb=lr_rgb_raw)
    lite_rgb_u8 = render_true_color_rgb(lite_rgb_raw, ref_rgb=lr_rgb_raw)
    swin_rgb_u8 = render_true_color_rgb(swin_rgb_raw, ref_rgb=lr_rgb_raw)

    lr_cir_u8 = render_false_color_cir(lr_cir_raw, ref_cir=lr_cir_raw)
    lite_cir_u8 = render_false_color_cir(lite_cir_raw, ref_cir=lr_cir_raw)
    swin_cir_u8 = render_false_color_cir(swin_cir_raw, ref_cir=lr_cir_raw)

    # Save individual previews
    Image.fromarray(lite_rgb_u8).save(out_dir / "sen2sr_lite_rgb.png")
    Image.fromarray(swin_rgb_u8).save(out_dir / "sen2sr_swin_rgb.png")
    Image.fromarray(lr_rgb_u8).save(out_dir / "input_10m_rgb.png")

    # Create 3-Panel Side-by-Side Comparison: [Input 10m | SEN2SR-Lite 2.5m | SEN2SR-Swin2SR 2.5m]
    h, w, _ = lr_rgb_u8.shape
    triptych_rgb = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    triptych_rgb[:, :w] = lr_rgb_u8
    triptych_rgb[:, w : w + 10] = 255  # white separator line
    triptych_rgb[:, w + 10 : 2 * w + 10] = lite_rgb_u8
    triptych_rgb[:, 2 * w + 10 : 2 * w + 20] = 255
    triptych_rgb[:, 2 * w + 20 :] = swin_rgb_u8

    triptych_cir = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    triptych_cir[:, :w] = lr_cir_u8
    triptych_cir[:, w : w + 10] = 255
    triptych_cir[:, w + 10 : 2 * w + 10] = lite_cir_u8
    triptych_cir[:, 2 * w + 10 : 2 * w + 20] = 255
    triptych_cir[:, 2 * w + 20 :] = swin_cir_u8

    rgb_comp_path = out_dir / "model_comparison_rgb_triptych.png"
    cir_comp_path = out_dir / "model_comparison_cir_triptych.png"
    Image.fromarray(triptych_rgb).save(rgb_comp_path)
    Image.fromarray(triptych_cir).save(cir_comp_path)

    # Also save a 350x350 zoom-in crop showing fine details
    # Mountain Lake center crop
    y0, x0 = 350, 450
    crop_size = 350
    zoom_lr_rgb = lr_rgb_u8[y0 : y0 + crop_size, x0 : x0 + crop_size]
    zoom_lite_rgb = lite_rgb_u8[y0 : y0 + crop_size, x0 : x0 + crop_size]
    zoom_swin_rgb = swin_rgb_u8[y0 : y0 + crop_size, x0 : x0 + crop_size]

    zoom_triptych = np.zeros((crop_size, crop_size * 3 + 16, 3), dtype=np.uint8)
    zoom_triptych[:, :crop_size] = zoom_lr_rgb
    zoom_triptych[:, crop_size : crop_size + 8] = 255
    zoom_triptych[:, crop_size + 8 : 2 * crop_size + 8] = zoom_lite_rgb
    zoom_triptych[:, 2 * crop_size + 8 : 2 * crop_size + 16] = 255
    zoom_triptych[:, 2 * crop_size + 16 :] = zoom_swin_rgb

    zoom_comp_path = out_dir / "model_comparison_zoom_triptych.png"
    Image.fromarray(zoom_triptych).save(zoom_comp_path)

    # Copy to artifact dir
    if artifact_dir.exists():
        import shutil
        shutil.copyfile(rgb_comp_path, artifact_dir / "model_comparison_rgb_triptych.png")
        shutil.copyfile(cir_comp_path, artifact_dir / "model_comparison_cir_triptych.png")
        shutil.copyfile(zoom_comp_path, artifact_dir / "model_comparison_zoom_triptych.png")
        print(f"      Saved comparison images to {artifact_dir}")

    print("\n[SUCCESS] Model Comparison Complete!")
    print(f"SEN2SR-Lite:    Runtime={lite_time:.2f}s, Peak VRAM={res_lite.peak_gpu_memory_mb:.1f} MB")
    print(f"SEN2SR-Swin2SR: Runtime={swin_time:.2f}s, Peak VRAM={res_swin.peak_gpu_memory_mb:.1f} MB")


if __name__ == "__main__":
    main()
