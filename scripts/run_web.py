#!/usr/bin/env python3
"""Run script for NTRO-SRM Interactive Super-Resolution Web Application.

Usage:
    python scripts/run_web.py
    python scripts/run_web.py --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src is on sys.path
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import uvicorn

from ntro_srm.web.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch NTRO-SRM Sentinel-2 Super-Resolution Web UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Network interface to bind server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP server.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Default auto-detects CUDA.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn hot reloading for development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load environment variables
    try:
        import dotenv
        dotenv.load_dotenv(project_root / ".env")
    except ImportError:
        pass

    # Detect hardware
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Fallback)"
    active_device = args.device if args.device else ("cuda" if cuda_avail else "cpu")

    import os
    cdse_id = os.environ.get("CDSE_CLIENT_ID") or os.environ.get("COPERNICUS_CLIENT_ID")
    cdse_status = f"Connected ({cdse_id[:8]}...)" if cdse_id else "Not configured (Using AWS STAC)"

    print("\n" + "=" * 48)
    print("NTRO-SRM Web Interface")
    print("=" * 48)
    print(f"Model:      SEN2SR-Lite")
    print(f"Device:     {active_device.upper()}")
    print(f"GPU:        {gpu_name}")
    print(f"Copernicus: {cdse_status}")
    print(f"\nServer:")
    print(f"http://{args.host}:{args.port}")
    print("=" * 48 + "\n")

    app = create_app(workspace_root=project_root, device=active_device)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
