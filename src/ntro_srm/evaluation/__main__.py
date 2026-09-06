"""Command-line entry point for NTRO-SRM scientific evaluation."""

from __future__ import annotations

import argparse
import json

from ntro_srm.evaluation.evaluator import evaluate_geotiffs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a super-resolved GeoTIFF against its Sentinel-2 source and/or HR reference."
    )
    parser.add_argument("prediction", help="Path to the super-resolved GeoTIFF")
    parser.add_argument("--source", help="Path to the low-resolution Sentinel-2 source")
    parser.add_argument("--reference", help="Path to a georeferenced high-resolution reference")
    parser.add_argument("--scale-factor", type=float, default=4.0, help="SR scale used by ERGAS")
    parser.add_argument("--output", help="Optional path for the JSON report")
    args = parser.parse_args()

    report = evaluate_geotiffs(
        args.prediction,
        source_path=args.source,
        reference_path=args.reference,
        scale_factor=args.scale_factor,
    )
    if args.output:
        report.write_json(args.output)
        print(args.output)
    else:
        print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
