#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# Local import (vendored tracklib.py in repo root)
from tracklib import get_bundle_backbone

LOG = logging.getLogger("bundle_skeleton")

def _existing_file(p: str) -> str:
    path = Path(p)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {p}")
    if path.is_dir():
        raise argparse.ArgumentTypeError(f"Expected a file, got directory: {p}")
    return str(path)

def _float_0_1(p: str) -> float:
    v = float(p)
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("Value must be in [0, 1].")
    return v

def _pos_float(p: str) -> float:
    v = float(p)
    if v < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return v

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bundle_skeleton.py",
        description="Compute bundle backbone / skeleton from streamline bundle (.tck) using tracklib.get_bundle_backbone().",
    )
    ap.add_argument("track_in", type=_existing_file, help="Input bundle tractogram (.tck expected).")
    ap.add_argument("track_out", type=str, help="Output tractogram path (will be written).")
    ap.add_argument("structural", type=_existing_file, help="Reference structural image (e.g., T1w NIfTI).")

    ap.add_argument("--perc", type=_float_0_1, default=0.0,
                    help="Core streamline percentile. Use 0 to disable (matches perc=None in shell wrapper).")
    ap.add_argument("--smooth-density", dest="smooth_density", action="store_true", default=True,
                    help="Enable density smoothing in core selection.")
    ap.add_argument("--no-smooth-density", dest="smooth_density", action="store_false",
                    help="Disable density smoothing in core selection.")

    ap.add_argument("--length-thr", type=_float_0_1, default=0.90,
                    help="Keep streamlines with length > length_thr * max_length.")
    ap.add_argument("--n-points", type=int, default=32,
                    help="Number of points used to resample each streamline.")

    ap.add_argument("--keep-endpoints", action="store_true", default=False,
                    help="Use endpoint_mode for endpoints and preserve endpoints through smoothing.")

    ap.add_argument("--average-type", choices=["mean", "median"], default="mean",
                    help="Aggregation method for interior points.")
    ap.add_argument("--endpoint-mode", choices=["mean", "median", "median_project"], default="median",
                    help="Endpoint aggregation method when keep_endpoints is enabled.")
    ap.add_argument("--representative", action="store_true", default=False,
                    help="Output closest streamline to backbone instead of the backbone.")
    ap.add_argument("--spline-smooth", type=_pos_float, default=None,
                    help="Spline smoothing parameter s. If omitted, no smoothing is applied.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"],
                    help="Logging verbosity.")

    return ap

def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    LOG.info("Input tractogram: %s", args.track_in)
    LOG.info("Reference image:  %s", args.structural)
    LOG.info("Output tract:     %s", args.track_out)

    if args.n_points < 2:
        raise SystemExit("ERROR: --n-points must be >= 2")

    out_path = Path(args.track_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Call tracklib. In tracklib, perc==0 triggers the non-core path; this matches your shell wrapper behaviour. fileciteturn0file0
    get_bundle_backbone(
        args.track_in,
        str(out_path),
        args.structural,
        N_points=int(args.n_points),
        perc=float(args.perc),
        smooth_density=bool(args.smooth_density),
        length_thr=float(args.length_thr),
        keep_endpoints=bool(args.keep_endpoints),
        average_type=str(args.average_type),
        endpoint_mode=str(args.endpoint_mode),
        representative=bool(args.representative),
        spline_smooth=args.spline_smooth,
    )

    # Optional: write product.json summary (keep small). citeturn3view2turn23search4
    summary = {
        "bundle_skeleton": {
            "perc": args.perc,
            "smooth_density": args.smooth_density,
            "length_thr": args.length_thr,
            "N_points": args.n_points,
            "keep_endpoints": args.keep_endpoints,
            "average_type": args.average_type,
            "endpoint_mode": args.endpoint_mode,
            "representative": args.representative,
            "spline_smooth": args.spline_smooth,
        },
        "tags": ["bundle_skeleton"]
    }
    with open("product.json", "w") as f:
        json.dump(summary, f, indent=2)

    LOG.info("Done")

if __name__ == "__main__":
    main()
