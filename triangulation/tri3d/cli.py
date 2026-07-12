"""Command-line interface and orchestration for stereo track triangulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calib import StereoCalibration, load_stereo_calibration
from .io_tracks import (
    load_existing_triangulation_csv,
    load_track_csv,
    write_triangulation_csv,
)
from .matching import match_frame_observations
from .syncmap import detect_sync_from_videos, load_calibration_sync
from .triangulate import triangulate_observations
from .viz_scene import visualize_existing_rows


def build_parser() -> argparse.ArgumentParser:
    """Build the historical ``points_to_3d.py`` command-line interface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo", default="calibration/work/stereo.npz", help="Path to stereo.npz")
    parser.add_argument("--left", default="sam2/sam2/out/left/tracks_2d.csv", help="Left tracks_2d.csv")
    parser.add_argument("--right", default="sam2/sam2/out/right/tracks_2d.csv", help="Right tracks_2d.csv")
    parser.add_argument("--left-video", default="sam2/sam2/in/left.mp4", help="Left video path for per-run sync detection or visualization")
    parser.add_argument("--right-video", default="sam2/sam2/in/right.mp4", help="Right video path for per-run sync detection")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive source frame for the virtual clip")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive source frame for the virtual clip")
    parser.add_argument("--frame-step", type=int, default=1, help="Source-frame stride used when SAM2 tracks were generated")
    parser.add_argument("--sync-mode", choices=["flash", "audio", "hybrid"], default=None,
                        help="Sync detection mode: audio peak, flash, or hybrid(visual)")
    parser.add_argument("--sync-scale", type=float, default=0.25, help="Downscale for sync detection")
    parser.add_argument("--sync-max-frames", type=int, default=-1, help="Frames to scan for sync (-1=all)")
    parser.add_argument("--sync-json", default="calibration/work/sync.json",
                        help="Calibration sync.json with trim_left/trim_right; use 'none' to disable")
    parser.add_argument("--left-drop", type=int, default=None, help="Manual left frame drop override")
    parser.add_argument("--right-drop", type=int, default=None, help="Manual right frame drop override")
    parser.add_argument("--out-sync", default=None, help="Optional path to save detected sync JSON")
    parser.add_argument("--out-csv", default="triangulation/out/triangulated_3d.csv")
    parser.add_argument("--out-summary", default="triangulation/out/summary.json")
    parser.add_argument("--visualize", action="store_true", help="Generate side-by-side original + CSV reconstruction video")
    parser.add_argument("--viz-only", action="store_true",
                        help="Skip triangulation and render visualization from an existing --out-csv")
    parser.add_argument("--viewer-only", action="store_true",
                        help="Write only the interactive Three.js viewer; skip visualization videos")
    parser.add_argument("--viz-out", default="triangulation/out/triangulated_3d_viz.mp4", help="Output path for visualization video")
    parser.add_argument("--viz-max-frames", type=int, default=-1, help="Max rendered frames with tracked nodes (-1=all)")
    parser.add_argument("--viz-mode", choices=["scene", "sidebyside"], default="scene",
                        help="scene: depth-colored overlay + isometric + top-down. sidebyside: original legacy view.")
    parser.add_argument("--viz-trail", type=int, default=60, help="Recent-frames trail length in 3D panel (scene mode; currently unused)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers for iso/topdown rendering. 0=auto (cpu_count).")
    parser.add_argument("--viz-encoder", choices=["auto", "nvenc", "mp4v"], default="auto",
                        help="Video encoder. auto=nvenc only if ffmpeg can initialize it, else mp4v.")
    parser.add_argument("--viz-grid-cols", type=int, default=0,
                        help="Marker grid columns for visualization net edges. 0=auto from object count/aspect.")
    parser.add_argument("--viz-grid-rows", type=int, default=0,
                        help="Marker grid rows for visualization net edges. 0=auto from object count/aspect.")
    parser.add_argument("--quality-min", type=float, default=0.0, help="Filter 2D points by quality >= this")
    parser.add_argument("--max-reproj", type=float, default=20.0, help="Reject if mean reproj err > this (px)")
    return parser


def _write_sync(path: str, sync_info: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sync_info, indent=2))
    print(f"[OK] Wrote {output}")


def _resolve_sync(args: argparse.Namespace) -> tuple[int, int, dict[str, Any]]:
    explicit_sync_mode = args.sync_mode is not None
    sync_mode = args.sync_mode or "audio"
    sync_json = None if str(args.sync_json).strip().lower() in {"", "none", "null", "false"} else args.sync_json
    if explicit_sync_mode and args.sync_json == "calibration/work/sync.json":
        sync_json = None
    has_manual_sync = args.left_drop is not None or args.right_drop is not None
    if explicit_sync_mode and sync_json:
        raise ValueError("Use either --sync-mode for new sync or --sync-json for saved sync, not both.")
    if sync_json and has_manual_sync:
        raise ValueError("Use either --sync-json or --left-drop/--right-drop, not both.")

    if sync_json:
        left_drop, right_drop, sync_info = load_calibration_sync(sync_json)
        print(
            "[INFO] Sync mapping (calibration sync JSON): "
            f"trim_left={sync_info['trim_left']}, trim_right={sync_info['trim_right']} -> "
            f"left_drop={left_drop}, right_drop={right_drop}"
        )
        if args.out_sync:
            _write_sync(args.out_sync, sync_info)
    elif has_manual_sync:
        left_drop = int(args.left_drop or 0)
        right_drop = int(args.right_drop or 0)
        sync_info = {"mode": "manual", "left_drop": left_drop, "right_drop": right_drop}
        print(f"[INFO] Sync mapping (manual): left_drop={left_drop}, right_drop={right_drop}")
    else:
        if not args.left_video or not args.right_video:
            raise ValueError("Provide --left-video and --right-video for auto sync, or set --left-drop/--right-drop manually.")
        left_drop, right_drop, sync_info = detect_sync_from_videos(
            args.left_video,
            args.right_video,
            mode=sync_mode,
            scale=args.sync_scale,
            max_frames=args.sync_max_frames,
            preview_out_path=args.out_sync,
        )
        print(
            f"[INFO] Sync mapping ({sync_info['mode']}): "
            f"left_drop={left_drop}, right_drop={right_drop}, "
            f"left_event={sync_info['left_event_frame']}, right_event={sync_info['right_event_frame']}"
        )
        if args.out_sync:
            _write_sync(args.out_sync, sync_info)
    return left_drop, right_drop, sync_info


def _print_calibration(calibration: StereoCalibration) -> None:
    print(f"[INFO] Loaded stereo: scale={calibration.scale}, image_size={calibration.image_size}")
    if not np.isclose(calibration.coordinate_scale, 1.0):
        print(
            f"[INFO] Scaling input 2D tracks by stereo scale={calibration.coordinate_scale:g} for triangulation; "
            "output/reprojection columns stay in original full-frame pixels."
        )


def _write_summary(
    args: argparse.Namespace,
    calibration: StereoCalibration,
    sync_info: dict[str, Any],
    rows: list[dict[str, Any]],
    reprojection_errors: list[float],
) -> Path:
    valid_count = sum(1 for row in rows if row["valid_3d"] == 1)
    total_count = len(rows)
    errors = np.array(reprojection_errors, dtype=np.float64) if reprojection_errors else np.array([], dtype=np.float64)
    summary = {
        "total_matched_obs": total_count,
        "valid_3d_obs": valid_count,
        "valid_pct": (100.0 * valid_count / max(total_count, 1)),
        "sync": sync_info,
        "quality_min": args.quality_min,
        "max_reproj_px": args.max_reproj,
        "stereo_scale": calibration.scale,
        "stereo_image_size": list(calibration.image_size) if calibration.image_size else None,
        "input_track_coordinate_space": "original_full_frame_pixels",
        "reprojection_error_coordinate_space": "original_full_frame_pixels",
        "frame_step": int(args.frame_step),
        "reproj_err_px": {
            "mean": float(errors.mean()) if errors.size else None,
            "median": float(np.median(errors)) if errors.size else None,
            "p95": float(np.percentile(errors, 95)) if errors.size else None,
        },
    }
    output = Path(args.out_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return output


def main(argv: Sequence[str] | None = None) -> None:
    """Run triangulation or visualization-only mode."""

    args = build_parser().parse_args(argv)
    if args.viewer_only and not args.visualize:
        raise ValueError("--viewer-only requires --visualize.")
    if args.viewer_only and args.viz_mode != "scene":
        raise ValueError("--viewer-only requires --viz-mode scene.")
    if args.viz_only:
        if not args.visualize:
            raise ValueError("--viz-only requires --visualize.")
        visualize_existing_rows(args, load_existing_triangulation_csv(args.out_csv))
        return

    calibration = load_stereo_calibration(args.stereo)
    _print_calibration(calibration)
    left_drop, right_drop, sync_info = _resolve_sync(args)
    left = load_track_csv(args.left, args.quality_min)
    right = load_track_csv(args.right, args.quality_min)
    keys, left_keyed, right_keyed = match_frame_observations(left, right, left_drop, right_drop)
    print(f"[INFO] Matched observations: {len(keys)}")
    rows, errors = triangulate_observations(
        calibration,
        keys,
        left_keyed,
        right_keyed,
        source_start_frame=args.start_frame,
        frame_step=args.frame_step,
        max_reproj=args.max_reproj,
    )
    out_csv = write_triangulation_csv(args.out_csv, rows)
    print(f"[OK] Wrote {out_csv}")
    out_summary = _write_summary(args, calibration, sync_info, rows, errors)
    print(f"[OK] Wrote {out_summary}")
    if args.visualize:
        visualize_existing_rows(args, rows)
