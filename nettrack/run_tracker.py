"""Queue-facing CLI and CSV/report serialization for mesh tracking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .bootstrap import bootstrap_mesh
from .geometry import load_stereo_calibration
from .topology import NetTopology
from .tracker import MeshTracker, MeshTrackResult


TRACK_FIELDNAMES = [
    "frame", "obj_id", "u", "v", "u_local", "v_local", "method",
    "area_px", "core_radius_px", "solidity", "quality", "valid",
]


def _point_xy(value) -> list[float]:
    if isinstance(value, dict):
        return [float(value.get("x", value.get("u"))), float(value.get("y", value.get("v")))]
    return [float(value[0]), float(value[1])]


def load_setup(path: str | Path) -> tuple[list, list, dict, dict]:
    """Load original-coordinate setup points, crop metadata, and frame metadata."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    left = [_point_xy(value) for value in payload["left"]]
    right = [_point_xy(value) for value in payload["right"]]
    if len(left) != len(right):
        raise ValueError("Setup left/right point counts differ")
    return left, right, payload.get("crops", {}), payload.get("frame_range", {})


def apply_initial_corrections(
    points: list[list[float]], side: str, corrections_path: str | Path | None, crop: dict | None
) -> list[list[float]]:
    """Apply frame-zero SAM2 corrections in ``cropped_original_scale`` space.

    SAM2 adds all positive clicks for an object at the correction frame.  A
    point tracker needs one seed, so their centroid is the equivalent initial
    marker location; a correction box center is used when it has no positives.
    Negative-only corrections do not define a replacement point.
    """
    adjusted = [list(point) for point in points]
    if not corrections_path or not Path(corrections_path).is_file():
        return adjusted
    payload = json.loads(Path(corrections_path).read_text(encoding="utf-8"))
    space = payload.get("coordinate_space", "cropped_original_scale")
    if space != "cropped_original_scale":
        raise ValueError(f"Unsupported corrections coordinate_space: {space}")
    offset_x = float((crop or {}).get("x", 0)) if (crop or {}).get("crop_applied", False) else 0.0
    offset_y = float((crop or {}).get("y", 0)) if (crop or {}).get("crop_applied", False) else 0.0
    for correction in payload.get("corrections", {}).get(side, []):
        if int(correction.get("frame", 0)) != 0:
            continue
        obj_id = int(correction.get("obj_id", -1))
        if obj_id < 0 or obj_id >= len(adjusted):
            continue
        positive = correction.get("positive", [])
        if positive:
            point = np.mean(np.asarray(positive, dtype=float).reshape(-1, 2), axis=0)
        elif correction.get("box") is not None:
            x0, y0, x1, y1 = (float(value) for value in correction["box"])
            point = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        else:
            continue
        adjusted[obj_id] = [float(point[0] + offset_x), float(point[1] + offset_y)]
    return adjusted


def sync_drops(path: str | Path | None) -> tuple[int, int, dict]:
    """Load calibration trims using the triangulation convention.

    ``points_to_3d.py`` defines ``gframe = clip_frame + drop`` and calibration
    sync JSON defines ``global = frame - trim``.  Thus ``drop = -trim`` and a
    pair satisfies ``left_frame + left_drop == right_frame + right_drop``.
    """
    if path is None or str(path).strip().lower() in ("", "none"):
        return 0, 0, {"mode": "none", "left_drop": 0, "right_drop": 0}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "trim_left" in payload and "trim_right" in payload:
        left_drop, right_drop = -int(payload["trim_left"]), -int(payload["trim_right"])
    elif "left_drop" in payload and "right_drop" in payload:
        left_drop, right_drop = int(payload["left_drop"]), int(payload["right_drop"])
    else:
        offset = int(payload.get("offset_left_minus_right", 0))
        left_drop, right_drop = (offset, 0) if offset > 0 else (0, -offset)
    info = dict(payload)
    info.update({"left_drop": left_drop, "right_drop": right_drop})
    return left_drop, right_drop, info


class _VideoFrames:
    """One virtual clip decoded with one seek followed by sequential reads."""

    def __init__(self, path: str | Path, source_start: int, count: int) -> None:
        self.path, self.source_start, self.count = str(path), int(source_start), int(count)

    def __iter__(self):
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, self.source_start)
        try:
            for index in range(self.count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Video ended at requested sequential frame {self.source_start + index}: {self.path}")
                yield frame
        finally:
            capture.release()


def paired_video_inputs(
    left_path: str | Path,
    right_path: str | Path,
    start_frame: int,
    end_frame: int | None,
    left_drop: int,
    right_drop: int,
):
    """Return sequential side readers and the exact triangulation frame pairs."""
    captures = [cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))]
    try:
        if not all(cap.isOpened() for cap in captures):
            raise RuntimeError("Could not open both input videos")
        available = [max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - int(start_frame)) for cap in captures]
    finally:
        for capture in captures:
            capture.release()
    if end_frame is not None:
        requested = max(0, int(end_frame) - int(start_frame))
        available = [min(value, requested) for value in available]
    g_start = max(left_drop, right_drop)
    g_stop = min(available[0] + left_drop, available[1] + right_drop)
    if g_stop <= g_start:
        raise ValueError("No synchronized frame pairs remain after trims")
    indices = [(g - left_drop, g - right_drop) for g in range(g_start, g_stop)]
    first_left, first_right = indices[0]
    return (
        _VideoFrames(left_path, int(start_frame) + first_left, len(indices)),
        _VideoFrames(right_path, int(start_frame) + first_right, len(indices)),
        indices,
    )


def _crop_offset(crop: dict | None) -> tuple[float, float]:
    if crop and crop.get("crop_applied", False):
        return float(crop.get("x", 0)), float(crop.get("y", 0))
    return 0.0, 0.0


def write_outputs(result: MeshTrackResult, out_dir: str | Path, crops: dict | None = None) -> None:
    """Write the exact SAM2 track schema for each side plus the mesh report."""
    out_dir = Path(out_dir)
    crops = crops or {}
    for side in ("left", "right"):
        side_dir = out_dir / side
        side_dir.mkdir(parents=True, exist_ok=True)
        offset_x, offset_y = _crop_offset(crops.get(side))
        with (side_dir / "tracks_2d.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACK_FIELDNAMES)
            writer.writeheader()
            for frame in result.frames:
                assignments = frame.left_assignments if side == "left" else frame.right_assignments
                outputs = frame.left_output if side == "left" else frame.right_output
                frame_number = frame.left_frame if side == "left" else frame.right_frame
                for index, obj_id in enumerate(result.topology.node_ids):
                    detection = assignments[index]
                    measured = detection is not None
                    u, v = outputs[index]
                    area = int(detection.area_px) if measured else 0
                    writer.writerow({
                        "frame": frame_number, "obj_id": obj_id, "u": float(u), "v": float(v),
                        "u_local": float(u - offset_x), "v_local": float(v - offset_y),
                        "method": "mesh" if measured else "mesh_inferred",
                        "area_px": area,
                        "core_radius_px": float(np.sqrt(area / np.pi)) if measured else 0.0,
                        "solidity": float(detection.solidity) if measured else 0.0,
                        "quality": float(detection.quality) if measured else float(frame.inferred_quality[index]),
                        "valid": int(frame.valid[index]),
                    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mesh_track_report.json").write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )


def print_bootstrap_audit(report: dict) -> None:
    """Print a compact setup-QA table with one row per physical grid node."""
    summary = report["summary"]
    print(
        "[BOOTSTRAP] "
        f"confirmed={summary['confirmed']} repaired={summary['repaired']} absent={summary['absent']} "
        f"detections={summary['detections_left']}/{summary['detections_right']} "
        f"ordering_flags={summary['monotonic_violations']}"
    )
    print("  row col  obj_id  status     views        distance_px (L/R)  reason")
    for node in report["nodes"]:
        distance = node["repair_distance_px"]
        values = "/".join("-" if distance[side] is None else f"{distance[side]:.1f}" for side in ("left", "right"))
        views = ",".join(node["repaired_views"]) or "-"
        reason = node["reason"] or ("ordering_violation" if node["monotonic_violation"] else "-")
        print(
            f"  {node['row']:3d} {node['col']:3d} {node['obj_id']:7d}  "
            f"{node['status']:<10} {views:<12} {values:<18} {reason}"
        )


def write_bootstrap_overlay(image: np.ndarray, report: dict, path: str | Path) -> None:
    """Write the requested human-review overlay of established left identities."""
    overlay = image.copy()
    colors = {"confirmed": (40, 210, 40), "repaired": (0, 180, 255), "absent": (40, 40, 230)}
    for node in report["nodes"]:
        point = node["left"]
        if point is None:
            continue
        center = tuple(int(round(value)) for value in point)
        color = colors[node["status"]]
        cv2.circle(overlay, center, 5, color, 2, cv2.LINE_AA)
        cv2.putText(overlay, str(node["obj_id"]), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Could not write bootstrap overlay: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topology-constrained stereo mesh tracker")
    parser.add_argument("--left-input", required=True)
    parser.add_argument("--right-input", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--setup-json", required=True)
    parser.add_argument("--corrections-json", default=None)
    parser.add_argument("--stereo", required=True)
    parser.add_argument("--sync-json", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid-cols", type=int, required=True)
    parser.add_argument("--grid-rows", type=int, required=True)
    parser.add_argument("--audit-only", action="store_true", help="Audit and repair first-frame identities without tracking")
    parser.add_argument("--review-html", action="store_true", help="Also write a self-contained marker_review.html")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    left_points, right_points, crops, _frame_range = load_setup(args.setup_json)
    left_points = apply_initial_corrections(left_points, "left", args.corrections_json, crops.get("left"))
    right_points = apply_initial_corrections(right_points, "right", args.corrections_json, crops.get("right"))
    topology = NetTopology.from_grid(args.grid_cols, args.grid_rows)
    if len(topology) != len(left_points):
        raise ValueError(
            f"Grid has {len(topology)} nodes but setup has {len(left_points)}; use the Python API with explicit present nodes for a holed net"
        )
    left_drop, right_drop, sync_info = sync_drops(args.sync_json)
    left_frames, right_frames, indices = paired_video_inputs(
        args.left_input, args.right_input, args.start_frame, args.end_frame, left_drop, right_drop
    )
    calibration = load_stereo_calibration(args.stereo)
    tracker = MeshTracker(calibration, topology)
    first_left_image = next(iter(left_frames))
    first_right_image = next(iter(right_frames))
    if args.audit_only:
        audit = bootstrap_mesh(
            first_left_image, first_right_image, np.asarray(left_points), np.asarray(right_points),
            topology, calibration, left_roi=crops.get("left"), right_roi=crops.get("right"),
        )
        print_bootstrap_audit(audit.report)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "setup_audit.json").write_text(json.dumps(audit.report, indent=2) + "\n", encoding="utf-8")
        write_bootstrap_overlay(first_left_image, audit.report, out / "bootstrap_overlay_left.png")
        if args.review_html:
            from .review import generate_review
            review_path, _ = generate_review(
                out, args.left_input, args.right_input, args.start_frame, frames=1,
                setup_report=audit.report, sync_info=sync_info, frame_pairs=[indices[0]],
            )
            print(f"[OK] Wrote marker review: {review_path}")
        print(f"[OK] Wrote setup audit and overlay: {out}")
        return 0

    def progress(done, metrics):
        if done == 1 or done % 10 == 0 or done == len(indices):
            print(
                f"[MESHTRACK] {done}/{len(indices)} pairs  detections="
                f"{metrics['detections_left']}/{metrics['detections_right']}  "
                f"inferred={metrics['inferred']} lost={metrics['lost']}  "
                f"{metrics['timing_s']['total']:.3f}s"
            )

    result = tracker.track(
        left_frames, right_frames, left_points, right_points,
        frame_indices=indices, left_roi=crops.get("left"), right_roi=crops.get("right"), progress=progress,
        bootstrap_callback=print_bootstrap_audit,
    )
    result.report["sync"] = sync_info
    result.report["inputs"] = {
        "left": str(args.left_input), "right": str(args.right_input),
        "start_frame": args.start_frame, "end_frame": args.end_frame,
    }
    write_outputs(result, args.out, crops)
    write_bootstrap_overlay(first_left_image, result.report["bootstrap"], Path(args.out) / "bootstrap_overlay_left.png")
    if args.review_html:
        from .review import generate_review
        review_path, _ = generate_review(
            args.out, args.left_input, args.right_input, args.start_frame,
        )
        print(f"[OK] Wrote marker review: {review_path}")
    print(f"[OK] Wrote mesh tracks and report: {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
