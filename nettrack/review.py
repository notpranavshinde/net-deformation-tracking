"""Build a self-contained, human-readable stereo marker review page."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .run_tracker import paired_video_inputs


def _read_tracks(path: Path) -> dict[int, dict[int, dict]]:
    frames: dict[int, dict[int, dict]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame, obj_id = int(row["frame"]), int(row["obj_id"])
            frames.setdefault(frame, {})[obj_id] = {
                "u": float(row["u"]), "v": float(row["v"]),
                "quality": float(row["quality"]), "method": row["method"],
                "status": "measured" if row["method"] == "mesh" else "inferred",
                "valid": bool(int(row.get("valid", 1))),
            }
    return frames


def _sample_indices(count: int, wanted: int) -> list[int]:
    if count < 1 or wanted < 1:
        raise ValueError("Review requires at least one available and requested frame")
    amount = min(count, wanted)
    return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, amount)))


def _setup_positions(nodes: list[dict], side: str) -> dict[int, tuple[float, float]]:
    """Return setup positions, filling old-report gaps from the visible grid."""
    result = {}
    grid, image = [], []
    for node in nodes:
        point = node.get(f"predicted_{side}") or node.get(side)
        if point is not None:
            result[int(node["obj_id"])] = (float(point[0]), float(point[1]))
            grid.append((float(node["col"]), float(node["row"])))
            image.append(point)
    missing = [node for node in nodes if int(node["obj_id"]) not in result]
    if missing and len(grid) >= 4:
        matrix, _ = cv2.findHomography(np.asarray(grid), np.asarray(image), method=0)
        if matrix is not None:
            query = np.asarray([[[float(n["col"]), float(n["row"])]] for n in missing])
            predicted = cv2.perspectiveTransform(query, matrix)[:, 0]
            result.update({int(n["obj_id"]): tuple(map(float, uv)) for n, uv in zip(missing, predicted)})
    return result


def _encode(frame: np.ndarray, scale: float) -> tuple[str, int, int, float, float]:
    height, width = frame.shape[:2]
    out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Could not JPEG-encode a review frame")
    return base64.b64encode(encoded).decode("ascii"), out_w, out_h, out_w / width, out_h / height


def _frame_pairs(report: dict, tracks: dict[str, dict]) -> list[tuple[int, int]]:
    reported = report.get("frames", [])
    if reported:
        return [(int(item["left_frame"]), int(item["right_frame"])) for item in reported]
    return list(zip(sorted(tracks["left"]), sorted(tracks["right"])))


def generate_review(
    run_dir: str | Path, left_video: str | Path, right_video: str | Path,
    start_frame: int, *, frames: int = 9, image_scale: float = 0.5,
    setup_report: dict | None = None, sync_info: dict | None = None,
    frame_pairs: list[tuple[int, int]] | None = None,
) -> tuple[Path, list[tuple[int, int]]]:
    """Generate ``marker_review.html`` and return its path and sampled pairs."""
    if not 0 < image_scale <= 1:
        raise ValueError("--image-scale must be greater than 0 and at most 1")
    run_dir = Path(run_dir)
    report_path = run_dir / "mesh_track_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    setup = setup_report or report.get("setup_check") or report.get("bootstrap")
    if not setup:
        audit_path = run_dir / "setup_audit.json"
        if audit_path.is_file():
            setup = json.loads(audit_path.read_text(encoding="utf-8"))
    if not setup:
        raise FileNotFoundError("No setup-check report found in the run directory")

    tracks = {side: (_read_tracks(run_dir / side / "tracks_2d.csv")
                     if (run_dir / side / "tracks_2d.csv").is_file() else {})
              for side in ("left", "right")}
    pairs = frame_pairs or _frame_pairs(report, tracks)
    if not pairs:  # audit-only: exactly the first synchronized pair
        sync = report.get("sync", {})
        pairs = [(max(0, int(sync.get("right_drop", 0)) - int(sync.get("left_drop", 0))),
                  max(0, int(sync.get("left_drop", 0)) - int(sync.get("right_drop", 0))))]
    chosen = _sample_indices(len(pairs), frames)
    sampled_pairs = [pairs[index] for index in chosen]

    sync = sync_info or report.get("sync", {})
    left_drop, right_drop = int(sync.get("left_drop", 0)), int(sync.get("right_drop", 0))
    end_frame = report.get("inputs", {}).get("end_frame")
    left_stream, right_stream, decoded_pairs = paired_video_inputs(
        left_video, right_video, start_frame, end_frame, left_drop, right_drop
    )
    pair_to_sample = {pair: i for i, pair in enumerate(sampled_pairs)}
    decoded: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(sampled_pairs)
    for left_image, right_image, pair in zip(left_stream, right_stream, decoded_pairs):
        if pair in pair_to_sample:
            decoded[pair_to_sample[pair]] = (left_image, right_image)
        if all(item is not None for item in decoded):
            break
    if any(item is None for item in decoded):
        raise RuntimeError("Could not decode every sampled stereo frame pair")

    setup_nodes = setup["nodes"]
    setup_by_id = {int(node["obj_id"]): node for node in setup_nodes}
    tracking_by_id = {int(node["obj_id"]): node for node in report.get("nodes", [])}
    setup_uv = {side: _setup_positions(setup_nodes, side) for side in ("left", "right")}
    payload_frames = []
    for pair, images in zip(sampled_pairs, decoded):
        view_payload, scales = {}, {}
        for side, image in zip(("left", "right"), images):
            jpeg, width, height, sx, sy = _encode(image, image_scale)
            view_payload[side] = {"image": "data:image/jpeg;base64," + jpeg,
                                  "width": width, "height": height}
            scales[side] = (sx, sy)
        node_ids = sorted(set(setup_by_id) | set(tracks["left"].get(pair[0], {})) |
                          set(tracks["right"].get(pair[1], {})))
        nodes = []
        for obj_id in node_ids:
            setup_node = setup_by_id.get(obj_id, {})
            intervals = tracking_by_id.get(obj_id, {}).get("suspect_intervals", [])
            one_view_intervals = tracking_by_id.get(obj_id, {}).get("one_view_intervals", [])
            node = {"id": obj_id, "row": setup_node.get("row"), "col": setup_node.get("col"),
                    "label": f"{setup_node.get('row', '?')},{setup_node.get('col', '?')}",
                    "setupStatus": setup_node.get("status", "confirmed"),
                    "reason": setup_node.get("reason"),
                    "repairedViews": setup_node.get("repaired_views", []),
                    "repairDistance": setup_node.get("repair_distance_px", {}),
                    "suspectIntervals": intervals,
                    "oneViewIntervals": one_view_intervals,
                    "suspectNow": any(
                        int(item["left_start_frame"]) <= pair[0] <= int(item["left_end_frame"])
                        for item in intervals
                    )}
            for side, frame_number in zip(("left", "right"), pair):
                record = tracks[side].get(frame_number, {}).get(obj_id)
                if record is None and setup_node.get("status") != "absent" and not tracks[side]:
                    point = setup_uv[side].get(obj_id)
                    record = None if point is None else {"u": point[0], "v": point[1],
                        "quality": None, "method": "setup_check", "status": "measured", "valid": True}
                if record is None and setup_node.get("status") == "absent":
                    point = setup_uv[side].get(obj_id)
                    record = None if point is None else {"u": point[0], "v": point[1],
                        "quality": None, "method": "absent", "status": "absent", "valid": False}
                if record is not None:
                    sx, sy = scales[side]
                    record = dict(record, uRaw=record["u"], vRaw=record["v"],
                                  u=record["u"] * sx, v=record["v"] * sy)
                node[side] = record
            nodes.append(node)
        payload_frames.append({"leftFrame": pair[0], "rightFrame": pair[1],
                               "sourceLeftFrame": start_frame + pair[0],
                               "sourceRightFrame": start_frame + pair[1],
                               "views": view_payload, "nodes": nodes})

    data = {"title": "Stereo marker review", "frames": payload_frames}
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    html = template.replace("__REVIEW_DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    output = run_dir / "marker_review.html"
    output.write_text(html, encoding="utf-8")
    return output, sampled_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a self-contained stereo marker review HTML")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--left-video", required=True)
    parser.add_argument("--right-video", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--image-scale", type=float, default=0.5)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output, pairs = generate_review(args.run_dir, args.left_video, args.right_video,
                                    args.start_frame, frames=args.frames, image_scale=args.image_scale)
    print(f"[OK] Wrote marker review: {output} ({len(pairs)} sampled pairs: {pairs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
