"""Create legacy and publication-oriented triangulation visualizations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .scene_geometry import (
    _attach_displacements,
    _build_net_reference_frame,
    _build_reference_edges,
    _choose_iso_azim,
    _to_local_xyz,
)
from .viewer_html import _write_threejs_viewer
from .viz_draw import color_for_obj, depth_to_bgr, draw_value_colorbar
from .viz_encode import (
    _has_nvenc,
    _open_writer,
    _visualization_fps,
    _writer_close,
    _writer_write,
    open_video,
)
from .viz_render import _render_panel_parallel


def create_3d_visualization_video(out_rows: list[dict[str, Any]],
                                  left_video: str,
                                  out_path: str,
                                  max_frames: int = -1,
                                  source_start_frame: int = 0) -> None:
    if len(out_rows) == 0:
        print("[VIS] No rows to visualize. Skipping.")
        return

    capL = open_video(left_video)
    capL.set(cv2.CAP_PROP_POS_FRAMES, int(source_start_frame))
    fpsL = float(capL.get(cv2.CAP_PROP_FPS))
    fps = _visualization_fps(fpsL)

    grouped = {}
    for row in out_rows:
        f = int(row["frame_L"])
        grouped.setdefault(f, []).append(row)

    target_frames = sorted(grouped.keys())
    if max_frames > 0:
        target_frames = target_frames[:max_frames]
    if len(target_frames) == 0:
        capL.release()
        print("[VIS] No frames selected for visualization. Skipping.")
        return

    ok, first_frame = capL.read()
    if not ok:
        capL.release()
        raise RuntimeError("[VIS] Could not read first frame from left video.")

    h, w_single = first_frame.shape[:2]
    w = w_single * 2
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    current_idx = 0
    frame = first_frame
    target_ptr = 0
    next_target = int(target_frames[target_ptr])

    while True:
        if current_idx == next_target:
            rows = grouped[next_target]
            vis_orig = frame.copy()
            vis_csv = np.zeros_like(vis_orig)

            valid_count = 0
            for r in rows:
                obj_id = int(r["obj_id"])
                col = color_for_obj(obj_id)
                valid = int(r["valid_3d"]) == 1
                if valid:
                    valid_count += 1

                u = int(round(float(r["uL"])))
                v = int(round(float(r["vL"])))
                if u < 0 or v < 0 or u >= w_single or v >= h:
                    continue

                draw_col = col if valid else (120, 120, 120)
                r_size = 5 if valid else 3
                cv2.circle(vis_csv, (u, v), r_size, draw_col, -1)
                cv2.circle(vis_orig, (u, v), 3, draw_col, -1)

            cv2.putText(vis_orig, "Original (Left)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_csv, "CSV Reconstruction", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            both = np.hstack([vis_orig, vis_csv])
            cv2.putText(both, f"frame={next_target}  active_nodes={len(rows)}  valid_3d={valid_count}",
                        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            writer.write(both)

            target_ptr += 1
            if target_ptr >= len(target_frames):
                break
            next_target = int(target_frames[target_ptr])

            if target_ptr % max(1, len(target_frames) // 10) == 0:
                print(f"[VIS] rendered {target_ptr}/{len(target_frames)} frames")

        ok, frame = capL.read()
        if not ok:
            break
        current_idx += 1

        if current_idx > int(target_frames[-1]):
            break

    writer.release()
    capL.release()
    print(f"[OK] Wrote visualization video: {out_path}")


def create_3d_scene_visualization(out_rows: list[dict[str, Any]],
                                  left_video: str,
                                  out_path: str,
                                  max_frames: int = -1,
                                  trail_len: int = 60,
                                  workers: int = 0,
                                  encoder: str = "auto",
                                  grid_cols: int = 0,
                                  grid_rows: int = 0,
                                  source_start_frame: int = 0,
                                  viewer_only: bool = False) -> None:
    """Render left overlay, isometric 3D, and top-down videos as separate files."""
    _ = trail_len
    if len(out_rows) == 0:
        print("[VIS] No rows to visualize. Skipping.")
        return

    valid_rows = [r for r in out_rows if int(r["valid_3d"]) == 1]
    if not valid_rows:
        if viewer_only:
            print("[VIS] No valid 3D rows. Skipping interactive viewer.")
            return
        print("[VIS] No valid 3D rows. Falling back to side-by-side viz.")
        create_3d_visualization_video(
            out_rows, left_video, out_path, max_frames, source_start_frame
        )
        return

    refs, fixed_ids, disp_max = _attach_displacements(valid_rows)
    origin, basis, ref_local, fixed_local = _build_net_reference_frame(refs, fixed_ids)
    edges, grid_shape = _build_reference_edges(refs, ref_local, grid_cols=grid_cols, grid_rows=grid_rows)
    iso_azim = _choose_iso_azim(ref_local, fixed_local)
    print(
        f"[VIS] displacement color scale: 0-{disp_max:.4f} m; "
        f"fixed_ids={len(fixed_ids)}; grid={grid_shape[0]}x{grid_shape[1]}; "
        f"edges={len(edges)}; iso_azim={iso_azim}"
    )

    local_all = np.array([
        _to_local_xyz([float(r["X"]), float(r["Y"]), float(r["Z"])], origin, basis)
        for r in valid_rows
    ], dtype=np.float64)
    Xs_all = local_all[:, 0]
    Ys_all = local_all[:, 1]
    Zs_all = local_all[:, 2]
    z_min, z_max = float(np.percentile(Zs_all, 2)), float(np.percentile(Zs_all, 98))
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0

    def pad_lim(arr):
        lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        pad = 0.1 * (hi - lo + 1e-6)
        return (lo - pad, hi + pad)

    def include_zero(lim):
        lo, hi = lim
        return (min(lo, 0.0), max(hi, 0.0))

    x_lim = include_zero(pad_lim(Xs_all))
    y_lim = pad_lim(Ys_all)
    z_lim = pad_lim(Zs_all)

    grouped = {}
    for row in out_rows:
        grouped.setdefault(int(row["frame_L"]), []).append(row)

    target_frames = sorted(grouped.keys())
    if max_frames > 0:
        target_frames = target_frames[:max_frames]
    if not target_frames:
        print("[VIS] No frames selected. Skipping.")
        return

    base = Path(out_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.with_suffix("")
    viewer_path = Path(f"{stem}_viewer.html")

    if viewer_only:
        per_frame_xyz = []
        for frame_idx in target_frames:
            valid_here = [r for r in grouped[frame_idx] if int(r["valid_3d"]) == 1]
            if valid_here:
                local = np.array([
                    _to_local_xyz([float(r["X"]), float(r["Y"]), float(r["Z"])], origin, basis)
                    for r in valid_here
                ], dtype=np.float64)
                obj_ids = np.array([int(r["obj_id"]) for r in valid_here], dtype=np.int32)
                displacements = np.array(
                    [float(r.get("_disp_m", 0.0)) for r in valid_here], dtype=np.float64
                )
                per_frame_xyz.append(
                    (obj_ids, local[:, 0], local[:, 1], local[:, 2], displacements)
                )
            else:
                per_frame_xyz.append((
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0),
                    np.zeros(0),
                    np.zeros(0),
                    np.zeros(0),
                ))
        _write_threejs_viewer(
            viewer_path,
            target_frames,
            per_frame_xyz,
            x_lim,
            y_lim,
            z_lim,
            disp_max,
            edges,
            iso_azim,
            grid_shape,
        )
        return

    capL = open_video(left_video)
    capL.set(cv2.CAP_PROP_POS_FRAMES, int(source_start_frame))
    fpsL = float(capL.get(cv2.CAP_PROP_FPS))
    fps = _visualization_fps(fpsL)

    ok, first_frame = capL.read()
    if not ok:
        capL.release()
        raise RuntimeError("[VIS] Could not read first frame from left video.")

    h, w_single = first_frame.shape[:2]

    left_path = Path(f"{stem}_left.mp4")
    iso_path = Path(f"{stem}_iso.mp4")
    top_path = Path(f"{stem}_topdown.mp4")

    if encoder == "auto":
        enc = "nvenc" if _has_nvenc() else "mp4v"
    else:
        enc = encoder
    if enc == "nvenc" and not _has_nvenc():
        print("[VIS] nvenc requested but not usable on this machine; falling back to mp4v.")
        enc = "mp4v"
    print(f"[VIS] encoder={enc}")

    if workers <= 0:
        workers = min(os.cpu_count() or 8, len(target_frames))
    print(f"[VIS] workers={workers}, frames={len(target_frames)}")

    writer_left = _open_writer(left_path, fps, w_single, h, enc)
    per_frame_xyz = []

    current_idx = 0
    frame = first_frame
    target_ptr = 0
    next_target = int(target_frames[target_ptr])

    while True:
        if current_idx == next_target:
            rows = grouped[next_target]
            valid_here = [r for r in rows if int(r["valid_3d"]) == 1]

            vis_left = frame.copy()
            for r in rows:
                u = int(round(float(r["uL"])))
                v = int(round(float(r["vL"])))
                if u < 0 or v < 0 or u >= w_single or v >= h:
                    continue
                valid = int(r["valid_3d"]) == 1
                if valid:
                    disp = float(r.get("_disp_m", 0.0))
                    col = depth_to_bgr(disp, 0.0, disp_max)
                    cv2.circle(vis_left, (u, v), 10, col, -1)
                    cv2.circle(vis_left, (u, v), 11, (255, 255, 255), 2)
                    cv2.putText(vis_left, f"{disp:.3f}", (u + 12, v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                else:
                    cv2.circle(vis_left, (u, v), 4, (120, 120, 120), -1)

            cv2.putText(vis_left, "Left view (displacement-colored)", (24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis_left,
                        f"frame={next_target}  nodes={len(rows)}  valid={len(valid_here)}",
                        (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            draw_value_colorbar(vis_left, 0.0, disp_max, x=w_single - 130, y=110, w=28, h=260, label="|d| m")
            _writer_write(writer_left, vis_left)

            if valid_here:
                local = np.array([
                    _to_local_xyz([float(r["X"]), float(r["Y"]), float(r["Z"])], origin, basis)
                    for r in valid_here
                ], dtype=np.float64)
                ObjIds = np.array([int(r["obj_id"]) for r in valid_here], dtype=np.int32)
                Xs = local[:, 0]
                Ys = local[:, 1]
                Zs = local[:, 2]
                Ds = np.array([float(r.get("_disp_m", 0.0)) for r in valid_here], dtype=np.float64)
            else:
                ObjIds = np.zeros(0, dtype=np.int32)
                Xs = np.zeros(0)
                Ys = np.zeros(0)
                Zs = np.zeros(0)
                Ds = np.zeros(0)
            per_frame_xyz.append((ObjIds, Xs, Ys, Zs, Ds))

            target_ptr += 1
            if target_ptr >= len(target_frames):
                break
            next_target = int(target_frames[target_ptr])
            if target_ptr % max(1, len(target_frames) // 10) == 0:
                print(f"[VIS] left pass: {target_ptr}/{len(target_frames)}")

        ok, frame = capL.read()
        if not ok:
            break
        current_idx += 1
        if current_idx > int(target_frames[-1]):
            break

    _writer_close(writer_left)
    capL.release()
    print(f"[OK] Wrote {left_path}")

    print("[VIS] rendering iso panel (parallel)...")
    _render_panel_parallel("iso", target_frames, per_frame_xyz, fps, iso_path,
                           w_single, h, x_lim, y_lim, z_lim, z_min, z_max,
                           workers=workers, encoder=enc, disp_max=disp_max, iso_azim=iso_azim, edges=edges)
    print(f"[OK] Wrote {iso_path}")

    print("[VIS] rendering topdown panel (parallel)...")
    _render_panel_parallel("topdown", target_frames, per_frame_xyz, fps, top_path,
                           w_single, h, x_lim, y_lim, z_lim, z_min, z_max,
                           workers=workers, encoder=enc, disp_max=disp_max, iso_azim=iso_azim, edges=edges)
    print(f"[OK] Wrote {top_path}")

    _write_threejs_viewer(
        viewer_path,
        target_frames,
        per_frame_xyz,
        x_lim,
        y_lim,
        z_lim,
        disp_max,
        edges,
        iso_azim,
        grid_shape,
    )


def visualize_existing_rows(
    args: argparse.Namespace, out_rows: list[dict[str, Any]]
) -> None:
    if not args.left_video:
        print("[VIS] --visualize requested but --left-video not provided. Skipping visualization.")
        return
    if args.viz_mode == "scene":
        create_3d_scene_visualization(
            out_rows=out_rows,
            left_video=args.left_video,
            out_path=args.viz_out,
            max_frames=args.viz_max_frames,
            trail_len=args.viz_trail,
            workers=args.workers,
            encoder=args.viz_encoder,
            grid_cols=args.viz_grid_cols,
            grid_rows=args.viz_grid_rows,
            source_start_frame=args.start_frame,
            viewer_only=args.viewer_only,
        )
    else:
        create_3d_visualization_video(
            out_rows=out_rows,
            left_video=args.left_video,
            out_path=args.viz_out,
            max_frames=args.viz_max_frames,
            source_start_frame=args.start_frame,
        )
