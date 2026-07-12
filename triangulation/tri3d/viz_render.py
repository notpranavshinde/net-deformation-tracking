"""Multiprocessing workers for rendered isometric and top-down scene panels."""

import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .viz_encode import (
    _concat_segments,
    _has_ffmpeg,
    _open_writer,
    _writer_close,
    _writer_write,
)

VIZ_CMAP = "viridis"
_WORKER = {}


def _worker_init(kind, w, h, x_lim, y_lim, z_lim, z_min, z_max, disp_max, iso_azim, edges):
    """Initialize a reusable matplotlib figure inside each worker process."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dpi = 100
    disp_max = max(float(disp_max), 1e-9)
    norm = colors.Normalize(vmin=0.0, vmax=disp_max)
    cmap = plt.get_cmap(VIZ_CMAP)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="white")
    if kind == "iso":
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        x_lo, x_hi = x_lim
        y_lo, y_hi = y_lim
        z_lo, z_hi = z_lim
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(z_lo, z_hi)
        ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
        ax.set_xlabel("x (m)", color="black", fontsize=22, labelpad=18)
        ax.set_ylabel("y (m)", color="black", fontsize=22, labelpad=18)
        ax.set_zlabel("z (m)", color="black", fontsize=22, labelpad=18)
        ax.tick_params(colors="black", labelsize=18)
        ax.view_init(elev=30, azim=float(iso_azim))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 1))
            axis.pane.set_edgecolor((0.75, 0.75, 0.75, 1))
            axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 1)
            axis._axinfo["grid"]["linewidth"] = 1.0
        ax.grid(True)
        yy, zz = np.meshgrid(
            np.linspace(y_lo, y_hi, 2),
            np.linspace(z_lo, z_hi, 2),
        )
        xx = np.zeros_like(yy)
        ax.plot_surface(
            xx, yy, zz,
            color=(0.86, 0.86, 0.86, 0.25),
            edgecolor=(0.50, 0.50, 0.50, 0.80),
            linewidth=1.0,
            shade=False,
            alpha=0.25,
            zorder=0,
        )
        edge_lines = [
            ax.plot([], [], [], color=(0.18, 0.18, 0.18, 0.90), linewidth=1.35)[0]
            for _edge in edges
        ]
        scatter = ax.scatter([], [], [], s=88, c=[], cmap=cmap, norm=norm,
                             depthshade=False, edgecolors="black", linewidths=0.25)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.05)
        cbar.set_label("|Δpos from t0| (m)", color="black", fontsize=18)
        cbar.ax.tick_params(colors="black", labelsize=14)
        fig.subplots_adjust(left=0.02, right=0.94, top=0.98, bottom=0.04)
    else:
        ax = fig.add_subplot(111, facecolor="white")
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("")
        ax.set_ylabel("z (m)", color="black", fontsize=24, labelpad=14)
        ax.xaxis.tick_top()
        ax.text(0.0, 1.08, "x (m)", transform=ax.transAxes,
                color="black", fontsize=26, ha="left", va="bottom")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(colors="black", labelsize=20, width=1.5, length=7)
        ax.grid(True, color=(0.78, 0.78, 0.78), linestyle="-", linewidth=1.0)
        ax.axvline(0.0, color=(0.45, 0.45, 0.45), linestyle="--", linewidth=1.5)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)
        edge_lines = [
            ax.plot([], [], color=(0.18, 0.18, 0.18, 0.90), linewidth=1.35)[0]
            for _edge in edges
        ]
        scatter = ax.scatter([], [], s=88, c=[], cmap=cmap, norm=norm,
                             edgecolors="black", linewidths=0.25)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)
        cbar.set_label("|Δpos from t0| (m)", color="black", fontsize=18)
        cbar.ax.tick_params(colors="black", labelsize=14)
        fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)

    _WORKER.update(dict(
        kind=kind, fig=fig, ax=ax, scatter=scatter,
        w=w, h=h, cmap=cmap, norm=norm, edges=edges, edge_lines=edge_lines,
    ))


def _draw_chunk(args):
    kind, frame_ids, chunk_xyz, fps, segment_path, encoder = args
    fig = _WORKER["fig"]
    scatter = _WORKER["scatter"]
    w = _WORKER["w"]
    h = _WORKER["h"]
    cmap = _WORKER["cmap"]
    norm = _WORKER["norm"]
    edges = _WORKER["edges"]
    edge_lines = _WORKER["edge_lines"]

    writer = _open_writer(segment_path, fps, w, h, encoder)

    for fid, (ObjIds, Xs, Ys, Zs, Ds) in zip(frame_ids, chunk_xyz):
        colors = cmap(norm(Ds)) if len(Ds) > 0 else np.zeros((0, 4), dtype=np.float32)
        index_by_obj = {int(obj_id): i for i, obj_id in enumerate(ObjIds)}
        if kind == "iso":
            for line, (obj_a, obj_b) in zip(edge_lines, edges):
                ia = index_by_obj.get(int(obj_a))
                ib = index_by_obj.get(int(obj_b))
                if ia is None or ib is None:
                    line.set_data([], [])
                    line.set_3d_properties([])
                else:
                    line.set_data([Xs[ia], Xs[ib]], [Ys[ia], Ys[ib]])
                    line.set_3d_properties([Zs[ia], Zs[ib]])
            scatter._offsets3d = (Xs, Ys, Zs)
            scatter.set_array(np.asarray(Ds, dtype=np.float64))
            scatter.set_facecolor(colors)
            scatter.set_edgecolor(colors)
        else:
            for line, (obj_a, obj_b) in zip(edge_lines, edges):
                ia = index_by_obj.get(int(obj_a))
                ib = index_by_obj.get(int(obj_b))
                if ia is None or ib is None:
                    line.set_data([], [])
                else:
                    line.set_data([Xs[ia], Xs[ib]], [Zs[ia], Zs[ib]])
            if len(Xs) > 0:
                scatter.set_offsets(np.column_stack([Xs, Zs]))
            else:
                scatter.set_offsets(np.zeros((0, 2)))
            scatter.set_array(np.asarray(Ds, dtype=np.float64))
            scatter.set_facecolor(colors)
            scatter.set_edgecolor(colors)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        cw, ch = fig.canvas.get_width_height()
        img = buf.reshape(ch, cw, 4)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))

        title = "Local net frame: out-of-plane x" if kind == "iso" else "Top-down: x deformation vs z"
        cv2.putText(img, f"{title}  frame={fid}", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        _writer_write(writer, img)

    _writer_close(writer)
    return segment_path


def _render_panel_parallel(kind, frame_ids, per_frame_xyz, fps, out_path,
                           w, h, x_lim, y_lim, z_lim, z_min, z_max,
                           workers, encoder, disp_max, iso_azim, edges):
    n = len(frame_ids)
    if n == 0:
        return
    workers = max(1, min(workers, n))
    chunk_size = (n + workers - 1) // workers
    chunks = []
    tmpdir = Path(tempfile.mkdtemp(prefix=f"viz_{kind}_"))
    for i in range(workers):
        s = i * chunk_size
        e = min(s + chunk_size, n)
        if s >= e:
            break
        seg = tmpdir / f"seg_{i:04d}.mp4"
        chunks.append((kind, frame_ids[s:e], per_frame_xyz[s:e], fps, str(seg), encoder))

    if workers == 1:
        _worker_init(kind, w, h, x_lim, y_lim, z_lim, z_min, z_max, disp_max, iso_azim, edges)
        segments = [_draw_chunk(c) for c in chunks]
    else:
        with Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(kind, w, h, x_lim, y_lim, z_lim, z_min, z_max, disp_max, iso_azim, edges),
        ) as pool:
            segments = pool.map(_draw_chunk, chunks)

    if not _has_ffmpeg() and len(segments) > 1:
        print("[VIS] ffmpeg not found; concatenating via cv2 (slower).")
        first = cv2.VideoCapture(segments[0])
        fps_v = first.get(cv2.CAP_PROP_FPS) or fps
        first.release()
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))
        for seg in segments:
            cap = cv2.VideoCapture(seg)
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                vw.write(frm)
            cap.release()
            Path(seg).unlink(missing_ok=True)
        vw.release()
    else:
        _concat_segments(segments, str(out_path))

    shutil.rmtree(tmpdir, ignore_errors=True)
