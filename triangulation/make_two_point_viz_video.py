import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


ERROR_CANDIDATES = [
    "err_3d",
    "err_xyz",
    "error_3d",
    "err",
    "err_px",
]


def _pick_error_column(df: pd.DataFrame) -> str | None:
    for col in ("X_err", "Y_err", "Z_err"):
        if col in df.columns:
            return None
    for col in ERROR_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _prepare_pair_data(df: pd.DataFrame, obj_ids: tuple[int, int], frame_col: str) -> tuple[pd.DataFrame, str | None, bool]:
    subset = df[df["obj_id"].isin(obj_ids)].copy()

    cols = [frame_col, "obj_id", "X", "Y", "Z"]
    axis_err_available = all(col in subset.columns for col in ("X_err", "Y_err", "Z_err"))
    scalar_err_col: str | None = None
    if axis_err_available:
        cols += ["X_err", "Y_err", "Z_err"]
    else:
        err_col = _pick_error_column(subset)
        if err_col is not None:
            scalar_err_col = err_col
            cols.append(err_col)

    subset = subset[cols].dropna()

    pieces = []
    for obj in obj_ids:
        obj_df = subset[subset["obj_id"] == obj].copy()
        rename_map = {"X": f"X_{obj}", "Y": f"Y_{obj}", "Z": f"Z_{obj}"}
        if axis_err_available:
            rename_map.update(
                {
                    "X_err": f"X_err_{obj}",
                    "Y_err": f"Y_err_{obj}",
                    "Z_err": f"Z_err_{obj}",
                }
            )
        elif scalar_err_col is not None:
            rename_map[scalar_err_col] = f"err_{obj}"

        obj_df = obj_df.rename(columns=rename_map).drop(columns=["obj_id"])
        pieces.append(obj_df)

    merged = pieces[0].merge(pieces[1], on=frame_col, how="inner").sort_values(frame_col)
    if merged.empty:
        raise ValueError("No overlapping frames found for the selected two obj_id values.")

    dx = merged[f"X_{obj_ids[1]}"] - merged[f"X_{obj_ids[0]}"]
    dy = merged[f"Y_{obj_ids[1]}"] - merged[f"Y_{obj_ids[0]}"]
    dz = merged[f"Z_{obj_ids[1]}"] - merged[f"Z_{obj_ids[0]}"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    merged["distance"] = dist

    if axis_err_available:
        sigma_dx = np.sqrt(merged[f"X_err_{obj_ids[0]}"] ** 2 + merged[f"X_err_{obj_ids[1]}"] ** 2)
        sigma_dy = np.sqrt(merged[f"Y_err_{obj_ids[0]}"] ** 2 + merged[f"Y_err_{obj_ids[1]}"] ** 2)
        sigma_dz = np.sqrt(merged[f"Z_err_{obj_ids[0]}"] ** 2 + merged[f"Z_err_{obj_ids[1]}"] ** 2)
    elif scalar_err_col is not None and f"err_{obj_ids[0]}" in merged.columns and f"err_{obj_ids[1]}" in merged.columns:
        # Scalar reprojection error (typically px) should not be treated as XYZ error bands.
        # Keep distance error as unavailable unless real XYZ error columns exist.
        sigma_dx = sigma_dy = sigma_dz = np.zeros(len(merged), dtype=float)
    else:
        sigma_dx = sigma_dy = sigma_dz = np.zeros(len(merged), dtype=float)

    safe_dist = np.where(dist > 1e-12, dist, 1e-12)
    sigma_dist = np.sqrt((dx / safe_dist) ** 2 * sigma_dx**2 + (dy / safe_dist) ** 2 * sigma_dy**2 + (dz / safe_dist) ** 2 * sigma_dz**2)
    merged["distance_err"] = sigma_dist

    return merged, scalar_err_col, axis_err_available


def _axis_limits(values_a: np.ndarray, values_b: np.ndarray, err_a: np.ndarray, err_b: np.ndarray) -> tuple[float, float]:
    lo = min(np.min(values_a - err_a), np.min(values_b - err_b))
    hi = max(np.max(values_a + err_a), np.max(values_b + err_b))
    span = hi - lo
    pad = 0.08 * span if span > 0 else 1.0
    return lo - pad, hi + pad


def _extract_err(merged: pd.DataFrame, obj: int, axis: str) -> np.ndarray:
    axis_key = f"{axis}_err_{obj}"
    if axis_key in merged.columns:
        return merged[axis_key].to_numpy(dtype=float)

    return np.zeros(len(merged), dtype=float)


def make_video(
    csv_path: Path,
    output_path: Path,
    fps: int,
    dpi: int,
    point_a: int | None,
    point_b: int | None,
    coord_scale: float,
    coord_unit: str,
) -> None:
    df = pd.read_csv(csv_path)

    required = {"obj_id", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame_col = "gframe" if "gframe" in df.columns else "frame"
    if frame_col not in df.columns:
        raise ValueError("CSV must contain `gframe` (preferred) or `frame` column.")

    if point_a is None or point_b is None:
        obj_ids = sorted(df["obj_id"].dropna().unique().tolist())
        if len(obj_ids) < 2:
            raise ValueError("Need at least two `obj_id` values in CSV.")
        chosen = (int(obj_ids[0]), int(obj_ids[1]))
    else:
        chosen = (int(point_a), int(point_b))

    merged, scalar_err_col, axis_err_available = _prepare_pair_data(df, chosen, frame_col)

    frames = merged[frame_col].to_numpy()
    x0 = merged[f"X_{chosen[0]}"].to_numpy(dtype=float) * coord_scale
    y0 = merged[f"Y_{chosen[0]}"].to_numpy(dtype=float) * coord_scale
    z0 = merged[f"Z_{chosen[0]}"].to_numpy(dtype=float) * coord_scale
    x1 = merged[f"X_{chosen[1]}"].to_numpy(dtype=float) * coord_scale
    y1 = merged[f"Y_{chosen[1]}"].to_numpy(dtype=float) * coord_scale
    z1 = merged[f"Z_{chosen[1]}"].to_numpy(dtype=float) * coord_scale

    ex0 = _extract_err(merged, chosen[0], "X")
    ey0 = _extract_err(merged, chosen[0], "Y")
    ez0 = _extract_err(merged, chosen[0], "Z")
    ex1 = _extract_err(merged, chosen[1], "X")
    ey1 = _extract_err(merged, chosen[1], "Y")
    ez1 = _extract_err(merged, chosen[1], "Z")

    d = merged["distance"].to_numpy(dtype=float) * coord_scale
    de = merged["distance_err"].to_numpy(dtype=float) * coord_scale

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_x, ax_y, ax_z, ax_d = axes.ravel()

    for axis_plot, name in ((ax_x, "X"), (ax_y, "Y"), (ax_z, "Z")):
        axis_plot.set_title(f"{name} coordinate ± error")
        axis_plot.set_xlabel("Frame")
        axis_plot.set_ylabel(f"{name} ({coord_unit})")
        axis_plot.grid(True, alpha=0.3)

    ax_d.set_title("Distance between points ± propagated error")
    ax_d.set_xlabel("Frame")
    ax_d.set_ylabel(f"Distance ({coord_unit})")
    ax_d.grid(True, alpha=0.3)

    x_lim = (frames.min(), frames.max())
    for axis_plot in (ax_x, ax_y, ax_z, ax_d):
        axis_plot.set_xlim(*x_lim)

    ax_x.set_ylim(*_axis_limits(x0, x1, ex0, ex1))
    ax_y.set_ylim(*_axis_limits(y0, y1, ey0, ey1))
    ax_z.set_ylim(*_axis_limits(z0, z1, ez0, ez1))
    d_span = max(np.max(d + de) - np.min(d - de), 1e-9)
    ax_d.set_ylim(np.min(d - de) - 0.08 * d_span, np.max(d + de) + 0.08 * d_span)

    # Static backgrounds
    if axis_err_available:
        ex0_s = ex0 * coord_scale
        ey0_s = ey0 * coord_scale
        ez0_s = ez0 * coord_scale
        ex1_s = ex1 * coord_scale
        ey1_s = ey1 * coord_scale
        ez1_s = ez1 * coord_scale
        ax_x.fill_between(frames, x0 - ex0_s, x0 + ex0_s, color="tab:blue", alpha=0.15)
        ax_x.fill_between(frames, x1 - ex1_s, x1 + ex1_s, color="tab:orange", alpha=0.15)
        ax_y.fill_between(frames, y0 - ey0_s, y0 + ey0_s, color="tab:blue", alpha=0.15)
        ax_y.fill_between(frames, y1 - ey1_s, y1 + ey1_s, color="tab:orange", alpha=0.15)
        ax_z.fill_between(frames, z0 - ez0_s, z0 + ez0_s, color="tab:blue", alpha=0.15)
        ax_z.fill_between(frames, z1 - ez1_s, z1 + ez1_s, color="tab:orange", alpha=0.15)
        ax_d.fill_between(frames, d - de, d + de, color="tab:green", alpha=0.18)

    # Animated line artists
    x0_line, = ax_x.plot([], [], color="tab:blue", lw=2, label=f"obj {chosen[0]}")
    x1_line, = ax_x.plot([], [], color="tab:orange", lw=2, label=f"obj {chosen[1]}")
    y0_line, = ax_y.plot([], [], color="tab:blue", lw=2, label=f"obj {chosen[0]}")
    y1_line, = ax_y.plot([], [], color="tab:orange", lw=2, label=f"obj {chosen[1]}")
    z0_line, = ax_z.plot([], [], color="tab:blue", lw=2, label=f"obj {chosen[0]}")
    z1_line, = ax_z.plot([], [], color="tab:orange", lw=2, label=f"obj {chosen[1]}")
    d_line, = ax_d.plot([], [], color="tab:green", lw=2, label="distance")

    for axis_plot in (ax_x, ax_y, ax_z, ax_d):
        axis_plot.axvline(frames[0], color="k", ls="--", lw=1, alpha=0.45)

    frame_cursor_x = ax_x.lines[-1]
    frame_cursor_y = ax_y.lines[-1]
    frame_cursor_z = ax_z.lines[-1]
    frame_cursor_d = ax_d.lines[-1]

    ax_x.legend(loc="upper left")
    ax_y.legend(loc="upper left")
    ax_z.legend(loc="upper left")
    ax_d.legend(loc="upper left")

    title = fig.suptitle("", fontsize=11)

    def _fmt_point(idx: int, obj: int, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> str:
        if axis_err_available:
            return (
                f"obj {obj}: "
                f"X={xx[idx]:.3f}±{(ex[idx] * coord_scale):.3f}, "
                f"Y={yy[idx]:.3f}±{(ey[idx] * coord_scale):.3f}, "
                f"Z={zz[idx]:.3f}±{(ez[idx] * coord_scale):.3f} {coord_unit}"
            )
        return (
            f"obj {obj}: "
            f"X={xx[idx]:.3f}, "
            f"Y={yy[idx]:.3f}, "
            f"Z={zz[idx]:.3f} {coord_unit}"
        )

    def update(i: int):
        fr = frames[: i + 1]

        x0_line.set_data(fr, x0[: i + 1])
        x1_line.set_data(fr, x1[: i + 1])
        y0_line.set_data(fr, y0[: i + 1])
        y1_line.set_data(fr, y1[: i + 1])
        z0_line.set_data(fr, z0[: i + 1])
        z1_line.set_data(fr, z1[: i + 1])
        d_line.set_data(fr, d[: i + 1])

        frame_cursor_x.set_xdata([frames[i], frames[i]])
        frame_cursor_y.set_xdata([frames[i], frames[i]])
        frame_cursor_z.set_xdata([frames[i], frames[i]])
        frame_cursor_d.set_xdata([frames[i], frames[i]])

        p0 = _fmt_point(i, chosen[0], x0, y0, z0, ex0, ey0, ez0)
        p1 = _fmt_point(i, chosen[1], x1, y1, z1, ex1, ey1, ez1)
        if axis_err_available:
            title.set_text(
                f"Frame {int(frames[i])} | {p0} | {p1} | "
                f"distance={d[i]:.3f}±{de[i]:.3f} {coord_unit}"
            )
        else:
            px_note = ""
            if scalar_err_col is not None and f"err_{chosen[0]}" in merged.columns and f"err_{chosen[1]}" in merged.columns:
                px_note = (
                    f" | {scalar_err_col}: "
                    f"obj {chosen[0]}={merged[f'err_{chosen[0]}'].iat[i]:.2f}, "
                    f"obj {chosen[1]}={merged[f'err_{chosen[1]}'].iat[i]:.2f}"
                )
            title.set_text(
                f"Frame {int(frames[i])} | {p0} | {p1} | "
                f"distance={d[i]:.3f} {coord_unit}{px_note}"
            )

        return (
            x0_line,
            x1_line,
            y0_line,
            y1_line,
            z0_line,
            z1_line,
            d_line,
            frame_cursor_x,
            frame_cursor_y,
            frame_cursor_z,
            frame_cursor_d,
            title,
        )

    anim = FuncAnimation(fig, update, frames=len(merged), interval=1000 / max(fps, 1), blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a frame-by-frame video for two tracked 3D points with coordinate error and inter-point distance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("triangulation/out/triangulated_3d.csv"),
        help="Input triangulated CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("triangulation/out/triangulated_3d_viz.mp4"),
        help="Output video path (mp4).",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--dpi", type=int, default=130, help="Render DPI.")
    parser.add_argument("--point-a", type=int, default=None, help="First obj_id to visualize.")
    parser.add_argument("--point-b", type=int, default=None, help="Second obj_id to visualize.")
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1000.0,
        help="Scale factor applied to X/Y/Z and distance for display (default 1000 -> meters to millimeters).",
    )
    parser.add_argument(
        "--coord-unit",
        type=str,
        default="mm",
        help="Displayed coordinate/distance unit label.",
    )

    args = parser.parse_args()
    make_video(
        args.input,
        args.output,
        args.fps,
        args.dpi,
        args.point_a,
        args.point_b,
        args.coord_scale,
        args.coord_unit,
    )
    print(f"Saved: {args.output}")
