"""Split a multi-segment wave run into its segments and measure the net's swing in each.

The run label lists the segments in order, e.g. A_mm_32_64_64_127_F_Hz_0.5_0.5_0.25_0.25 is
+-32 mm at 0.5 Hz, +-64 mm at 0.5 Hz, +-64 mm at 0.25 Hz, +-127 mm at 0.25 Hz. Each segment is
`--cycles` steady cycles plus a ramp-up and a ramp-down cycle, with a still dwell in between.

Segments are found from the data (the carriage runs up to ~0.6 s off its schedule, and the
video start is not tied to the carriage): the swing envelope of the bottom row splits the run
at the dwells. Only the steady cycles are measured: the middle `--cycles` periods of each
segment, less a quarter period at each end.

The cameras ride on the carriage, so every number is relative to the carriage. Per knot, the
swing is the amplitude (+- mm) of a sinusoid fitted at the segment's measured frequency, along
the segment's main direction of motion.

    python experiments/cotracker3/wave_segments.py triangulation/results/<run>_cotracker3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FPS = 29.97
GRID = 12
COLORS = ["#2a78d6", "#1baf7a", "#e87ba4", "#008300"]  # blue, aqua, magenta, green


def segments_from_label(name: str) -> list[tuple[float, float]]:
    m = re.search(r"A_mm_([\d._]+?)_F_Hz_([\d._]+?)(?:_cotracker3)?$", name)
    if not m:
        raise SystemExit(f"Cannot read segments from `{name}`; expected A_mm_<a1>_<a2>..._F_Hz_<f1>_<f2>...")
    amps = [float(a) for a in m.group(1).split("_")]
    freqs = [float(f) for f in m.group(2).split("_")]
    if len(amps) != len(freqs):
        raise SystemExit(f"{len(amps)} amplitudes but {len(freqs)} frequencies in `{name}`")
    return list(zip(amps, freqs))


def load_positions(result: Path) -> np.ndarray:
    df = pd.read_csv(result / "triangulated_3d.csv", usecols=["gframe", "obj_id", "X", "Y", "Z", "valid_3d"])
    ok = df[df["valid_3d"] == 1]
    p = np.full((int(df["gframe"].max()) + 1, GRID * GRID, 3), np.nan)
    p[ok["gframe"].to_numpy(), ok["obj_id"].to_numpy()] = ok[["X", "Y", "Z"]].to_numpy()
    return p * 1000  # mm


def main_axis(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit direction of most motion of an (T, 3) path, and the share of motion along it."""
    x = pd.DataFrame(xyz).interpolate().bfill().ffill().to_numpy()
    _, sv, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    axis = vt[0] if vt[0][np.argmax(np.abs(vt[0]))] > 0 else -vt[0]
    return axis, float(sv[0] ** 2 / (sv ** 2).sum())


def find_segments(signal: np.ndarray, plan: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Frame ranges where the bottom row is swinging, one per planned segment, in order."""
    s = pd.Series(signal - np.median(signal))
    rms = s.pow(2).rolling(int(2 * FPS), center=True, min_periods=1).mean().pow(0.5).to_numpy()
    local = pd.Series(rms).rolling(int(20 * FPS), center=True, min_periods=1).max().to_numpy()
    active = rms > 0.3 * local
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2])]
    shortest = 0.5 * min((2 + 10) / f for _, f in plan) * FPS
    runs = [r for r in runs if r[1] - r[0] >= shortest]
    if len(runs) != len(plan):
        raise SystemExit(f"Found {len(runs)} swinging stretches, expected {len(plan)}: {[(a / FPS, b / FPS) for a, b in runs]}")
    return runs


def fit_amplitude(y: np.ndarray, t: np.ndarray, freq: float) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 0.5 * len(y):
        return float("nan")
    w = 2 * np.pi * freq * t[ok]
    a = np.column_stack([np.sin(w), np.cos(w), np.ones_like(w)])
    c, *_ = np.linalg.lstsq(a, y[ok], rcond=None)
    return float(np.hypot(c[0], c[1]))


def measured_frequency(y: np.ndarray, guess: float) -> float:
    y = pd.Series(y).interpolate().bfill().ffill().to_numpy()
    n = len(y)
    pad = 16 * n
    spec = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(n), pad))
    freq = np.fft.rfftfreq(pad, 1 / FPS)
    band = (freq > 0.5 * guess) & (freq < 1.5 * guess)
    return float(freq[band][np.argmax(spec[band])])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result", type=Path, help="Result folder with triangulated_3d.csv")
    ap.add_argument("--cycles", type=int, default=10, help="Steady cycles per segment")
    ap.add_argument("--out", type=Path, help="Output folder (default: <result>/wave_segments)")
    args = ap.parse_args()

    plan = segments_from_label(args.result.name)
    out = args.out or args.result / "wave_segments"
    out.mkdir(parents=True, exist_ok=True)
    p = load_positions(args.result)
    t_all = np.arange(len(p)) / FPS
    bottom = np.nanmean(p[:, -GRID:], axis=1)
    axis_all, _ = main_axis(bottom)
    runs = find_segments(pd.Series(bottom @ axis_all).interpolate().bfill().ffill().to_numpy(), plan)

    rows, summary = [], []
    for k, ((amp, freq), (a, b)) in enumerate(zip(plan, runs), 1):
        period = 1 / freq
        centre = (a + b) / 2
        half = (args.cycles / 2 - 0.25) * period * FPS
        s0, s1 = int(round(centre - half)), int(round(centre + half))
        seg = p[s0:s1]
        axis, share = main_axis(np.nanmean(seg[:, -GRID:], axis=1))
        along = (seg - np.nanmean(seg, axis=0)) @ axis
        f_meas = measured_frequency(np.nanmean(along[:, -GRID:], axis=1), freq)
        t = t_all[s0:s1]
        knot = np.array([fit_amplitude(along[:, i], t, f_meas) for i in range(GRID * GRID)]).reshape(GRID, GRID)
        row_amp = np.nanmean(knot, axis=1)
        summary.append({
            "segment": k, "carriage_amplitude_mm": amp, "commanded_hz": freq, "measured_hz": round(f_meas, 4),
            "swinging_s": [round(a / FPS, 2), round(b / FPS, 2)], "steady_s": [round(s0 / FPS, 2), round(s1 / FPS, 2)],
            "motion_axis_xyz": [round(float(v), 3) for v in axis], "share_on_axis": round(share, 4),
            "row_amplitude_mm": [round(float(v), 1) for v in row_amp],
            "bottom_row_amplitude_over_carriage": round(float(row_amp[-1] / amp), 3),
        })
        for r in range(GRID):
            for c in range(GRID):
                rows.append({"segment": k, "carriage_amplitude_mm": amp, "commanded_hz": freq,
                             "measured_hz": f_meas, "row": r, "col": c, "obj_id": r * GRID + c,
                             "amplitude_mm": knot[r, c]})
        print(f"[{k}] +-{amp:g} mm {freq:g} Hz: measured {f_meas:.3f} Hz, steady {s0 / FPS:.1f}-{s1 / FPS:.1f} s, "
              f"{share * 100:.1f}% on one line, row amplitude (mm) top->bottom "
              + " ".join(f"{v:.0f}" for v in row_amp))

    pd.DataFrame(rows).to_csv(out / "knot_amplitude.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    trace = pd.Series(bottom @ axis_all).interpolate().to_numpy()
    ax[0].plot(t_all, trace - np.nanmedian(trace), color="#52514e", lw=0.7)
    for k, s in enumerate(summary):
        ax[0].axvspan(*s["steady_s"], color=COLORS[k], alpha=0.15, lw=0)
        ax[0].text(np.mean(s["steady_s"]), ax[0].get_ylim()[1] * 0.92 if k else ax[0].get_ylim()[1] * 0.92,
                   f"±{s['carriage_amplitude_mm']:g} mm, {s['commanded_hz']:g} Hz", ha="center", fontsize=9, color="#2b2b29")
    ax[0].set(xlabel="time (s)", ylabel="bottom row (mm)", title="Bottom row vs the carriage; shaded = steady cycles measured")
    rows_idx = np.arange(GRID)
    for k, s in enumerate(summary):
        lab = f"±{s['carriage_amplitude_mm']:g} mm, {s['commanded_hz']:g} Hz"
        ax[1].plot(rows_idx, s["row_amplitude_mm"], "-o", ms=4, lw=2, color=COLORS[k], label=lab)
        ax[2].plot(rows_idx, np.array(s["row_amplitude_mm"]) / s["carriage_amplitude_mm"], "-o", ms=4, lw=2, color=COLORS[k], label=lab)
    ax[1].set(xlabel="row (0 = top, 11 = bottom)", ylabel="amplitude (± mm)", title="Swing by row, vs the carriage")
    ax[2].set(xlabel="row (0 = top, 11 = bottom)", ylabel="net amplitude / carriage amplitude", title="Swing per mm of carriage motion")
    ax[1].legend(frameon=False, fontsize=9)
    for a_ in ax:
        a_.grid(alpha=0.25)
        a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out / "wave_segments.png", dpi=110)
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
