"""Append the first wave-test result (A = 0.05 m, 0.5 Hz, trial 2) to the CoTracker3 deck.
Edits nothing else in the deck. Render the video first with make_wave_media.py.

    python experiments/cotracker3/extend_deck_wave.py <in.pptx> <out.pptx>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_AXIS_CROSSES, XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extend_deck_lighting as d  # noqa: E402

ROOT = d.ROOT
RESULT = ROOT / "triangulation" / "results" / "A_0.05_F_0.25_trial2_cotracker3"
QUEUE_RUN = ROOT / "work" / "pipeline_queue" / "20261106_052812_545745" / "run_001_A_0.1_F_0.25_trial2"
FPS = 29.97
GRID = 12
COLUMN = 6  # knots plotted over time: this column, top / middle / bottom row
ROWS = [(0, "top row", "cyan"), (6, "middle row", "green"), (11, "bottom row", "orange")]


def motion():
    """Displacement (mm) of every knot about its mean position, along the main direction of the bottom row's motion."""
    df = pd.read_csv(RESULT / "triangulated_3d.csv")
    ok = df[df["valid_3d"] == 1]
    n = int(df["gframe"].max()) + 1
    p = np.full((n, GRID * GRID, 3), np.nan)
    p[ok["gframe"], ok["obj_id"]] = ok[["X", "Y", "Z"]].to_numpy()
    disp = (p - np.nanmean(p, axis=0)) * 1000
    bottom = pd.DataFrame(np.nanmean(disp[:, -GRID:], axis=1)).interpolate().bfill().ffill().to_numpy()
    _, sv, vt = np.linalg.svd(bottom - bottom.mean(0), full_matrices=False)
    axis = vt[0] if vt[0][np.argmax(np.abs(vt[0]))] > 0 else -vt[0]
    along = disp @ axis
    spec = np.abs(np.fft.rfft((bottom @ axis) * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1 / FPS)
    peak = float(freq[1:][np.argmax(spec[1:])])
    share = float(sv[0] ** 2 / (sv ** 2).sum())
    return along, peak, share, df


def style_axis(ax, title_s):
    ink, rule = RGBColor.from_string(d.HEX["ink"]), RGBColor.from_string(d.HEX["rule"])
    ax.tick_labels.font.size, ax.tick_labels.font.color.rgb = Pt(11), ink
    ax.format.line.color.rgb = rule
    ax.has_major_gridlines = True
    ax.major_gridlines.format.line.color.rgb = rule
    ax.has_title = True
    ax.axis_title.text_frame.text = title_s
    f = ax.axis_title.text_frame.paragraphs[0].runs[0].font
    f.size, f.bold, f.color.rgb = Pt(12), False, RGBColor.from_string(d.HEX["muted"])


def time_chart(slide, along, x, y, w, h):
    data = XyChartData()
    t = np.arange(along.shape[0]) / FPS
    for row, label, _ in ROWS:
        s = pd.Series(along[:, row * GRID + COLUMN]).interpolate(limit_area="inside")
        ser = data.add_series(label)
        for ti, v in zip(t[::2], s.to_numpy()[::2]):
            if np.isfinite(v):
                ser.add_data_point(round(float(ti), 3), round(float(v), 2))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), data)
    gf.name = "wave time chart"
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "Displacement over time, middle column of knots (mm)"
    f = ch.chart_title.text_frame.paragraphs[0].runs[0].font
    f.size, f.bold, f.color.rgb = Pt(14), False, RGBColor.from_string(d.HEX["ink"])
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size, ch.legend.font.color.rgb = Pt(12), RGBColor.from_string(d.HEX["ink"])
    for ser, (_, _, col) in zip(ch.plots[0].series, ROWS):
        ser.format.line.color.rgb = RGBColor.from_string(d.HEX[col])
        ser.format.line.width = Pt(1.75)
        ser.smooth = False
    style_axis(ch.category_axis, "time (s)")
    style_axis(ch.value_axis, "mm")
    ch.category_axis.minimum_scale, ch.category_axis.maximum_scale = 0, round(float(t[-1]), 1)
    ch.category_axis.has_major_gridlines = False
    ch.value_axis.crosses = XL_AXIS_CROSSES.MINIMUM  # time labels along the bottom, not across the curves


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    prs = Presentation(src)
    d.NOTES_BODY = next(sl.notes_slide.notes_placeholder._element for sl in prs.slides
                        if sl.has_notes_slide and sl.notes_slide.notes_placeholder is not None)
    d.SLIDE_NUMBER = next(sh._element for sl in prs.slides for sh in sl.shapes if sh.name.startswith("Slide Number"))
    media = json.loads((d.MEDIA / "wave_tracking.json").read_text())
    d.MANIFEST["wave_tracking"] = media

    summary = json.loads((RESULT / "summary.json").read_text())
    meta = {s: json.loads((QUEUE_RUN / "cotracker3" / s / "run_meta.json").read_text()) for s in ("left", "right")}
    along, peak, share, df = motion()
    n_frames = meta["left"]["n_frames"]
    never = sorted(int(k) for k, v in df.groupby("obj_id")["valid_3d"].max().items() if v == 0)
    track_min = max(sum(m["seconds"].values()) for m in meta.values()) / 60

    # 05 Wave test
    s = prs.slides.add_slide(d.layout(prs, "SECTION"))
    d.text(s, "05", d.MX, 2.75, 1.4, 1.0, size=54, color=d.ORANGE, bold=True, head=True, anchor=MSO_ANCHOR.MIDDLE)
    s.placeholders[100].text = "Wave test"
    s.placeholders[101].text = "First wave run through the full pipeline: A = 0.05 m, 0.5 Hz (2 s period), trial 2. Preliminary."

    # Video + key numbers, same layout as the tow-tracking slide
    s = prs.slides.add_slide(d.layout(prs, "CONTENT"))
    d.title(s, "The pipeline runs end to end on a wave test")
    d.video(s, "wave_tracking", d.MX, 1.3, 5.2, 5.5, left=True)
    x = 6.28
    d.text(s, "Left camera, played at half speed. Green dots are CoTracker3 after the snap; lines join grid neighbours.",
           x, 1.35, d.W - d.MX - x, 0.8, size=14, color=d.MUTED)
    stats = [("Clip", f"{n_frames} frames ({n_frames / FPS:.1f} s), 144 knots queried once in frame 0"),
             ("3D points kept", f"{summary['valid_pct']:.1f}% ({summary['valid_3d_obs']:,} of {summary['total_matched_obs']:,})"),
             ("Reprojection error", f"median {summary['reproj_err_px']['median']:.1f} px, 95th percentile {summary['reproj_err_px']['p95']:.1f} px"),
             ("Not reconstructed", f"the 2 top corner knots ({', '.join(map(str, never))}) fail the 20 px check in every frame"
              if len(never) == 2 else f"{len(never)} knots fail the 20 px check in every frame"),
             ("Run time", f"tracking ~{track_min:.0f} min per camera on the GPU box; calibration took most of the run")]
    for i, (k, v) in enumerate(stats):
        y = 2.4 + i * 0.86
        d.dot(s, x, y + 0.07, 0.18, d.GREEN)
        d.text(s, k, x + 0.35, y, 5.5, 0.35, size=15, bold=True, head=True)
        d.text(s, v, x + 0.35, y + 0.33, d.W - d.MX - x - 0.35, 0.5, size=14, color=d.MUTED)
    d.notes(s, "Wave test 2 at A = 0.05 m, 0.5 Hz (the queue still labels it A_0.1_F_0.25; both labels were wrong). Queue made with run_pipeline_queue.py "
               "--setup-only on the laptop, processed on the Linux box with remote_pipeline.py run --tracker cotracker3. "
               "Same CoTracker3 settings as the tow runs: offline model, 384 x 512 input, dtpeak snap radius 40, hidden points not snapped.")

    # Motion
    s = prs.slides.add_slide(d.layout(prs, "CONTENT"))
    d.title(s, "The net swings with the wave, most at the bottom")
    time_chart(s, along, d.MX, 1.25, 7.9, 4.85)
    rows = np.nanpercentile(along, 95, axis=0) - np.nanpercentile(along, 5, axis=0)
    per_row = [float(np.nanmedian(rows[r * GRID:(r + 1) * GRID])) for r in range(GRID)]
    d.add_chart(s, 8.75, 1.25, d.W - d.MX - 8.75, 4.85, "Swing by row, top to bottom (mm)",
                [f"row {r + 1}" for r in range(GRID)], [round(v) for v in per_row],
                [d.HEX["green"]] * GRID, "0")
    d.text(s, f"Displacement about each knot's mean position, along the main direction of motion ({share * 100:.1f}% of the bottom row's "
              f"motion lies on one line). Swing = 5th to 95th percentile. "
              f"Measured {peak:.2f} Hz vs commanded 0.50 Hz; 5 steady cycles plus ramp up and down, 14 s, as commanded.",
           d.MX, 6.25, d.W - 2 * d.MX, 0.6, size=13, color=d.MUTED)
    d.notes(s, "Commanded motion: amplitude 0.05 m, period 2.0 s, 5 steady cycles + 1 ramp up + 1 ramp down, 14 s, tow speed 0. The FFT of a 20 s clip resolves about 0.05 Hz, so 0.49 Hz matches 0.50 Hz. Preliminary, one trial. No reference or FEM comparison for the wave case yet. The main direction of motion is mostly "
               "along the camera axis with a vertical component, i.e. the net swings toward and away from the cameras.")

    prs.save(dst)
    print(f"[OK] {dst}  slides={len(prs.slides)}  peak={peak:.3f} Hz  share={share:.3f}  rows={[round(v) for v in per_row]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
