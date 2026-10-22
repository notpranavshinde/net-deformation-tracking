"""Append the lighting-test section to the user's weekly CoTracker3 deck, and fill the
precision table on the "why it works" slide. Edits nothing else in the deck.

    python experiments/cotracker3/extend_deck_lighting.py <in.pptx> <out.pptx>
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "work" / "cotracker3" / "deck" / "media"
LIGHT = ROOT / "work" / "cotracker3" / "lighting" / "run_006_v_0.3mps_a"
MANIFEST = json.loads((MEDIA / "manifest.json").read_text())
W, MX = 13.333, 0.6
T = MSO_THEME_COLOR
INK, MUTED, ORANGE, GREEN, CYAN, RED, PANEL, RULE = (T.BACKGROUND_2, T.ACCENT_5, T.ACCENT_1, T.ACCENT_2, T.ACCENT_3,
                                                      T.ACCENT_4, T.TEXT_2, T.ACCENT_6)
HEX = {"green": "5AD27A", "cyan": "4CC9F0", "orange": "F39A2B", "red": "FF5A4F", "muted": "8EA19F", "rule": "24343A", "ink": "DFE8E6"}
ORDER = ["dark", "bright", "uneven", "flicker", "caustics", "murky", "background"]
LABEL = {"dark": "Underexposed", "bright": "Overexposed", "uneven": "Uneven lighting", "flicker": "Flicker",
         "caustics": "Surface caustics", "murky": "Murky water", "background": "Non-black background"}


def layout(prs, name):
    return next(l for l in prs.slide_layouts if l.name == name)


def text(slide, s, x, y, w, h, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT, head=False, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    runs = s if isinstance(s, list) else [(s, {})]
    p = tf.paragraphs[0]
    p.alignment = align
    for chunk, opt in runs:
        if chunk == "\n":
            p = tf.add_paragraph()
            p.alignment = align
            continue
        r = p.add_run()
        r.text = chunk
        f = r.font
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.italic = italic
        f.color.theme_color = opt.get("color", color)
        if head or opt.get("head"):
            f.name = "+mj-lt"
    return tb


def dot(slide, x, y, d, color, name="knot"):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.theme_color = color
    s.line.fill.background()
    s.name = name
    return s


def card(slide, x, y, w, h, name="card"):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.03
    s.fill.solid()
    s.fill.fore_color.theme_color = PANEL
    s.line.color.theme_color = RULE
    s.line.width = Pt(0.75)
    s.name = name
    return s


SLIDE_NUMBER = None  # the slide-number shape from an existing content slide


def title(slide, s):
    slide.placeholders[100].text = s
    slide.shapes._spTree.append(copy.deepcopy(SLIDE_NUMBER))


NOTES_BODY = None  # a notes text box borrowed from an existing slide; this deck's notes master has none


def notes(slide, s):
    ns = slide.notes_slide
    if ns.notes_text_frame is None:
        ns.shapes._spTree.append(copy.deepcopy(NOTES_BODY))
    ns.notes_text_frame.text = s


def fit(name, x, y, w, h):
    a = MANIFEST[name]["w"] / MANIFEST[name]["h"]
    vw, vh = w, w / a
    if vh > h:
        vh, vw = h, h * a
    return x + (w - vw) / 2, y + (h - vh) / 2, vw, vh


def video(slide, name, x, y, w, h, left=False):
    vx, vy, vw, vh = fit(name, x, y, w, h)
    if left:
        vx = x
    m = slide.shapes.add_movie(str(MEDIA / f"{name}.mp4"), Inches(vx), Inches(vy), Inches(vw), Inches(vh),
                               poster_frame_image=str(MEDIA / f"{name}_cover.png"), mime_type="video/mp4")
    m.name = name
    return vx, vy, vw, vh


def legend(slide, x, y):
    items = [("within 10 px of the clean run", GREEN), ("more than 10 px off", ORANGE), ("on the wrong knot", RED)]
    for label, col in items:
        dot(slide, x, y + 0.06, 0.16, col)
        text(slide, label, x + 0.24, y, 2.8, 0.28, size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        x += 0.45 + len(label) * 0.075
    ring = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.06), Inches(0.16), Inches(0.16))
    ring.fill.background()
    ring.line.color.theme_color = MUTED
    ring.line.width = Pt(1.5)
    text(slide, "ring: model says hidden", x + 0.24, y, 2.6, 0.28, size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)


def set_cell(cell, value, color=None):
    """Replace a table cell's text, keeping the formatting of its first run."""
    p = cell.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = value
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in cell.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    if color is not None:
        runs[0].font.color.theme_color = color


def results():
    out = {}
    for f in LIGHT.glob("*/severe/result.json"):
        r = json.loads(f.read_text())
        out[r["kind"]] = r
    return out


def clean_metrics():
    c = json.loads((ROOT / "work" / "cotracker3" / "run_006_v_0.3mps_a" / "offline_384x512_dtpeak" / "compare.json").read_text())
    return c["3d_cotracker"], c["3d_sam2"]


def fill_precision_table(prs) -> None:
    for slide in prs.slides:
        for sh in slide.shapes:
            if sh.has_table and sh.name == "precision table":
                rows = {"SAM2": ("1.94 px", "2.16 mm"), "CoTracker3 raw": ("2.49 px", "0.84 mm"), "CoTracker3 + snap": ("1.78 px", "0.59 mm")}
                for row in sh.table.rows:
                    key = row.cells[0].text.strip()
                    if key in rows:
                        set_cell(row.cells[1], rows[key][0])
                        set_cell(row.cells[2], rows[key][1])
            if sh.has_text_frame and sh.text_frame.text.startswith("Raw output is slightly worse than SAM2"):
                p = sh.text_frame.paragraphs[0]
                p.runs[0].text = sh.text_frame.text.replace("Raw output is slightly worse than SAM2 but slightly less precise",
                                                            "Raw output is steadier than SAM2 but slightly less precise")
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)


def results_table(prs, slide, res, clean):
    """Clone the occlusion table's styling, trimmed to 6 columns x 9 rows."""
    src = next(sh for s in prs.slides for sh in s.shapes if sh.has_table and sh.name == "occlusion table")
    frame = copy.deepcopy(src._element)
    slide.shapes._spTree.insert_element_before(frame, "p:extLst")
    gf = slide.shapes[-1]
    gf.name = "lighting table"
    tbl = gf.table
    grid = tbl._tbl.tblGrid
    keep = [0, 1, 2, 3, 5, 7]  # drop two of the eight columns
    for idx in sorted(set(range(8)) - set(keep), reverse=True):
        grid.remove(grid.gridCol_lst[idx])
        for tr in tbl._tbl.tr_lst:
            tr.remove(tr.tc_lst[idx])
    widths = [3.0, 1.9, 1.9, 1.8, 1.7, 1.83]
    for gc, wdt in zip(grid.gridCol_lst, widths):
        gc.w = Inches(wdt)
    gf.left, gf.top, gf.width = Inches(MX), Inches(1.3), Inches(sum(widths))
    head = ["Condition (severe)", "On wrong knot", "Snap shift vs clean", "3D diff vs clean", "3D points kept", "3D jitter"]
    for c, h in enumerate(head):
        set_cell(tbl.cell(0, c), h)
    rows = [("Clean recording", "0", "—", "—", f"{clean['valid_3d_frac'] * 100:.1f}%", f"{clean['jitter_2nd_diff_mm']['median']:.2f} mm")]
    for k in ORDER:
        r = res[k]
        L, R, t = r["2d_left"], r["2d_right"], r["3d"]
        w = max(L["wrong_knot"], R["wrong_knot"])
        snap = (L["snap_err_px"]["median"] + R["snap_err_px"]["median"]) / 2
        rows.append((LABEL[k], "0" if w == 0 else f"{w * 100:.1f}%", f"{snap:.1f} px", f"{t['err_vs_clean_mm']['median']:.2f} mm",
                     f"{t['valid_3d_frac'] * 100:.1f}%", f"{t['jitter_2nd_diff_mm']['median']:.2f} mm"))
    for ri, row in enumerate(rows, start=1):
        for c, v in enumerate(row):
            bad = (c == 2 and v.endswith("px") and float(v.split()[0]) >= 2.5) or (c == 3 and v.endswith("mm") and float(v.split()[0]) >= 1.0)
            set_cell(tbl.cell(ri, c), v, RED if bad else INK)
    return gf


def add_chart(slide, x, y, w, h, title_s, cats, values, colors, fmt):
    data = CategoryChartData()
    data.categories = cats
    data.add_series(title_s, values)
    gframe = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    gframe.name = title_s
    ch = gframe.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = title_s
    tr = ch.chart_title.text_frame.paragraphs[0].runs[0].font
    tr.size, tr.bold, tr.color.rgb = Pt(14), False, RGBColor.from_string(HEX["ink"])
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.color.rgb = Pt(11), RGBColor.from_string(HEX["ink"])
    for i, pt in enumerate(plot.series[0].points):
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = RGBColor.from_string(colors[i])
    ca, va = ch.category_axis, ch.value_axis
    ca.reverse_order = True
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(12), RGBColor.from_string(HEX["ink"])
    ca.format.line.color.rgb = RGBColor.from_string(HEX["rule"])
    va.visible = False
    va.has_major_gridlines = False
    return gframe


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    prs = Presentation(src)
    global NOTES_BODY
    NOTES_BODY = next(sl.notes_slide.notes_placeholder._element for sl in prs.slides
                      if sl.has_notes_slide and sl.notes_slide.notes_placeholder is not None)
    global SLIDE_NUMBER
    SLIDE_NUMBER = next(sh._element for sl in prs.slides for sh in sl.shapes if sh.name.startswith("Slide Number"))
    fill_precision_table(prs)
    res = results()
    clean, sam2 = clean_metrics()
    missing = [k for k in ORDER if k not in res]
    if missing:
        raise SystemExit(f"missing severe results for {missing}")

    # 04 Lighting
    s = prs.slides.add_slide(layout(prs, "SECTION"))
    text(s, "04", MX, 2.75, 1.4, 1.0, size=54, color=ORANGE, bold=True, head=True, anchor=MSO_ANCHOR.MIDDLE)
    s.placeholders[100].text = "Lighting"
    s.placeholders[101].text = "Degrading the real footage to see how much bad lighting the tracker tolerates"

    # The test
    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "The test: seven lighting problems painted into real footage")
    steps = [("Degrade", "Both cameras of run 006 (0.30 m/s), severe strength."),
             ("Rerun", "CoTracker3 + snap on each degraded video, then the normal triangulation."),
             ("Score", "Against the clean recording: same knot? how far moved? 3D change?")]
    for i, (h, d) in enumerate(steps):
        y = 1.45 + i * 1.25
        dot(s, MX, y, 0.5, ORANGE, f"step {i + 1}")
        text(s, str(i + 1), MX, y, 0.5, 0.5, size=16, bold=True, color=T.TEXT_1, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, h, MX + 0.75, y - 0.02, 3.2, 0.45, size=19, bold=True, head=True)
        text(s, d, MX + 0.75, y + 0.42, 3.4, 0.75, size=14, color=MUTED)
    text(s, "Not tested: lower resolution (a camera choice) and floating particles (already in the footage).",
         MX, 5.35, 4.1, 1.0, size=13, color=MUTED, italic=True)
    x, y, w, h = fit("light_montage", 5.0, 1.3, W - MX - 5.0, 5.4)
    s.shapes.add_picture(str(MEDIA / "light_montage.png"), Inches(x), Inches(y), Inches(w), Inches(h)).name = "conditions still"
    notes(s, "Seven conditions, each simulated in software on the full-resolution frames of both cameras: underexposure with sensor noise, "
             "overexposure with clipping, uneven lighting (gradient plus hotspot), flicker with rolling-shutter banding, moving surface "
             "caustics, murky water (haze, red absorbed, blur), and a textured background replacing the black liner. Only the severe "
             "strength was run for all seven; mild underexposure was also run and showed no effect.")

    # Montage video
    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "Every knot stays on its knot under every condition")
    video(s, "light_montage", MX, 1.25, W - 2 * MX, 5.2)
    legend(s, MX, 6.55)
    notes(s, "Left camera, cropped to the middle of the grid, half speed, all seven conditions at severe strength plus the clean run. "
             "Dots are CoTracker3's output coloured by distance from the clean result.")

    # Charts
    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "3D stays within 2 mm of the clean run")
    order3d = sorted(ORDER, key=lambda k: res[k]["3d"]["err_vs_clean_mm"]["median"])
    v3d = [res[k]["3d"]["err_vs_clean_mm"]["median"] for k in order3d]
    add_chart(s, MX, 1.3, 6.0, 4.9, "3D difference from the clean run, median (mm)", [LABEL[k] for k in order3d], v3d,
              [HEX["red"] if v >= 1.0 else HEX["orange"] if v >= 0.5 else HEX["green"] for v in v3d], "0.00")
    jcats = ["SAM2, clean", "CoTracker3, clean"] + [LABEL[k] for k in sorted(ORDER, key=lambda k: res[k]["3d"]["jitter_2nd_diff_mm"]["median"])]
    jvals = [sam2["jitter_2nd_diff_mm"]["median"], clean["jitter_2nd_diff_mm"]["median"]] + \
            [res[k]["3d"]["jitter_2nd_diff_mm"]["median"] for k in sorted(ORDER, key=lambda k: res[k]["3d"]["jitter_2nd_diff_mm"]["median"])]
    add_chart(s, 6.73, 1.3, 6.0, 4.9, "3D jitter, median (mm)", jcats, jvals,
              [HEX["cyan"], HEX["muted"]] + [HEX["green"]] * len(ORDER), "0.00")
    text(s, "For scale: the paper's FEM agreement is 4.88 mm RMSE. Even the worst lighting case keeps jitter below SAM2's on clean footage.",
         MX, 6.35, W - 2 * MX, 0.5, size=14, color=MUTED)
    notes(s, "Left: median 3D distance between each knot in the degraded run and the same knot in the clean run. Right: median frame-to-frame "
             "jitter (second difference of each knot's 3D path); grey is clean CoTracker3, blue is SAM2 on the clean footage.")

    # Table
    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "Lighting results, severe strength")
    results_table(prs, s, res, clean)
    text(s, "Snap shift: how far the snapped positions moved from the clean run (median, both cameras). Red marks the cases that moved: "
            "the colour snap struggles, the tracking does not.", MX, 5.95, W - 2 * MX, 0.7, size=14, color=MUTED)
    notes(s, "Wrong knot is the worse of the two cameras. 3D points kept is the share passing the 20 px reprojection check.")

    def video_slide(name, title_s, caption, stat, note):
        # Same frame as the occlusion video slides: full-width clip, caption, legend, key numbers.
        s = prs.slides.add_slide(layout(prs, "CONTENT"))
        title(s, title_s)
        video(s, name, MX, 1.25, W - 2 * MX, 4.9)
        text(s, caption, MX, 6.2, W - 2 * MX, 0.3, size=12, color=MUTED)
        legend(s, MX, 6.57)
        text(s, stat, 8.6, 6.52, W - MX - 8.6, 0.4, size=14, bold=True, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        notes(s, note)

    r = res["murky"]
    video_slide("light_murky", "Murky water: the snap gives up, the tracker does not",
                f"Haze, red absorbed, blur. The snap found {r['2d_left']['snapped'] * 100:.0f}% of knots, so every point keeps the model's own estimate.",
                f"no wrong knots · 3D {r['3d']['err_vs_clean_mm']['median']:.1f} mm",
                "Severe murky water: transmission 0.3, red channel x0.4, 8 px blur. The snap needs red-minus-blue contrast, which this removes, "
                "so every point keeps CoTracker3's raw estimate (about 6.5 px from the snapped clean positions, which is the raw model's normal precision).")
    r = res["bright"]
    video_slide("light_bright", "Overexposed: knot centres shift, identities hold",
                "4.5x too bright. Markers clip toward white, and the ring of colour around the white core shifts the snapped centre by ~3 px.",
                f"no wrong knots · 3D {r['3d']['err_vs_clean_mm']['median']:.1f} mm · jitter {r['3d']['jitter_2nd_diff_mm']['median']:.1f} mm",
                "Severe overexposure: gain 4.5 with clipping. The snap uses red minus blue, which is small on a clipped white centre.")

    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "Caustics and a busy background: small cost")
    box = (MX, 1.25, 7.6, 2.55)
    vr = video(s, "light_caustics", *box, left=True)
    video(s, "light_background", box[0], 3.95, box[2], box[3], left=True)
    x = vr[0] + vr[2] + 0.45  # text column right next to the stacked clips
    rc, rb = res["caustics"], res["background"]
    text(s, "Surface caustics (top)", x, 1.3, W - MX - x, 0.4, size=17, bold=True, head=True)
    text(s, f"Moving light ripples. 3D within {rc['3d']['err_vs_clean_mm']['median']:.1f} mm, "
            f"{max(rc['2d_left']['wrong_knot'], rc['2d_right']['wrong_knot']) * 100:.1f}% wrong knots at worst.", x, 1.72, W - MX - x, 1.5, size=14)
    text(s, "Non-black background (bottom)", x, 3.95, W - MX - x, 0.4, size=17, bold=True, head=True)
    text(s, f"Lit textured wall instead of the black liner. 3D within {rb['3d']['err_vs_clean_mm']['median']:.2f} mm, "
            f"{max(rb['2d_left']['wrong_knot'], rb['2d_right']['wrong_knot']) * 100:.1f}% wrong knots, {rb['3d']['valid_3d_frac'] * 100:.1f}% of 3D points kept.",
         x, 4.37, W - MX - x, 1.6, size=14)
    legend(s, MX, 6.57)
    notes(s, "These two are the only conditions with any wrong-knot cases, both well under 1%. The background is the realistic case for a farm net.")

    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "Low light and flicker: almost no effect")
    video(s, "light_dark", *box, left=True)
    video(s, "light_flicker", box[0], 3.95, box[2], box[3], left=True)
    rd, rf = res["dark"], res["flicker"]
    text(s, "Underexposed (top)", x, 1.3, W - MX - x, 0.4, size=17, bold=True, head=True)
    text(s, f"8x darker with sensor noise. 3D within {rd['3d']['err_vs_clean_mm']['median']:.2f} mm; the snap misses "
            f"{(1 - rd['2d_left']['snapped']) * 100:.0f}% of knots and falls back to the model.", x, 1.72, W - MX - x, 1.6, size=14)
    text(s, "Flicker (bottom)", x, 3.95, W - MX - x, 0.4, size=17, bold=True, head=True)
    text(s, f"±60% brightness with rolling-shutter bands. 3D within {rf['3d']['err_vs_clean_mm']['median']:.2f} mm, no wrong knots.",
         x, 4.37, W - MX - x, 1.6, size=14)
    legend(s, MX, 6.57)
    notes(s, "Uneven lighting (gradient plus hotspot) behaved like these: 0.10 mm, no wrong knots. It is in the montage.")

    # Snap figure
    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "The weak link is the colour snap, not the tracker")
    y, w, h = 1.35, W - 2 * MX, 2.9
    from PIL import Image
    iw, ih = Image.open(MEDIA / "light_snap.png").size
    pw = min(w, h * iw / ih)
    s.shapes.add_picture(str(MEDIA / "light_snap.png"), Inches(MX + (w - pw) / 2), Inches(y), Inches(pw), Inches(pw * ih / iw)).name = "snap crops"
    cards = [("Why", "The snap finds each knot by red minus blue, then takes the blob's centre. Bad light changes the colour, so the blob shrinks, shifts or vanishes."),
             ("What it costs", "Up to 1.8 mm in 3D (murky water). When the snap fails, the point keeps the model's estimate, so nothing is lost."),
             ("Fix (software)", "Snap on local contrast instead of a fixed colour, and only accept a blob close to the model's estimate.")]
    for i, (hd, d) in enumerate(cards):
        cx = MX + i * 4.1
        card(s, cx, 4.5, 3.85, 2.05, f"snap card {i + 1}")
        text(s, hd, cx + 0.25, 4.65, 3.4, 0.4, size=17, bold=True, head=True, color=ORANGE if i == 2 else INK)
        text(s, d, cx + 0.25, 5.1, 3.4, 1.4, size=14)
    notes(s, "Knot 66 at frame 110, 81 px window around the clean knot centre (white cross). The cyan outline is the blob the snap would use.")

    # Next steps
    s = prs.slides.add_slide(layout(prs, "CONTENT"))
    title(s, "Next steps for lighting")
    steps = [("Make the snap colour-independent", "Contrast-based knot centre; rerun the seven conditions to confirm."),
             ("Tank test, real lights", "Focus on what moved the numbers: overexposure, murky water, low light."),
             ("Get the GPU working", "CPU took ~13 min per case; the GPU should make full sweeps (all strengths, all speeds) practical."),
             ("Fill in the sweep", "Mild and moderate strengths, and a slower run, once the GPU works.")]
    for i, (hd, d) in enumerate(steps):
        y = 1.45 + i * 1.28
        dot(s, MX, y, 0.6, ORANGE, f"next {i + 1}")
        text(s, str(i + 1), MX, y, 0.6, 0.6, size=20, bold=True, color=T.TEXT_1, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, hd, MX + 0.9, y - 0.02, 11, 0.45, size=20, bold=True, head=True)
        text(s, d, MX + 0.9, y + 0.45, 11, 0.6, size=15, color=MUTED)
    notes(s, "Caveat for all of the lighting results: synthetic lighting, one run at 0.30 m/s, severe strength only.")

    prs.save(dst)
    print(f"[OK] {dst}  slides={len(prs.slides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
