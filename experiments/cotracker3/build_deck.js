// Build the CoTracker3 proposal deck from the rendered media in work/cotracker3/deck/media.
// Usage: NODE_PATH=<dir with pptxgenjs> node experiments/cotracker3/build_deck.js <applyTheme module path>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.argv[2]);

const ROOT = path.resolve(__dirname, "..", "..");
const MEDIA = path.join(ROOT, "work", "cotracker3", "deck", "media");
const OUTFILE = path.join(ROOT, "work", "cotracker3", "deck", "CoTracker3_net_tracking_proposal.pptx");
const MANIFEST = JSON.parse(fs.readFileSync(path.join(MEDIA, "manifest.json"), "utf8"));

// Dark tank palette, matching the footage and the viewers.
const THEME = {
  name: "Towing Tank",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0D1416", lt1: "FFFFFF", dk2: "142024", lt2: "DFE8E6",
    accent1: "F39A2B", // painted-knot orange
    accent2: "5AD27A", // CoTracker3
    accent3: "4CC9F0", // SAM2
    accent4: "FF5A4F", // error
    accent5: "8EA19F", // muted text
    accent6: "24343A", // hairlines
    hlink: "F39A2B", folHlink: "F39A2B",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Pranav Shinde";
pres.title = "CoTracker3 for net deformation tracking";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.6;

// ---------- Layouts ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 2.2, w: 6.6, h: 1.9, fontSize: 40, bold: true, color: C.background2, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 4.3, w: 6.6, h: 1.4, fontSize: 18, color: C.accent5, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX + 1.6, y: 2.75, w: 10.5, h: 1.0, fontSize: 38, bold: true, color: C.background2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX + 1.6, y: 3.8, w: 10.5, h: 1.0, fontSize: 18, color: C.accent5, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: W - 2 * MX, h: 0.8, fontSize: 28, bold: true, color: C.background2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "CoTracker3 for net deformation tracking", options: { x: MX, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------- Helpers ----------
let sectionName = "Opening";
function slide(master) { return pres.addSlide({ masterName: master, sectionTitle: sectionName }); }
function section(name) { sectionName = name; pres.addSection({ title: name }); }
const b64 = (file) => "data:image/png;base64," + fs.readFileSync(file).toString("base64");

// Knot dot: the deck's motif, a painted orange knot.
function knot(s, x, y, d = 0.22, color = C.accent1) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 }, objectName: "knot" });
}

function fit(name, box) {
  const { w, h } = MANIFEST[name];
  const a = w / h;
  let vw = box.w, vh = box.w / a;
  if (vh > box.h) { vh = box.h; vw = box.h * a; }
  return { x: box.x + (box.w - vw) / 2, y: box.y + (box.h - vh) / 2, w: vw, h: vh };
}

function video(s, name, box, align = "center") {
  const r = fit(name, box);
  if (align === "left") r.x = box.x;
  s.addMedia({ type: "video", path: path.join(MEDIA, name + ".mp4"), cover: b64(path.join(MEDIA, name + "_cover.png")), ...r, objectName: name });
  return r;
}

function text(s, t, opts) { s.addText(t, { isTextBox: true, margin: 0, color: C.background2, fontSize: 15, valign: "top", ...opts }); }

function card(s, x, y, w, h, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.text2 }, line: { color: C.accent6, width: 0.75 }, objectName: name });
}

function legendRow(s, x, y, items) {
  let cx = x;
  for (const [label, color] of items) {
    knot(s, cx, y + 0.06, 0.16, color);
    text(s, label, { x: cx + 0.24, y, w: 2.6, h: 0.28, fontSize: 12, color: C.accent5, valign: "middle" });
    cx += 0.32 + label.length * 0.085;
  }
}

// ---------- 1. Title ----------
section("Opening");
{
  const s = slide("TITLE");
  s.addText("Tracking the net with CoTracker3", { placeholder: "title" });
  s.addText("Proposal to replace SAM2 marker tracking in the next revision of the net-deformation paper\nPranav Shinde · October 2026", { placeholder: "body" });
  const r = fit("cotracker_tracking", { x: 7.75, y: 0.55, w: 5.0, h: 6.4 });
  s.addImage({ path: path.join(MEDIA, "cotracker_tracking_cover.png"), ...r, objectName: "hero frame" });
  s.addNotes("This deck covers a month of side work: a new point tracker, CoTracker3, tested on the trial 15 towing runs we already have, and a synthetic occlusion study. The ask at the end is to use it in the revision.");
}

// ---------- 2. Summary ----------
{
  const s = slide("CONTENT");
  s.addText("CoTracker3 beat SAM2 on every run we have", { placeholder: "title" });
  text(s, "Same footage, same calibration, same triangulation code. Only the 2D tracker changed. 11 usable trial 15 runs, 0.05 to 0.30 m/s.", { x: MX, y: 1.25, w: 12, h: 0.5, fontSize: 16, color: C.accent5 });
  const stats = [
    ["4–5×", "lower frame-to-frame 3D jitter\n2.2–2.9 mm → 0.5–0.6 mm"],
    ["397→2.5", "mm worst knot-to-knot error at 0.30 m/s; SAM2 swaps knots, CoTracker3 does not"],
    ["~99%", "of knots reconstructed in 3D at every speed (SAM2 96–98%)"],
    ["3–5 mm", "3D error on knots hidden for up to 2 s in the occlusion test"],
  ];
  stats.forEach(([big, small], i) => {
    const x = MX + i * 3.09, y = 2.1, w = 2.85, h = 2.85;
    card(s, x, y, w, h, `stat ${i + 1}`);
    knot(s, x + 0.3, y + 0.35, 0.2);
    text(s, big, { x: x + 0.3, y: y + 0.75, w: w - 0.5, h: 0.8, fontSize: big.length > 6 ? 30 : 40, bold: true, fontFace: THEME.headFontFace, color: i === 3 ? C.accent1 : C.accent2, valign: "middle" });
    text(s, small, { x: x + 0.3, y: y + 1.65, w: w - 0.55, h: 1.1, fontSize: 14, color: C.background2 });
  });
  card(s, MX, 5.3, W - 2 * MX, 1.25, "the ask");
  text(s, [
    { text: "The ask  ", options: { bold: true, color: C.accent1 } },
    { text: "Switch the tracking stage of the pipeline to CoTracker3 for the revision, and add a short occlusion study showing the method keeps working when knots are briefly hidden." },
  ], { x: MX + 0.3, y: 5.45, w: W - 2 * MX - 0.6, h: 0.95, fontSize: 17, valign: "middle" });
  s.addNotes("Headline numbers. Jitter is the second difference of each knot's 3D path. Worst knot-to-knot error is the largest variation over time in the 3D distance between neighbouring knots, which should be near constant. run_001a is excluded because its setup prompts are misplaced; both trackers fail there the same way.");
}

// ---------- Section 1: the model ----------
section("The model");
{
  const s = slide("SECTION");
  text(s, "01", { x: MX, y: 2.75, w: 1.4, h: 1.0, fontSize: 54, bold: true, fontFace: THEME.headFontFace, color: C.accent1, valign: "middle" });
  s.addText("The model", { placeholder: "title" });
  s.addText("What CoTracker3 is, how it works, and why it suits a painted-knot net", { placeholder: "body" });
}
{
  const s = slide("CONTENT");
  s.addText("CoTracker3 follows points, SAM2 follows masks", { placeholder: "title" });
  text(s, [
    { text: "CoTracker3 is a point tracker from Meta AI (2024). ", options: { bold: true } },
    { text: "Give it a video and a list of points in one frame; it returns where each point is in every frame, and whether it is visible.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "It was trained on synthetic videos with known motion, then on about 100,000 real videos labelled by other trackers. It needs no training on our data.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Our setup step already produces the 144 knot centres in frame 0, so CoTracker3 drops in where SAM2 was. Triangulation is unchanged." },
  ], { x: MX, y: 1.45, w: 5.2, h: 4.6, fontSize: 16 });
  const rows = [
    ["", "SAM2 (Hiera-L)", "CoTracker3"],
    ["Output", "A mask per knot", "A point per knot + visibility"],
    ["Knots tracked", "One by one", "All 144 together"],
    ["Parameters", "~224 M", "25.4 M"],
    ["Native input", "1024 × 1024", "384 × 512"],
    ["Hidden knot", "No mask, no position", "Estimated, flagged hidden"],
    ["Our runtime", "Remote GPU", "~1 min per camera on CPU"],
  ];
  const head = (t, c) => ({ text: t, options: { bold: true, color: c, fill: { color: C.text2 } } });
  s.addTable(rows.map((r, i) => i === 0
    ? [head("", C.accent5), head(r[1], C.accent3), head(r[2], C.accent2)]
    : [{ text: r[0], options: { color: C.accent5 } }, { text: r[1] }, { text: r[2] }]), {
    x: 6.3, y: 1.45, w: 6.43, colW: [1.7, 2.3, 2.43], fontSize: 14, color: C.background2,
    border: { type: "solid", pt: 0.75, color: HEX.accent6 }, rowH: 0.52, valign: "middle", margin: 0.08, objectName: "comparison table",
  });
  s.addNotes("SAM2 is a segmentation model: it propagates a mask for each prompted object. CoTracker3 is built for exactly our task, tracking query points. Parameter counts: SAM2 Hiera-Large about 224 M, CoTracker3 offline 25.4 M, counted from the checkpoint. CPU runtime is about 30 s of model time plus loading and the full-resolution snap.");
}
{
  const s = slide("CONTENT");
  s.addText("How it works: guess, compare, correct, repeat", { placeholder: "title" });
  const steps = [
    ["Features", "A small CNN turns each frame into 128-number descriptors on a ¼-resolution grid, at 4 scales."],
    ["Fingerprint", "Each knot's look is sampled as a patch of features around it in frame 0."],
    ["Compare", "Around the current guess in every frame, the fingerprint is matched in 7 × 7 windows at all 4 scales."],
    ["Transformer", "Time attention along each track; space attention across all tracks via 64 shared tokens. Outputs a small correction."],
  ];
  const bw = 2.72, gap = 0.36, y = 1.55, bh = 2.75;
  steps.forEach(([t, d], i) => {
    const x = MX + i * (bw + gap);
    card(s, x, y, bw, bh, `step ${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.45, h: 0.45, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `step ${i + 1} badge` });
    text(s, String(i + 1), { x: x + 0.25, y: y + 0.25, w: 0.45, h: 0.45, fontSize: 16, bold: true, color: C.text1, align: "center", valign: "middle" });
    text(s, t, { x: x + 0.85, y: y + 0.25, w: bw - 1.0, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    text(s, d, { x: x + 0.25, y: y + 0.9, w: bw - 0.45, h: bh - 1.05, fontSize: 14 });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.05, y: y + bh / 2 - 0.13, w: gap - 0.1, h: 0.26, fill: { color: C.accent5 }, line: { color: C.accent5, width: 0 }, objectName: `arrow ${i + 1}` });
  });
  // Loop back from the transformer to compare.
  s.addShape(pres.shapes.LINE, { x: MX + 3 * (bw + gap) + bw / 2, y: y + bh, w: 0, h: 0.4, line: { color: C.accent1, width: 2 }, objectName: "loop down" });
  s.addShape(pres.shapes.LINE, { x: MX + 2 * (bw + gap) + bw / 2, y: y + bh + 0.4, w: bw + gap, h: 0, line: { color: C.accent1, width: 2 }, objectName: "loop across" });
  s.addShape(pres.shapes.LINE, { x: MX + 2 * (bw + gap) + bw / 2, y: y + bh, w: 0, h: 0.4, line: { color: C.accent1, width: 2, endArrowType: "triangle" }, flipV: true, objectName: "loop up" });
  text(s, "repeat 6 times with fresh comparisons", { x: MX + 2 * (bw + gap) + bw / 2 + 0.2, y: y + bh + 0.48, w: 3.4, h: 0.3, fontSize: 13, color: C.accent1, italic: true });
  card(s, MX, 5.35, W - 2 * MX, 1.25, "our step");
  knot(s, MX + 0.3, 5.7, 0.2, C.accent2);
  text(s, [
    { text: "Then our full-resolution snap.  ", options: { bold: true, color: C.accent2 } },
    { text: "One grid cell at 384 × 512 is about 40 px in the 5312 × 4648 GoPro frame, so each point the model calls visible is snapped to the orange blob's distance-transform peak, the same centre SAM2 used. Points it calls hidden keep the model's estimate." },
  ], { x: MX + 0.65, y: 5.5, w: W - 2 * MX - 0.95, h: 0.95, fontSize: 15, valign: "middle" });
  s.addNotes("The model starts every track at the query position and refines all of them jointly. Correlation windows are 7x7 at 4 pyramid levels, so the coarsest level sees about +/-100 px of motion at model resolution. The offline model sees about 60 frames at once, forwards and backwards. The snap step: 81x81 px window, red-minus-blue Otsu threshold, nearest blob, distance-transform peak with a sub-pixel weighted centroid.");
}
{
  const s = slide("CONTENT");
  s.addText("Why it suits a painted-knot net", { placeholder: "title" });
  const items = [
    [C.accent2, "The grid moves together", "Space attention lets each knot see its neighbours. A track that jumps to the next knot would move unlike the knots around it, so it gets pulled back. SAM2 tracks each knot alone, so nothing stops a mask sliding onto its neighbour."],
    [C.accent1, "It coasts through gaps", "Time attention looks along the whole track, before and after. A knot lost for a few frames is placed from its own motion and its neighbours'."],
    [C.accent3, "It says when it is guessing", "Each point carries a visibility estimate. Inferred points can be marked as inferred in the dataset instead of passed off as measured."],
  ];
  items.forEach(([col, t, d], i) => {
    const y = 1.5 + i * 1.75;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: col }, line: { color: col, width: 0 }, objectName: `reason ${i + 1} dot` });
    text(s, t, { x: MX + 0.85, y, w: 5.5, h: 0.5, fontSize: 20, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    text(s, d, { x: MX + 0.85, y: y + 0.55, w: 5.6, h: 1.1, fontSize: 15, color: C.accent5 });
  });
  card(s, 7.35, 1.5, 5.38, 5.05, "precision card");
  text(s, "Does the 384 × 512 input cost precision?", { x: 7.65, y: 1.75, w: 4.8, h: 0.5, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
  text(s, "Run 006, 0.30 m/s, both cameras", { x: 7.65, y: 2.25, w: 4.8, h: 0.3, fontSize: 13, color: C.accent5 });
  const rows = [
    ["", "Reprojection", "Jitter"],
    ["SAM2", "1.94 px", "2.16 mm"],
    ["CoTracker3 raw", "2.49 px", "0.84 mm"],
    ["CoTracker3 + snap", "1.78 px", "0.59 mm"],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? C.accent5 : (i === 3 ? C.accent2 : C.background2), align: j ? "right" : "left" } }))), {
    x: 7.65, y: 2.75, w: 4.8, colW: [2.2, 1.3, 1.3], fontSize: 14, rowH: 0.48, valign: "middle", margin: 0.06,
    border: { type: "solid", pt: 0.75, color: HEX.accent6 }, objectName: "precision table",
  });
  text(s, "Raw output is already steadier than SAM2 but slightly less precise. The snap recovers the precision, so the low input resolution is not a limit for us.", { x: 7.65, y: 4.85, w: 4.8, h: 1.5, fontSize: 15 });
  s.addNotes("Raw CoTracker3 is smooth and keeps identities, but its reprojection error is higher because one feature cell is about 40 full-resolution pixels. The snap brings reprojection below SAM2's.");
}
{
  const s = slide("CONTENT");
  s.addText("All 144 knots through a 0.30 m/s tow", { placeholder: "title" });
  const r = video(s, "cotracker_tracking", { x: MX, y: 1.3, w: 5.6, h: 5.55 }, "left");
  const x = r.x + r.w + 0.5, w = W - MX - x;
  text(s, "Left camera, run 006, played at half speed. Green dots are CoTracker3 after the snap; lines join grid neighbours.", { x, y: 1.4, w, h: 1.1, fontSize: 16 });
  const facts = [["Frames", "203 (6.8 s)"], ["Knots", "144, queried once in frame 0"], ["Peak motion", "~11 px per frame"], ["Model time", "~30 s on CPU"]];
  facts.forEach(([k, v], i) => {
    const y = 2.85 + i * 0.85;
    knot(s, x, y + 0.07, 0.18);
    text(s, k, { x: x + 0.35, y, w: 2.0, h: 0.35, fontSize: 13, color: C.accent5 });
    text(s, v, { x: x + 0.35, y: y + 0.32, w: w - 0.35, h: 0.4, fontSize: 17, bold: true });
  });
  s.addNotes("The clip covers the whole run: still, deflection, steady towing and the return when the carriage stops.");
}

// ---------- Section 2: comparison ----------
section("SAM2 vs CoTracker3");
{
  const s = slide("SECTION");
  text(s, "02", { x: MX, y: 2.75, w: 1.4, h: 1.0, fontSize: 54, bold: true, fontFace: THEME.headFontFace, color: C.accent1, valign: "middle" });
  s.addText("SAM2 vs CoTracker3", { placeholder: "title" });
  s.addText("Same footage, same calibration, same triangulation: only the 2D tracker changes", { placeholder: "body" });
}
{
  const s = slide("CONTENT");
  s.addText("Side by side on the same footage", { placeholder: "title" });
  video(s, "sam2_vs_cotracker_full", { x: MX, y: 1.25, w: W - 2 * MX, h: 5.25 });
  legendRow(s, MX, 6.6, [["SAM2", C.accent3], ["CoTracker3", C.accent2], ["SAM2 knot lost or on the wrong knot", C.accent4]]);
  s.addNotes("Run 006, 0.30 m/s, left camera. A SAM2 knot is marked red when its position is closer to a different knot than its own, using the CoTracker3 positions as reference, which were checked by eye.");
}
{
  const s = slide("CONTENT");
  s.addText("SAM2 loses and swaps knots; CoTracker3 keeps them", { placeholder: "title" });
  const r = video(s, "identity_zoom", { x: MX, y: 1.25, w: 8.6, h: 5.4 }, "left");
  const x = r.x + r.w + 0.45, w = W - MX - x;
  const notes = [
    ["115", "SAM2 loses it for ~150 frames, then picks it up one row too high."],
    ["125", "SAM2 slides onto the neighbouring bottom-row knot at frame 37 and stays there."],
    ["137", "SAM2's track is ~2,000 px away, in the upper rows."],
  ];
  notes.forEach(([id, d], i) => {
    const y = 1.4 + i * 1.6;
    text(s, `Knot ${id}`, { x, y, w, h: 0.4, fontSize: 18, bold: true, fontFace: THEME.headFontFace, color: C.accent4 });
    text(s, d, { x, y: y + 0.42, w, h: 1.05, fontSize: 15 });
  });
  text(s, "CoTracker3 keeps all three on their own knots for the whole run.", { x, y: 6.05, w, h: 0.6, fontSize: 15, bold: true, color: C.accent2 });
  s.addNotes("These three were checked frame by frame. On the right camera the same pattern shows up on knots 129, 130, 72 and 84; those have not been checked by eye yet.");
}
{
  const s = slide("CONTENT");
  s.addText("Swapped knots tear the reconstructed net", { placeholder: "title" });
  video(s, "nets3d", { x: MX, y: 1.25, w: W - 2 * MX, h: 5.25 });
  legendRow(s, MX, 6.6, [["edge within 1 mm of its usual length", C.accent5], ["~5 mm off", C.accent1], ["10 mm or more", C.accent4]]);
  s.addNotes("Each edge is coloured by how far its 3D length is from that edge's median length over the run. The twine between knots barely stretches, so a red edge is a tracking error. The faint grid is frame 0. The poster frame is frame 59, where SAM2's worst edge is about 970 mm.");
}
{
  const s = slide("CONTENT");
  s.addText("Steadier at every tow speed, and no blow-ups", { placeholder: "title" });
  const speeds = ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30"];
  const chartBase = {
    x: MX, y: 1.3, w: 6.0, h: 4.6, barGrouping: "clustered", chartColors: [HEX.accent3, HEX.accent2],
    showLegend: true, legendPos: "b", legendColor: HEX.lt2, legendFontSize: 12, legendFontFace: "+mn-lt",
    catAxisLabelColor: HEX.accent5, valAxisLabelColor: HEX.accent5, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, valGridLine: { color: HEX.accent6, size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: false, valAxisLineShow: false, showTitle: true, titleColor: HEX.lt2, titleFontSize: 15, titleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "Tow speed (m/s)", catAxisTitleColor: HEX.accent5, catAxisTitleFontSize: 12,
    dataLabelColor: HEX.lt2, dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt",
  };
  s.addChart(pres.charts.BAR, [
    { name: "SAM2", labels: speeds, values: [2.58, 2.90, 2.82, 2.58, 2.43, 2.15] },
    { name: "CoTracker3", labels: speeds, values: [0.50, 0.62, 0.58, 0.55, 0.57, 0.60] },
  ], { ...chartBase, title: "3D jitter per knot, median (mm)", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 3.5, objectName: "jitter chart" });
  s.addChart(pres.charts.BAR, [
    { name: "SAM2", labels: speeds, values: [4.3, 4.8, 19.2, 118.3, 245.9, 397.1] },
    { name: "CoTracker3", labels: speeds, values: [1.2, 1.7, 1.4, 1.0, 5.0, 2.5] },
  ], { ...chartBase, x: 6.73, title: "Worst knot-to-knot error (mm, log scale)", valAxisLogScaleBase: 10, valAxisMinVal: 0.5, valAxisMaxVal: 1000, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", objectName: "worst edge chart" });
  text(s, "Knots reconstructed in 3D: SAM2 96–98%, CoTracker3 99% at every speed. Mean of repeats a and b; 0.05 m/s uses repeat b only.", { x: MX, y: 6.1, w: W - 2 * MX, h: 0.6, fontSize: 14, color: C.accent5 });
  s.addNotes("Jitter: median second difference of each knot's 3D path. Worst knot-to-knot error: the largest temporal standard deviation of any grid edge's 3D length, worst of the two repeats. SAM2's blow-ups start at 0.15 m/s and grow with speed, which matches the paper's Table 2 where tracking success drops at higher speeds. CoTracker3's typical edge wobble does rise from 0.10 to 0.20 mm with speed, probably motion blur or real stretch.");
}

// ---------- Section 3: occlusion ----------
section("Occlusion");
{
  const s = slide("SECTION");
  text(s, "03", { x: MX, y: 2.75, w: 1.4, h: 1.0, fontSize: 54, bold: true, fontFace: THEME.headFontFace, color: C.accent1, valign: "middle" });
  s.addText("Occlusion", { placeholder: "title" });
  s.addText("Hiding knots in real footage to see whether the tracker can fill the gap", { placeholder: "body" });
}
{
  const s = slide("CONTENT");
  s.addText("A hidden knot: SAM2 has nothing, CoTracker3 has an estimate", { placeholder: "title" });
  const cols = [
    [C.accent3, "SAM2", [
      "Propagates a mask per knot from a memory of past masks.",
      "A covered knot gives an empty mask, so there is no position at all for those frames.",
      "When the knot comes back, the memory can attach to whatever blob is nearest. On real footage at 0.30 m/s, knot 115 came back one row off after 150 frames.",
    ]],
    [C.accent2, "CoTracker3", [
      "Estimates every point in every frame, using the point's own path and the motion of all the other knots.",
      "A covered knot keeps moving with its neighbours, so the estimate follows the net's deflection.",
      "It also lowers the knot's visibility score, so the estimate can be stored as inferred and excluded or flagged later.",
    ]],
  ];
  cols.forEach(([col, name, lines], i) => {
    const x = MX + i * 6.15, w = 5.9;
    card(s, x, 1.45, w, 4.55, `${name} card`);
    knot(s, x + 0.35, 1.78, 0.26, col);
    text(s, name, { x: x + 0.8, y: 1.68, w: 4.5, h: 0.45, fontSize: 22, bold: true, fontFace: THEME.headFontFace, color: col, valign: "middle" });
    text(s, lines.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < lines.length - 1, paraSpaceAfter: 10 } })), { x: x + 0.35, y: 2.4, w: w - 0.7, h: 3.4, fontSize: 15 });
  });
  text(s, "Not yet measured: SAM2 on the occluded clips. The SAM2 column describes how the model works plus what we saw on real footage; a direct run is on the next-steps list.", { x: MX, y: 6.15, w: W - 2 * MX, h: 0.6, fontSize: 13, color: C.accent5, italic: true });
  s.addNotes("Be clear here: the occlusion test was run with CoTracker3 only. SAM2 needs the GPU box; running it on the same edited clips is the obvious next step for a fair comparison. The real-footage evidence for SAM2 is the knot 115 dropout in run 006.");
}
{
  const s = slide("CONTENT");
  s.addText("The test: paint occluders into real runs and rerun", { placeholder: "title" });
  const steps = [
    ["Paint", "Dark patches drawn into the left camera of run 006 (0.30 m/s) and run 003 (0.15 m/s), mostly during the deflection."],
    ["Rerun", "CoTracker3 tracks the edited video; the right camera keeps its normal tracks, as if one lens were blocked."],
    ["Score", "Each covered knot is compared with where it really was: the same run without the patch."],
  ];
  steps.forEach(([t, d], i) => {
    const y = 1.45 + i * 1.32;
    s.addShape(pres.shapes.OVAL, { x: MX, y, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `test step ${i + 1}` });
    text(s, String(i + 1), { x: MX, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.text1, align: "center", valign: "middle" });
    text(s, t, { x: MX + 0.75, y: y - 0.02, w: 4.6, h: 0.45, fontSize: 19, bold: true, fontFace: THEME.headFontFace });
    text(s, d, { x: MX + 0.75, y: y + 0.42, w: 4.7, h: 0.85, fontSize: 14, color: C.accent5 });
  });
  const kinds = [
    ["Moves with the net", "Debris or fouling on the netting. 1 knot for 0.5 s; 1, 2×2, 3×3 and 4×4 knots for 2 s; 3×3 until the end of the clip."],
    ["Fixed patch", "Something on the camera housing or a line in front of the lens. The net slides under it for 3 s."],
    ["Passing fish", "A dark ellipse sweeping across the grid in 2 s, between camera and net."],
  ];
  kinds.forEach(([t, d], i) => {
    const y = 1.45 + i * 1.5, x = 6.55, w = 6.18;
    card(s, x, y, w, 1.3, `kind ${i + 1}`);
    text(s, t, { x: x + 0.3, y: y + 0.15, w: w - 0.6, h: 0.4, fontSize: 17, bold: true, fontFace: THEME.headFontFace, color: C.accent1 });
    text(s, d, { x: x + 0.3, y: y + 0.55, w: w - 0.6, h: 0.7, fontSize: 14 });
  });
  text(s, "Measured: distance from the true knot (px and 3D mm), how often the model flags the knot hidden, whether it comes back on the right knot, and how many 3D points the reprojection check rejects.", { x: MX, y: 5.95, w: W - 2 * MX, h: 0.7, fontSize: 14, color: C.accent5 });
  s.addNotes("8 setups per run, 16 in total. The reference is CoTracker3 plus snap on the unedited run, consistent to about 1 to 2 px. The patch edge sits about halfway to the next uncovered knot.");
}

function occSlide(name, title, caption, notes) {
  const s = slide("CONTENT");
  s.addText(title, { placeholder: "title" });
  video(s, name, { x: MX, y: 1.25, w: W - 2 * MX, h: 4.95 });
  text(s, "Left: the video the tracker saw.   Right: the same frames with the patch made see-through, true positions as white rings.", { x: MX, y: 6.2, w: W - 2 * MX, h: 0.3, fontSize: 12, color: C.accent5 });
  legendRow(s, MX, 6.55, [["tracked, visible", C.accent2], ["inferred while hidden", C.accent1], ["on the wrong knot", C.accent4]]);
  text(s, caption, { x: 7.0, y: 6.5, w: W - MX - 7.0, h: 0.4, fontSize: 14, bold: true, align: "right", valign: "middle" });
  s.addNotes(notes);
}
occSlide("occ_fish", "Passing fish: the grid is never lost", "median 11.8 px · 3D 3.2 mm",
  "A dark ellipse crosses row 7 in 2 s at 0.30 m/s. 115 knot-frames covered, 99% flagged hidden. Median error 11.8 px, about 4% of the knot spacing, 3.2 mm after triangulation. Every knot returns on its own knot.");
occSlide("occ_block4", "4×4 block for 2 s during the deflection", "median 24 px · 3D 4.8 mm",
  "16 knots covered for 2 s while they move about 300 px. Median error 24 px, a tenth of the spacing; 3D 4.8 mm. One failure: knot 88, the bottom-left corner, drifts up onto knot 76, comes back marked visible and stays on the wrong knot for about 2 s before recovering. At 0.15 m/s the same block has a median of 11 px and no swaps.");
occSlide("occ_static", "Fixed patch: knots slide in and out under it", "median 17.7 px · 3D 4.1 mm",
  "The patch stays fixed in the image for 3 s while the net slides underneath, so knots disappear and reappear at different times. 898 knot-frames covered, median 17.7 px, 3D 4.1 mm, no swaps.");
{
  const s = slide("CONTENT");
  s.addText("Long gaps: fine at 0.15 m/s, drifts at 0.30 m/s", { placeholder: "title" });
  const box = { x: MX, y: 1.25, w: 7.9, h: 2.72 };
  video(s, "occ_rest_015", box, "left");
  video(s, "occ_rest_030", { ...box, y: 4.13 }, "left");
  const x = 8.85, w = W - MX - x;
  text(s, "3×3 block hidden from frame 25 to the end, about 6 s.", { x, y: 1.3, w, h: 0.75, fontSize: 15 });
  text(s, "0.15 m/s (top)", { x, y: 2.2, w, h: 0.35, fontSize: 17, bold: true, fontFace: THEME.headFontFace, color: C.accent2 });
  text(s, "Median 16.8 px, 3D 3.5 mm. Every knot stays on its own knot.", { x, y: 2.55, w, h: 0.9, fontSize: 14 });
  text(s, "0.30 m/s (bottom)", { x, y: 4.13, w, h: 0.35, fontSize: 17, bold: true, fontFace: THEME.headFontFace, color: C.accent4 });
  text(s, "Median 91 px; a third of the guesses sit nearer a neighbour. The reprojection check rejects 73% of them in 3D, because the right camera still sees the real knot.", { x, y: 4.48, w, h: 1.6, fontSize: 14 });
  s.addNotes("This is the limit we found: a long gap during fast motion. Even then the model flags the knots hidden throughout, and the existing 20 px reprojection gate catches most of the bad estimates.");
}
{
  const s = slide("CONTENT");
  s.addText("Gaps up to 2 s cost about 3–5 mm in 3D", { placeholder: "title" });
  const rows = [
    ["Occlusion", "Covered\nknot-frames", "Flagged\nhidden", "Median error\n0.30 m/s", "Median error\n0.15 m/s", "3D median\n0.30 m/s", "3D median\n0.15 m/s", "Came back on\nwrong knot"],
    ["1 knot, 0.5 s", "15", "100%", "4.9 px", "7.3 px", "2.4 mm", "4.2 mm", "none"],
    ["1 knot, 2 s", "60", "100%", "15.9 px", "6.2 px", "5.0 mm", "3.3 mm", "none"],
    ["2×2 knots, 2 s", "240", "100%", "15.2 px", "8.3 px", "3.5 mm", "2.9 mm", "none"],
    ["3×3 knots, 2 s", "540", "100%", "17.5 px", "8.2 px", "5.1 mm", "2.6 mm", "none"],
    ["4×4 knots, 2 s", "960", "100%", "24.2 px", "11.5 px", "4.8 mm", "4.1 mm", "1 knot at 0.30"],
    ["3×3 knots, to end", "1602", "99.9%", "91.0 px", "16.8 px", "8.5 mm", "3.5 mm", "no reappearance"],
    ["Fixed patch, 3 s", "898 / 810", "99.9%", "17.7 px", "7.2 px", "4.1 mm", "2.4 mm", "none"],
    ["Passing fish, 2 s", "115 / 118", "98%", "11.8 px", "8.2 px", "3.2 mm", "2.8 mm", "none"],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({
    text: c, options: {
      bold: i === 0 || j === 0, align: j ? "right" : "left",
      color: i === 0 ? C.accent5 : (i === 6 && (j === 3 || j === 5)) || (i === 5 && j === 7) ? C.accent4 : C.background2,
      fill: { color: i === 0 ? C.text2 : C.text1 }, fontSize: i === 0 ? 11 : 13,
    },
  }))), {
    x: MX, y: 1.3, w: W - 2 * MX, colW: [2.3, 1.35, 1.1, 1.45, 1.45, 1.35, 1.35, 1.78], rowH: 0.47, valign: "middle", margin: 0.06,
    border: { type: "solid", pt: 0.75, color: HEX.accent6 }, color: C.background2, objectName: "occlusion table",
  });
  text(s, "Knot spacing is 225–290 px, so the 2 s gaps stay under a tenth of a spacing. The model flagged fewer than 0.3% of uncovered knots as hidden. Red marks the two cases that broke.", { x: MX, y: 5.95, w: W - 2 * MX, h: 0.7, fontSize: 14, color: C.accent5 });
  s.addNotes("Flagged hidden is averaged over both speeds. 3D error is against the unoccluded triangulation, for the covered knot-frames that pass the 20 px reprojection gate. For comparison, the paper's FEM agreement is 4.88 mm RMSE.");
}

// ---------- Close ----------
section("Next steps");
{
  const s = slide("CONTENT");
  s.addText("What is not settled yet", { placeholder: "title" });
  const items = [
    ["No hand-labelled ground truth", "The reference is CoTracker3 on clean footage, checked by eye. A few hundred hand-clicked knots would make the accuracy claims independent."],
    ["SAM2 not run on the occluded clips", "Needs the GPU box. Without it, the occlusion comparison rests on the mechanism and the knot 115 example."],
    ["Confident wrong knot", "Knot 88 came back on its neighbour while marked visible. A duplicate or grid-spacing check would catch this."],
    ["Local GPU crashes", "The RTX 5070 blue-screens under CoTracker3 (hypervisor error). Everything here ran on CPU; an IT ticket is the likely fix."],
    ["Run 001a setup", "Its setup prompts are misplaced in both cameras, so both trackers fail there. It needs a new setup before rerunning."],
  ];
  items.forEach(([t, d], i) => {
    // Two columns of two, then the last item across the full width.
    const col = i % 2, row = Math.floor(i / 2), last = i === items.length - 1;
    const x = last ? MX : MX + col * 6.15, y = 1.4 + row * 1.75, w = last ? W - 2 * MX : 5.9;
    card(s, x, y, w, 1.55, `limit ${i + 1}`);
    knot(s, x + 0.3, y + 0.3, 0.2, i < 3 ? C.accent1 : C.accent5);
    text(s, t, { x: x + 0.7, y: y + 0.2, w: w - 1.0, h: 0.4, fontSize: 17, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    text(s, d, { x: x + 0.7, y: y + 0.65, w: w - 1.0, h: 0.85, fontSize: 13, color: C.accent5 });
  });
  s.addNotes("These are the honest gaps. None of them changes the main result on the existing runs, but the first three matter for how strongly the paper can phrase the occlusion claim.");
}
{
  const s = slide("CONTENT");
  s.addText("Proposal for the revision", { placeholder: "title" });
  const steps = [
    ["Swap the tracker", "Use CoTracker3 plus the full-resolution snap as the tracking stage; rerun all trial 15 and modified-net runs. About 2–3 minutes per run on CPU."],
    ["Harden the output", "Add a duplicate and grid-spacing check, and store each knot-frame as measured or inferred."],
    ["Close the comparison", "Run SAM2 on the same occluded clips, and hand-label a few hundred knots as an independent check."],
    ["Add an occlusion section", "Report the synthetic study; optionally one tank run with an object passing in front of the net."],
  ];
  steps.forEach(([t, d], i) => {
    const y = 1.45 + i * 1.28;
    s.addShape(pres.shapes.OVAL, { x: MX, y, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `proposal ${i + 1}` });
    text(s, String(i + 1), { x: MX, y, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.text1, align: "center", valign: "middle" });
    text(s, t, { x: MX + 0.9, y: y - 0.02, w: 6.6, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace });
    text(s, d, { x: MX + 0.9, y: y + 0.45, w: 6.9, h: 0.75, fontSize: 14, color: C.accent5 });
  });
  card(s, 8.6, 1.45, 4.13, 4.85, "why now");
  text(s, "Why it is worth it", { x: 8.9, y: 1.7, w: 3.6, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.accent2 });
  const why = [
    "Fixes the identity errors that grow with tow speed",
    "4–5× less 3D noise on the same footage",
    "Runs on a laptop CPU, no remote GPU",
    "Turns occlusion from a limitation into a tested capability",
  ];
  text(s, why.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < why.length - 1, paraSpaceAfter: 12 } })), { x: 8.9, y: 2.35, w: 3.6, h: 3.7, fontSize: 15 });
  s.addNotes("The order matters: the tracker swap is cheap and already done for trial 15. The checks and the SAM2 occlusion run are what make the occlusion claim publishable.");
}

(async () => {
  await pres.writeFile({ fileName: OUTFILE });
  await applyTheme(OUTFILE, THEME);
  console.log("[OK] " + OUTFILE + " " + (fs.statSync(OUTFILE).size / 1e6).toFixed(1) + " MB");
})();
