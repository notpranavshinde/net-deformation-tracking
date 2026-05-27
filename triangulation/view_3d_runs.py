r"""
Build an interactive Three.js overlay viewer for multiple triangulation runs.

Example:
    python triangulation/view_3d_runs.py \
        triangulation/run_a/triangulated_3d.csv \
        triangulation/run_b/triangulated_3d.csv \
        --labels run-a,run-b \
        --out triangulation/out/multi_run_viewer.html

Each run is converted into its own local net frame using the same reference
frame logic as points_to_3d.py, then overlaid in one browser viewer.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triangulation.points_to_3d import (  # noqa: E402
    _attach_displacements,
    _build_net_reference_frame,
    _build_reference_edges,
    _choose_iso_azim,
    _to_local_xyz,
)


RUN_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#444444",
]


def _load_rows(path: Path):
    df = pd.read_csv(path)
    required = ["frame_L", "obj_id", "X", "Y", "Z", "valid_3d"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows = df.to_dict(orient="records")
    valid_rows = [r for r in rows if int(r["valid_3d"]) == 1]
    if not valid_rows:
        raise ValueError(f"{path} has no valid_3d rows.")
    return valid_rows


def _pad_lim(values):
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    pad = 0.1 * (hi - lo + 1e-6)
    return lo - pad, hi + pad


def _build_run_payload(path: Path, label: str, color: str, max_frames: int, grid_cols: int, grid_rows: int):
    valid_rows = _load_rows(path)
    refs, fixed_ids, disp_max = _attach_displacements(valid_rows)
    origin, basis, ref_local, fixed_local = _build_net_reference_frame(refs, fixed_ids)
    edges, grid_shape = _build_reference_edges(refs, ref_local, grid_cols=grid_cols, grid_rows=grid_rows)

    grouped = {}
    for row in valid_rows:
        grouped.setdefault(int(row["frame_L"]), []).append(row)
    frame_ids = sorted(grouped)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    frames = []
    local_points = []
    for frame_id in frame_ids:
        rows = grouped[frame_id]
        ids = []
        xs = []
        ys = []
        zs = []
        ds = []
        for row in rows:
            local = _to_local_xyz([float(row["X"]), float(row["Y"]), float(row["Z"])], origin, basis)
            ids.append(int(row["obj_id"]))
            xs.append(float(local[0]))
            ys.append(float(local[1]))
            zs.append(float(local[2]))
            ds.append(float(row.get("_disp_m", 0.0)))
            local_points.append(local)
        frames.append({
            "frame": int(frame_id),
            "ids": ids,
            "x": xs,
            "y": ys,
            "z": zs,
            "d": ds,
        })

    iso_azim = _choose_iso_azim(ref_local, fixed_local)
    local_points = np.asarray(local_points, dtype=np.float64)
    limits = {
        "x": _pad_lim(local_points[:, 0]),
        "y": _pad_lim(local_points[:, 1]),
        "z": _pad_lim(local_points[:, 2]),
    }
    return {
        "label": label,
        "path": str(path),
        "color": color,
        "frames": frames,
        "edges": [[int(a), int(b)] for a, b in edges],
        "dispMax": float(max(disp_max, 1e-9)),
        "isoAzim": float(iso_azim),
        "grid": {"cols": int(grid_shape[0]), "rows": int(grid_shape[1])},
        "limits": {k: [float(v[0]), float(v[1])] for k, v in limits.items()},
        "frameCount": len(frames),
        "nodeCount": len(refs),
        "fixedCount": len(fixed_ids),
    }


def _merge_limits(runs):
    out = {}
    for axis in ("x", "y", "z"):
        lo = min(run["limits"][axis][0] for run in runs)
        hi = max(run["limits"][axis][1] for run in runs)
        out[axis] = [float(min(lo, 0.0)), float(max(hi, 0.0))]
    return out


def _write_viewer(out_path: Path, runs):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_disp = max(run["dispMax"] for run in runs)
    max_frames = max(run["frameCount"] for run in runs)
    iso_azim = runs[0]["isoAzim"]
    payload = {
        "runs": runs,
        "limits": _merge_limits(runs),
        "dispMax": float(max_disp),
        "maxFrames": int(max_frames),
        "isoAzim": float(iso_azim),
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Triangulated 3D Multi-Run Viewer</title>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #fff; font-family: Arial, sans-serif; color: #111; }}
    #viewer {{ width: 100vw; height: 100vh; display: block; }}
    #hud {{ position: fixed; left: 14px; top: 12px; background: rgba(255,255,255,0.92); border: 1px solid #cfcfcf; padding: 10px 12px; min-width: 440px; max-width: 620px; }}
    #row {{ display: flex; gap: 8px; align-items: center; }}
    button {{ height: 28px; min-width: 58px; border: 1px solid #999; background: #f6f6f6; color: #111; cursor: pointer; }}
    input[type=range] {{ width: 260px; }}
    #runs {{ margin-top: 8px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 5px 12px; font-size: 13px; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }}
    #meta {{ margin-top: 7px; font-size: 13px; line-height: 1.35; }}
    #colorbar {{ position: fixed; right: 24px; top: 64px; width: 22px; height: 240px; border: 1px solid #555; background: linear-gradient(to top, #440154, #3b528b, #21918c, #5ec962, #fde725); }}
    #cmax, #cmin {{ position: fixed; right: 54px; font-size: 12px; color: #111; }}
    #cmax {{ top: 62px; }}
    #cmin {{ top: 292px; }}
    #clabel {{ position: fixed; right: 6px; top: 312px; font-size: 12px; writing-mode: vertical-rl; transform: rotate(180deg); color: #111; }}
  </style>
</head>
<body>
<canvas id="viewer"></canvas>
<div id="hud">
  <div id="row">
    <button id="play">Play</button>
    <input id="slider" type="range" min="0" max="0" value="0">
    <span id="frameLabel">frame</span>
  </div>
  <div id="runs"></div>
  <div id="meta"></div>
</div>
<div id="colorbar"></div>
<div id="cmax"></div>
<div id="cmin">0.000 m</div>
<div id="clabel">|Δpos from t0| (m)</div>
<script type="application/json" id="scene-data">{data_json}</script>
<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
  }}
}}
</script>
<script type="module">
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import {{ OrbitControls }} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const data = JSON.parse(document.getElementById('scene-data').textContent);
const canvas = document.getElementById('viewer');
const renderer = new THREE.WebGLRenderer({{canvas, antialias: true}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xffffff, 1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const lx = data.limits.x, ly = data.limits.y, lz = data.limits.z;
const center = new THREE.Vector3((lx[0]+lx[1])/2, (ly[0]+ly[1])/2, (lz[0]+lz[1])/2);
const radius = Math.max(lx[1]-lx[0], ly[1]-ly[0], lz[1]-lz[0], 0.1);
controls.target.copy(center);
const az = THREE.MathUtils.degToRad(data.isoAzim);
camera.position.set(center.x + radius * Math.cos(az), center.y - radius * 1.25, center.z + radius * 0.75);
camera.near = Math.max(radius / 1000, 0.0001);
camera.far = radius * 100;
camera.updateProjectionMatrix();
controls.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0xb0b0b0, 2.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.5);
dir.position.set(2, -3, 5);
scene.add(dir);

const grid = new THREE.GridHelper(radius * 2.2, 12, 0xbdbdbd, 0xd9d9d9);
grid.rotation.z = Math.PI / 2;
grid.position.set(0, center.y, center.z);
scene.add(grid);
const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(Math.max(ly[1]-ly[0], 0.01), Math.max(lz[1]-lz[0], 0.01)),
  new THREE.MeshBasicMaterial({{color: 0xdcdcdc, transparent: true, opacity: 0.18, side: THREE.DoubleSide}})
);
plane.rotation.y = Math.PI / 2;
plane.position.set(0, center.y, center.z);
scene.add(plane);
const axes = new THREE.AxesHelper(radius * 0.55);
axes.position.copy(center);
scene.add(axes);

function viridis(t) {{
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.267,0.005,0.329], [0.231,0.322,0.545], [0.129,0.569,0.549], [0.369,0.788,0.384], [0.992,0.906,0.145]];
  const p = t * (stops.length - 1);
  const i = Math.min(Math.floor(p), stops.length - 2);
  const f = p - i;
  return new THREE.Color(
    stops[i][0] * (1 - f) + stops[i + 1][0] * f,
    stops[i][1] * (1 - f) + stops[i + 1][1] * f,
    stops[i][2] * (1 - f) + stops[i + 1][2] * f
  );
}}

function mixColor(a, b, amount) {{
  return a.clone().lerp(b, amount);
}}

const sphereGeo = new THREE.SphereGeometry(Math.max(radius * 0.008, 0.0025), 16, 10);
const runScenes = [];
for (let r = 0; r < data.runs.length; r++) {{
  const run = data.runs[r];
  const group = new THREE.Group();
  scene.add(group);
  const meshes = new Map();
  const materials = new Map();
  const ids = new Set();
  for (const frame of run.frames) for (const id of frame.ids) ids.add(id);
  const tint = new THREE.Color(run.color);
  for (const id of ids) {{
    const mat = new THREE.MeshStandardMaterial({{color: tint, roughness: 0.45, metalness: 0.0}});
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.visible = false;
    meshes.set(id, mesh);
    materials.set(id, mat);
    group.add(mesh);
  }}
  const lineMaterial = new THREE.LineBasicMaterial({{color: tint, transparent: true, opacity: 0.72}});
  const lineGeometry = new THREE.BufferGeometry();
  const linePositions = new Float32Array(Math.max(run.edges.length * 2 * 3, 6));
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  lineGeometry.setDrawRange(0, 0);
  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  group.add(lines);
  runScenes.push({{run, group, meshes, materials, lineGeometry, linePositions}});
}}

const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const frameLabel = document.getElementById('frameLabel');
const meta = document.getElementById('meta');
const runsDiv = document.getElementById('runs');
document.getElementById('cmax').textContent = data.dispMax.toFixed(3) + ' m';
slider.max = Math.max(data.maxFrames - 1, 0);

for (let r = 0; r < data.runs.length; r++) {{
  const run = data.runs[r];
  const label = document.createElement('label');
  label.innerHTML = `<input type="checkbox" checked data-run="${{r}}"> <span class="swatch" style="background:${{run.color}}"></span>${{run.label}}`;
  runsDiv.appendChild(label);
}}
runsDiv.addEventListener('change', (e) => {{
  if (e.target.matches('input[type=checkbox]')) {{
    const idx = Number(e.target.dataset.run);
    runScenes[idx].group.visible = e.target.checked;
  }}
}});

let frameIndex = 0;
let playing = false;
let lastStep = 0;

function setFrame(idx) {{
  frameIndex = Math.max(0, Math.min(data.maxFrames - 1, idx));
  slider.value = frameIndex;
  let metaText = [];
  for (const state of runScenes) {{
    const run = state.run;
    const frame = run.frames[Math.min(frameIndex, run.frames.length - 1)];
    const indexById = new Map();
    for (let i = 0; i < frame.ids.length; i++) {{
      const id = frame.ids[i];
      indexById.set(id, i);
      const mesh = state.meshes.get(id);
      mesh.visible = true;
      mesh.position.set(frame.x[i], frame.y[i], frame.z[i]);
      const dispColor = viridis(frame.d[i] / data.dispMax);
      state.materials.get(id).color.copy(mixColor(dispColor, new THREE.Color(run.color), 0.28));
    }}
    for (const [id, mesh] of state.meshes) {{
      if (!indexById.has(id)) mesh.visible = false;
    }}
    let cursor = 0;
    for (const [a, b] of run.edges) {{
      const ia = indexById.get(a), ib = indexById.get(b);
      if (ia === undefined || ib === undefined) continue;
      state.linePositions[cursor++] = frame.x[ia]; state.linePositions[cursor++] = frame.y[ia]; state.linePositions[cursor++] = frame.z[ia];
      state.linePositions[cursor++] = frame.x[ib]; state.linePositions[cursor++] = frame.y[ib]; state.linePositions[cursor++] = frame.z[ib];
    }}
    state.lineGeometry.setDrawRange(0, cursor / 3);
    state.lineGeometry.attributes.position.needsUpdate = true;
    metaText.push(`${{run.label}}: frame=${{frame.frame}} nodes=${{frame.ids.length}} grid=${{run.grid.cols}}x${{run.grid.rows}}`);
  }}
  frameLabel.textContent = 'step=' + frameIndex;
  meta.textContent = metaText.join('   |   ');
}}

slider.addEventListener('input', () => setFrame(Number(slider.value)));
playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
}});

function resize() {{
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize);
resize();
setFrame(0);

function animate(ts) {{
  requestAnimationFrame(animate);
  controls.update();
  if (playing && ts - lastStep > 140) {{
    lastStep = ts;
    setFrame((frameIndex + 1) % data.maxFrames);
  }}
  renderer.render(scene, camera);
}}
requestAnimationFrame(animate);
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote multi-run 3D viewer: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Create an interactive Three.js overlay viewer for multiple triangulated 3D CSVs.")
    ap.add_argument("csv", nargs="+", type=Path, help="Triangulated CSV files from points_to_3d.py")
    ap.add_argument("--labels", default=None, help="Comma-separated labels matching the CSV order")
    ap.add_argument("--out", type=Path, default=Path("triangulation/out/multi_run_3d_viewer.html"), help="Output HTML path")
    ap.add_argument("--max-frames", type=int, default=-1, help="Max frames per run to include; -1 includes all")
    ap.add_argument("--viz-grid-cols", type=int, default=0, help="Marker grid columns for net connections. 0=auto")
    ap.add_argument("--viz-grid-rows", type=int, default=0, help="Marker grid rows for net connections. 0=auto")
    return ap.parse_args()


def main():
    args = parse_args()
    labels = None
    if args.labels:
        labels = [part.strip() for part in args.labels.split(",")]
        if len(labels) != len(args.csv):
            raise ValueError("--labels must have the same count as input CSVs")
    else:
        labels = [path.stem for path in args.csv]

    runs = []
    for idx, (path, label) in enumerate(zip(args.csv, labels)):
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        run = _build_run_payload(
            path=path,
            label=label,
            color=color,
            max_frames=args.max_frames,
            grid_cols=args.viz_grid_cols,
            grid_rows=args.viz_grid_rows,
        )
        print(
            f"[RUN] {label}: frames={run['frameCount']} nodes={run['nodeCount']} "
            f"edges={len(run['edges'])} grid={run['grid']['cols']}x{run['grid']['rows']}"
        )
        runs.append(run)

    _write_viewer(args.out, runs)


if __name__ == "__main__":
    main()
