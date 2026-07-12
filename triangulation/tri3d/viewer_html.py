"""Generate the standalone interactive Three.js triangulation viewer."""

import json
from pathlib import Path


def _write_threejs_viewer(out_path: Path,
                          target_frames,
                          per_frame_xyz,
                          x_lim,
                          y_lim,
                          z_lim,
                          disp_max,
                          edges,
                          iso_azim,
                          grid_shape):
    """Write a standalone browser viewer for the same local 3D scene."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    all_obj_ids = set()
    for frame_id, (ObjIds, Xs, Ys, Zs, Ds) in zip(target_frames, per_frame_xyz):
        ids = [int(v) for v in ObjIds.tolist()]
        all_obj_ids.update(ids)
        frames.append({
            "frame": int(frame_id),
            "ids": ids,
            "x": [float(v) for v in Xs.tolist()],
            "y": [float(v) for v in Ys.tolist()],
            "z": [float(v) for v in Zs.tolist()],
            "d": [float(v) for v in Ds.tolist()],
        })

    payload = {
        "frames": frames,
        "edges": [[int(a), int(b)] for a, b in edges],
        "objIds": sorted(all_obj_ids),
        "limits": {
            "x": [float(x_lim[0]), float(x_lim[1])],
            "y": [float(y_lim[0]), float(y_lim[1])],
            "z": [float(z_lim[0]), float(z_lim[1])],
        },
        "dispMax": float(max(disp_max, 1e-9)),
        "isoAzim": float(iso_azim),
        "grid": {"cols": int(grid_shape[0]), "rows": int(grid_shape[1])},
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Triangulated 3D Viewer</title>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #fff; font-family: Arial, sans-serif; color: #111; }}
    #viewer {{ width: 100vw; height: 100vh; display: block; }}
    #hud {{ position: fixed; left: 14px; top: 12px; background: rgba(255,255,255,0.90); border: 1px solid #cfcfcf; padding: 10px 12px; min-width: 360px; }}
    #row {{ display: flex; gap: 8px; align-items: center; }}
    button {{ height: 28px; min-width: 58px; border: 1px solid #999; background: #f6f6f6; color: #111; cursor: pointer; }}
    input[type=range] {{ width: 230px; }}
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
const spans = [lx[1]-lx[0], ly[1]-ly[0], lz[1]-lz[0]];
const radius = Math.max(...spans, 0.1);
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

function makeTextSprite(text, position) {{
  const cnv = document.createElement('canvas');
  cnv.width = 256; cnv.height = 96;
  const ctx = cnv.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.font = '36px Arial';
  ctx.fillText(text, 8, 56);
  const tex = new THREE.CanvasTexture(cnv);
  const mat = new THREE.SpriteMaterial({{map: tex, transparent: true}});
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(radius * 0.22, radius * 0.08, 1);
  sprite.position.copy(position);
  scene.add(sprite);
}}

const grid = new THREE.GridHelper(radius * 2.2, 12, 0xbdbdbd, 0xd9d9d9);
grid.rotation.z = Math.PI / 2;
grid.position.set(0, center.y, center.z);
scene.add(grid);

const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(Math.max(ly[1]-ly[0], 0.01), Math.max(lz[1]-lz[0], 0.01)),
  new THREE.MeshBasicMaterial({{color: 0xdcdcdc, transparent: true, opacity: 0.22, side: THREE.DoubleSide}})
);
plane.rotation.y = Math.PI / 2;
plane.position.set(0, center.y, center.z);
scene.add(plane);

const axes = new THREE.AxesHelper(radius * 0.55);
axes.position.copy(center);
scene.add(axes);
makeTextSprite('x (m)', new THREE.Vector3(center.x + radius * 0.62, center.y, center.z));
makeTextSprite('y (m)', new THREE.Vector3(center.x, center.y + radius * 0.62, center.z));
makeTextSprite('z (m)', new THREE.Vector3(center.x, center.y, center.z + radius * 0.62));

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

const sphereGeo = new THREE.SphereGeometry(Math.max(radius * 0.01, 0.003), 18, 12);
const materials = new Map();
const meshes = new Map();
for (const id of data.objIds) {{
  const mat = new THREE.MeshStandardMaterial({{color: 0x440154, roughness: 0.45, metalness: 0.0}});
  materials.set(id, mat);
  const mesh = new THREE.Mesh(sphereGeo, mat);
  mesh.visible = false;
  mesh.userData.objId = id;
  meshes.set(id, mesh);
  scene.add(mesh);
}}

const lineMaterial = new THREE.LineBasicMaterial({{color: 0x303030, transparent: true, opacity: 0.85}});
const lineGeometry = new THREE.BufferGeometry();
const linePositions = new Float32Array(Math.max(data.edges.length * 2 * 3, 6));
lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
lineGeometry.setDrawRange(0, 0);
const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
scene.add(lines);

const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const frameLabel = document.getElementById('frameLabel');
const meta = document.getElementById('meta');
document.getElementById('cmax').textContent = data.dispMax.toFixed(3) + ' m';
slider.max = Math.max(data.frames.length - 1, 0);
let frameIndex = 0;
let playing = false;
let lastStep = 0;

function setFrame(idx) {{
  frameIndex = Math.max(0, Math.min(data.frames.length - 1, idx));
  slider.value = frameIndex;
  const frame = data.frames[frameIndex];
  const indexById = new Map();
  for (let i = 0; i < frame.ids.length; i++) {{
    const id = frame.ids[i];
    indexById.set(id, i);
    const mesh = meshes.get(id);
    mesh.visible = true;
    mesh.position.set(frame.x[i], frame.y[i], frame.z[i]);
    materials.get(id).color.copy(viridis(frame.d[i] / data.dispMax));
  }}
  for (const [id, mesh] of meshes) {{
    if (!indexById.has(id)) mesh.visible = false;
  }}
  let cursor = 0;
  for (const [a, b] of data.edges) {{
    const ia = indexById.get(a), ib = indexById.get(b);
    if (ia === undefined || ib === undefined) continue;
    linePositions[cursor++] = frame.x[ia]; linePositions[cursor++] = frame.y[ia]; linePositions[cursor++] = frame.z[ia];
    linePositions[cursor++] = frame.x[ib]; linePositions[cursor++] = frame.y[ib]; linePositions[cursor++] = frame.z[ib];
  }}
  lineGeometry.setDrawRange(0, cursor / 3);
  lineGeometry.attributes.position.needsUpdate = true;
  frameLabel.textContent = 'frame=' + frame.frame;
  meta.textContent = 'nodes=' + frame.ids.length + '  edges=' + data.edges.length + '  grid=' + data.grid.cols + 'x' + data.grid.rows;
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
    setFrame((frameIndex + 1) % data.frames.length);
  }}
  renderer.render(scene, camera);
}}
requestAnimationFrame(animate);
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote interactive 3D viewer: {out_path}")
