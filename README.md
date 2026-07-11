# Net Deformation Tracking

Stereo aquaculture net deformation tracking pipeline for GoPro left/right videos.

## What Is Tracked In Git

This repository tracks code and lightweight configuration only. Raw videos,
archives, model weights, extracted frames, debug images, outputs, and local work
folders are ignored by `.gitignore`.

On a new machine, copy or place trial videos into the expected local input
folders as needed. Do not commit raw recordings.

## Environment

Use Python 3.11. This is the version the current SAM2 workflow has been tested
with locally.

### Windows

From an Anaconda/Miniconda PowerShell:

```powershell
conda create -n sam2py311 python=3.11 -y
conda activate sam2py311
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy rich scipy scikit-image pandas matplotlib pillow hydra-core iopath huggingface_hub
cd sam2\sam2
pip install -e .
```

Optional but recommended for faster frame extraction:

```powershell
winget install Gyan.FFmpeg
```

### Linux

From a terminal:

```bash
conda create -n sam2py311 python=3.11 -y
conda activate sam2py311
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy rich scipy scikit-image pandas matplotlib pillow hydra-core iopath huggingface_hub
cd sam2/sam2
pip install -e .
```

Optional but recommended for faster frame extraction:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

SAM2 model weights are not tracked in Git. The SAM2 scripts use Hugging Face
`from_pretrained(...)`, so weights should download automatically on the first
run and then stay cached on that machine. DINO external repos are also not
tracked; the DINO prompt scripts need their separate setup again if you return
to them later.

## Calibration

Put the left/right calibration videos here:

```text
calibration/in/left.mp4
calibration/in/right.mp4
```

Then run from `calibration`:

```powershell
python .\stereo_checker_debug.py sync --sync-mode audio
python .\stereo_checker_debug.py stats --step 50 --max-scan 1000 --workers 16
python .\stereo_checker_debug.py mono --max-scan 100 --workers 16
python .\stereo_checker_debug.py stereo --max-pairs 50 --workers 16
python .\stereo_checker_debug.py rectify --frame-offset 150 --alpha 0.2
```

Linux uses the same arguments with `/` paths:

```bash
python stereo_checker_debug.py sync --sync-mode audio
python stereo_checker_debug.py stats --step 50 --max-scan 1000 --workers 16
python stereo_checker_debug.py mono --max-scan 100 --workers 16
python stereo_checker_debug.py stereo --max-pairs 50 --workers 16
python stereo_checker_debug.py rectify --frame-offset 150 --alpha 0.2
```

Run sync, exhaustive stats, mono calibration, and stereo calibration
sequentially with 32 workers:

```bash
python stereo_checker_debug.py sync --sync-mode audio; python stereo_checker_debug.py stats --step 1 --max-scan 100000 --workers 32 --no-adaptive; python stereo_checker_debug.py mono --max-scan 100 --workers 32; python stereo_checker_debug.py stereo --max-pairs 50 --workers 32
```

If you want stats to scan every frame instead of adaptively narrowing around
detections, use the same worker count explicitly:

```powershell
python .\stereo_checker_debug.py stats --step 1 --max-scan 100000 --workers 16 --no-adaptive
```

```bash
python stereo_checker_debug.py stats --step 1 --max-scan 100000 --workers 16 --no-adaptive
```

By default, `stereo_checker_debug.py` uses:

```text
scale=1.0
in/left.mp4
in/right.mp4
work/sync.json
work/stats
work/mono.npz
work/stereo.npz
work/rectify
```

On Linux, `.mp4` and `.MP4` are different filenames. The calibration and SAM2
scripts will try the requested suffix plus lower/upper-case variants, but using
the standard lowercase names keeps commands predictable.

That means the file input/output flags are optional for the standard layout.
Override paths only when needed:

```text
--left
--right
--sync
--out
--mono
--stereo
--stats-dir
```

Useful calibration flags:

```text
--scale 1.0              processing scale; must match between mono, stereo, and rectify
--cols 9 --rows 7        checkerboard inner-corner count for the current board
--square-mm 40.0         checker square size
--workers 16             worker processes for stats, mono reuse, and stereo reuse paths
--reuse-stats-indices true|false
                         reuse positive detections from stats for faster mono/stereo; default true
--use-cuda               use OpenCV CUDA where supported, with CPU fallback
```

In the recommended workflow, `--step` is only used by `stats`. By default,
`mono` and `stereo` use the positive detections from `work/stats` instead of
scanning by step. If you run `mono` or `stereo` with
`--reuse-stats-indices false`, then `--step` matters again.

`stats`, `mono`, and `stereo` use Rich progress bars. Left and right detection
show separate progress where they run in parallel.

## SAM2 Tracking

From `sam2\sam2`:

```powershell
python .\run_sam2_markers.py --setup
python .\run_sam2_markers.py --semi-auto-setup
python .\run_sam2_markers.py --semi-auto-setup-sections
python .\run_sam2_markers.py --modify-setup
python .\run_sam2_markers.py --reuse-setup --scale 0.5 --gpu-mode single --single-gpu-index 0
```

`--semi-auto-setup` asks for grid columns/rows, then asks you to click the
outer marker-grid corners on LEFT and RIGHT in this order: top-left, top-right,
bottom-left, bottom-right. It detects the painted marker centroids, fills any
misses from the grid estimate, and opens an editable review window before
saving the normal setup package.

`--semi-auto-setup-sections` is for irregular nets that can be covered by
multiple local grid patches. For each section, the four local marker-grid
corners are ordered top-left, top-right, bottom-left, bottom-right. When a
fixed `--section-layout` is provided, all section corners are clicked in one
pass: for two sections, points `0-3` are section 1 and points `4-7` are section
2. Review/edit the generated prompts, then add another section or finish.
Sections are appended into one matched setup package; overlapping points that
land on the same marker in both cameras are skipped.

In `--modify-setup`, existing prompts use the same two-click move workflow:
left-click a prompt to select it, left-click the corrected marker center to
place it, and press `n` before a left-click when you want to add a new prompt.

Memory-friendly batched run:

```powershell
python .\run_sam2_objectwise.py --scale 0.5 --gpu-mode single --single-gpu-index 0 --batch-size 24
```

Useful SAM2 flags:

```text
--gpu-mode auto|single|dual|4090-only
--single-gpu-index N    CUDA device for single-GPU mode
--batch-size N          objectwise only; objects per SAM2 run
--preview true|false    live preview/correction UI
--scale 0.5             processing scale after crop
```

Dual-GPU mode runs left and right in separate GPU worker processes while the
main process owns the OpenCV preview/correction windows. Rich progress is used
for propagation and frame loading; left/right frame-loading bars are labeled
separately when both sides run.

Both SAM2 scripts write triangulation-compatible 2D tracks:

```text
sam2/sam2/out/left/tracks_2d.csv
sam2/sam2/out/right/tracks_2d.csv
```

## Queued Runs

Use the queue runner to split one raw stereo recording, calibrate, track every
experiment, and triangulate the results:

```bash
python run_pipeline_queue.py
```

It asks once for the raw LEFT and RIGHT videos, then opens the Linux dual-video
splitter using RIGHT as the cutting reference. Mark the clips in this order:

1. Checkerboard calibration clip.
2. First experiment clip.
3. Remaining experiment clips.

Clip pair 1 is reserved for calibration and is never sent to SAM2. The runner
asks only for velocity labels for clip pairs 2 onward. It then runs calibration
on clip pair 1 using `sync`, exhaustive `stats`, `mono`, and `stereo` with the
9x7 checkerboard, 40 mm squares, scale 1.0, and 32 workers. Calibration outputs
are isolated inside the queue directory and passed explicitly to every
triangulation run. Audio sync uses whole-range frame-level correlation and asks
you to confirm the proposed LEFT-minus-RIGHT offset before continuing.

The splitter does not create or reencode clip videos for queued runs. It stores
the original stereo paths plus inclusive start and exclusive end frame indices
in `split_manifest.json`. Calibration and SAM2 read those ranges directly from
the originals. SAM2's working frame cache uses high-quality JPEG frames, so the
only geometric reduction is the explicitly requested processing scale.

After calibration, the runner opens semi-auto setup for every experiment so
all manual marker adjustments can be finished in one pass. Queue creation asks
whether setup should use one rectangular grid or sectioned local grids. In
sectioned mode, it asks once for the number of sections and each section's
columns/rows, then derives the full visualization grid from those stacked
sections. That physical layout is reused for every experiment clip, so setup
collects all section corners in one pass before prompt review. It then processes each
prepared run unattended with the queue's SAM2 settings. The default queue
settings are:

```text
--scale 0.2 --gpu-mode dual --batch-size 36 --preview false
```

To test a non-default SAM2 scale, create the queue with an explicit override:

```bash
python run_pipeline_queue.py --sam2-scale 0.25
```

If you explicitly resume a queue with a different scale, stale SAM2/3D outputs
are invalidated by the queue fingerprints and recomputed:

```bash
python run_pipeline_queue.py --resume work/pipeline_queue/<queue-id> --sam2-scale 0.25
```

LEFT runs on CUDA 0 and RIGHT runs on CUDA 1. After each objectwise run,
triangulation writes the CSV, summary, and interactive Three.js viewer to:

```text
triangulation/results/<velocity>/
```

The virtual split manifest, calibration products, queue state, isolated setup
files, SAM2 outputs, and logs are retained under:

```text
work/pipeline_queue/<queue-id>/
```

If processing is interrupted, resume from its manifest without repeating
completed setups or batches:

```bash
python run_pipeline_queue.py --resume work/pipeline_queue/<queue-id>
```

Resume checks are data-aware. The runner reuses selected virtual frame ranges,
validates each calibration stage against the original videos, exact range, and
dependencies, verifies each setup belongs to its source videos and range,
verifies objectwise batch fingerprints, frame cache scale, and every expected
`(frame, object)` track key, and fingerprints triangulation inputs before reusing
`--viz-only`. If tracks, source videos, ranges, calibration, sync JSON, setup
points, corrections, grid size, or queue settings change, the affected stage
is recomputed instead of reusing stale files.

The setup and objectwise scripts also accept `--corrections-json` for isolated
correction files. Semi-auto setup accepts `--grid-cols` and `--grid-rows` to
skip its grid-size prompts.

## Triangulation

From the repository root:

```powershell
python .\triangulation\points_to_3d.py
python .\triangulation\points_to_3d.py --visualize
python .\triangulation\points_to_3d.py --visualize --workers 16 --viz-encoder auto
python .\triangulation\points_to_3d.py --viz-only --visualize --workers 16
python .\triangulation\points_to_3d.py --sync-mode audio --visualize --workers 16
```

By default, triangulation uses:

```text
calibration/work/stereo.npz
calibration/work/sync.json
sam2/sam2/out/left/tracks_2d.csv
sam2/sam2/out/right/tracks_2d.csv
```

Useful triangulation flags:

```text
--quality-min 0.0       minimum 2D track quality before triangulation
--max-reproj 20.0       maximum mean reprojection error in pixels
--visualize             write verification videos
--viz-only              visualize existing --out-csv without triangulating again
--viz-mode scene        scene writes *_left.mp4, *_iso.mp4, *_topdown.mp4, and *_viewer.html
--workers 0             scene visualization workers; 0=auto CPU count
--viz-encoder auto      auto chooses NVENC when ffmpeg is present, else mp4v
--viz-grid-cols 0       marker grid columns for net connections; 0=auto
--viz-grid-rows 0       marker grid rows for net connections; 0=auto
```

The `*_viewer.html` output is an interactive Three.js view of the same local
net frame used by the iso video. Open it in a browser to orbit, pan, zoom, scrub
frames, and inspect the deformation-colored net without rerendering videos.

To compare multiple triangulation runs in one interactive 3D viewer:

```powershell
python .\triangulation\view_3d_runs.py `
  .\triangulation\run_a\triangulated_3d.csv `
  .\triangulation\run_b\triangulated_3d.csv `
  --labels run-a,run-b `
  --out .\triangulation\out\multi_run_3d_viewer.html
```

Each run is converted into the same kind of local net frame as the iso
visualization, then overlaid with per-run visibility toggles and a shared frame
slider.

## Video Splitting

Use `dual_video_splitter.py` to split left/right videos at matching frame IDs.

```powershell
python .\dual_video_splitter.py
```

On Linux, use the GoPro IMU-aware splitter:

```bash
python dual_video_splitter_linux.py
```

The right video is the cutting reference. Its embedded GoPro accelerometer and
gyroscope activity is displayed along the bottom of the picker; press `n` to
jump to the next detected acceleration event with 0.5 seconds of pre-roll.
