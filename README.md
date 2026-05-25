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
python .\run_sam2_markers.py --modify-setup
python .\run_sam2_markers.py --reuse-setup --scale 0.5 --gpu-mode single --single-gpu-index 0
```

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
--viz-mode scene        scene writes *_left.mp4, *_iso.mp4, and *_topdown.mp4
--workers 0             scene visualization workers; 0=auto CPU count
--viz-encoder auto      auto chooses NVENC when ffmpeg is present, else mp4v
--viz-grid-cols 0       marker grid columns for net connections; 0=auto
--viz-grid-rows 0       marker grid rows for net connections; 0=auto
```

## Video Splitting

Use `dual_video_splitter.py` to split left/right videos at matching frame IDs.

```powershell
python .\dual_video_splitter.py
```
