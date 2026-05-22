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
pip install opencv-python numpy tqdm scipy scikit-image pandas matplotlib pillow hydra-core iopath huggingface_hub
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
pip install opencv-python numpy tqdm scipy scikit-image pandas matplotlib pillow hydra-core iopath huggingface_hub
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
python .\stereo_checker_debug.py stats --step 50 --max-scan 1000 --cols 8 --rows 6 --workers 16
python .\stereo_checker_debug.py mono --step 2 --max-scan 10000 --cols 8 --rows 6 --reuse-stats-indices --workers 16
python .\stereo_checker_debug.py stereo --step 25 --max-pairs 50 --cols 8 --rows 6 --reuse-stats-indices --workers 16
python .\stereo_checker_debug.py rectify --frame-offset 150 --alpha 0.2
```

Linux uses the same arguments with `/` paths:

```bash
python stereo_checker_debug.py sync --sync-mode audio
python stereo_checker_debug.py stats --step 50 --max-scan 1000 --cols 8 --rows 6 --workers 16
python stereo_checker_debug.py mono --step 2 --max-scan 10000 --cols 8 --rows 6 --reuse-stats-indices --workers 16
python stereo_checker_debug.py stereo --step 25 --max-pairs 50 --cols 8 --rows 6 --reuse-stats-indices --workers 16
python stereo_checker_debug.py rectify --frame-offset 150 --alpha 0.2
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
--cols 8 --rows 6        checkerboard inner-corner count for the current board
--square-mm 40.0         checker square size
--workers 16             multiprocessing workers for stats-index reuse paths
--reuse-stats-indices    reuse positive detections from stats for faster mono/stereo
--use-cuda               use OpenCV CUDA where supported, with CPU fallback
```

## SAM2 Tracking

From `sam2\sam2`:

```powershell
python .\run_sam2_markers.py --setup
python .\run_sam2_markers.py --modify-setup
python .\run_sam2_markers.py --reuse-setup --scale 0.5 --gpu-mode 4090-only
```

Memory-friendly batched run:

```powershell
python .\run_sam2_objectwise.py --scale 0.5 --gpu-mode 4090-only --batch-size 24
```

Dual-GPU live preview is parent-owned: GPU workers do SAM2 compute and the
main process owns the OpenCV preview/correction windows. The current preview
transport sends rendered preview frames through multiprocessing queues. If this
becomes the bottleneck on the Linux/Ada machine, keep the user workflow the same
and replace only that transport layer with `multiprocessing.shared_memory` or
another shared-memory image buffer.

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
python .\triangulation\points_to_3d.py --sync-mode audio --visualize
```

By default, triangulation uses:

```text
calibration/work/stereo.npz
calibration/work/sync.json
sam2/sam2/out/left/tracks_2d.csv
sam2/sam2/out/right/tracks_2d.csv
```

## Video Splitting

Use `dual_video_splitter.py` to split left/right videos at matching frame IDs.

```powershell
python .\dual_video_splitter.py
```
