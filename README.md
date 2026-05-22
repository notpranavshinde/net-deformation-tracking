# Net Deformation Tracking

Stereo aquaculture net deformation tracking pipeline for GoPro left/right videos.

## What Is Tracked In Git

This repository tracks code and lightweight configuration only. Raw videos,
archives, model weights, extracted frames, debug images, outputs, and local work
folders are ignored by `.gitignore`.

On a new machine, copy or place trial videos into the expected local input
folders as needed. Do not commit raw recordings.

## Environment

Core pipeline dependencies:

```powershell
pip install opencv-python numpy tqdm scipy scikit-image pandas
```

SAM2 runs need a PyTorch/CUDA environment and SAM2 model weights available
locally. Model weights are intentionally not tracked in Git.

## Calibration

From `calibration`:

```powershell
python .\stereo_checker_debug.py sync --left .\in\left.mp4 --right .\in\right.mp4 --out work\sync.json --scale 1.0 --sync-mode audio
python .\stereo_checker_debug.py stats --left .\in\left.mp4 --right .\in\right.mp4 --sync work\sync.json --out work\stats --scale 1.0 --step 50 --max_scan 1000 --cols 8 --rows 6
python .\stereo_checker_debug.py mono --left .\in\left.mp4 --right .\in\right.mp4 --sync work\sync.json --out work\mono.npz --scale 1.0 --step 2 --max_scan 10000 --cols 8 --rows 6 --reuse_stats_indices --stats_dir .\work\stats
python .\stereo_checker_debug.py stereo --left .\in\left.mp4 --right .\in\right.mp4 --sync work\sync.json --mono work\mono.npz --out work\stereo.npz --scale 1.0 --step 25 --max_pairs 50 --cols 8 --rows 6 --reuse_stats_indices --stats_dir .\work\stats
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
