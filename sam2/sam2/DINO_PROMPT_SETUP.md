# DINO Prompt Setup

Run these from `sam2\sam2`.

## Current Default: DINOv2

DINOv2 is public and does not need Hugging Face approval.

The official DINOv2 repo is cloned here:

```text
external\dinov2
```

The first DINOv2 run downloads public weights automatically into the Torch
cache.

## Optional DINOv3 Setup

The dependency is installed in this environment for later DINOv3 use:

```powershell
python -m pip install "transformers>=4.56.0"
```

The official DINOv3 repo is cloned here:

```text
external\dinov3
```

## One-Time Hugging Face Access

DINOv3 weights are gated by Meta. You only need this if running
`--backend huggingface`.

1. Open:

```text
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
```

2. Log in and accept the model terms.

3. Option A: store a token in `sam2\sam2\.env`:

```text
HF_TOKEN=your_token_here
```

Option B: if available on the machine:

```powershell
huggingface-cli login
```

## First Run

```powershell
python .\dino_detect_sam2_prompts.py --frame 150 --select-crop --click-examples
```

## Rerun With Saved Examples

```powershell
python .\dino_detect_sam2_prompts.py --frame 150 --examples .\work\dino_prompts\examples.json
```

## Remote Machine

On the SSH machine, repeat:

```bash
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
```

Then copy this project, videos, crop metadata, and `work/dino_prompts/examples.json`
to the remote machine. The heavy DINO/SAM2 computation can then run headlessly.
