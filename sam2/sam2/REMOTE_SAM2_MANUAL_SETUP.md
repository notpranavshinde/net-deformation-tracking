# Manual SAM2 Remote Workflow

Run interactive setup locally:

```powershell
python run_sam2_markers.py --setup
```

Then copy the needed files to the GPU machine and run headless SAM2 there.
The script does not SSH, SCP, poll, or fetch results by itself.

Remote headless command, from the remote `sam2/sam2` directory:

```bash
~/vtorch/bin/python run_sam2_markers.py --reuse-setup --save-masks false --save-overlay false --save-tracks true --preview false --gpu-mode dual
```

Outputs:

```text
out/left/tracks_2d.csv
out/right/tracks_2d.csv
```

