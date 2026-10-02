#!/usr/bin/env bash
# Track every trial15 run with CoTracker3 (CPU) and compare it against SAM2.
# Skips runs whose compare.json already exists. Usage: bash experiments/cotracker3/run_all.sh
set -u
cd "$(dirname "$0")/../.."
VARIANT=offline_384x512_dtpeak
PY=.venv-cotracker/Scripts/python
mkdir -p work/cotracker3/logs
for run_dir in work/trial15_baseline/run_*; do
  run=$(basename "$run_dir")
  out=work/cotracker3/$run/$VARIANT
  if [ -f "$out/compare.json" ]; then echo "SKIP $run"; continue; fi
  ok=1
  for side in left right; do
    if [ ! -f "$out/$side/tracks_2d.csv" ]; then
      "$PY" experiments/cotracker3/track_cotracker.py --device cpu --run-dir "$run_dir" --side $side \
        --video "raw videos/trial15$side.MP4" --model-res 384x512 --refine dtpeak --out "$out/$side" \
        > "work/cotracker3/logs/${run}_$side.log" 2>&1 || { echo "FAILED $run $side"; ok=0; break; }
    fi
  done
  [ $ok = 1 ] || continue
  python experiments/cotracker3/compare.py --run-dir "$run_dir" --cotracker "$out" \
    > "work/cotracker3/logs/${run}_compare.log" 2>&1 && echo "DONE $run" || echo "FAILED $run compare"
done
echo "ALL FINISHED"
