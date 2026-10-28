#!/usr/bin/env bash
# Full lighting + occlusion sweep on a Linux GPU box (run from the workspace root, under nohup).
# Lighting: 7 conditions x 3 strengths on run 006 and run 003. Occlusion: 8 setups on one run per tow speed.
# CPU-side work runs in parallel; tc.gpu_lock lets one model call at a time onto each GPU.
# Split the job list across GPUs with SHARD/NSHARD and PAR (parallel jobs for this GPU):
#   CT3_PY=... CUDA_VISIBLE_DEVICES=0 SHARD=0 NSHARD=2 PAR=6 nohup bash experiments/cotracker3/remote_sweep.sh > work/cotracker3/sweep0.log 2>&1 &
set -u
PY=${CT3_PY:-python}
export CT3_TRI_PYTHON=$PY OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=work/cotracker3/logs
mkdir -p "$LOG"

jobs_list() {
  for run in run_006_v_0.3mps_a run_003_v_0.15mps_a; do
    for kind in dark bright uneven flicker caustics murky background; do
      for level in mild moderate severe; do
        echo "light $run $kind $level"
      done
    done
  done
  for run in run_007_v_0.05mps_b run_002_v_0.1mps_a run_003_v_0.15mps_a run_004_v_0.2mps_a run_005_v_0.25mps_a run_006_v_0.3mps_a; do
    echo "occl $run"
  done
}

run_job() {
  set -- $1
  if [ "$1" = light ]; then
    nice -n 10 "$PY" experiments/cotracker3/lighting_test.py --run "$2" --only "$3" --levels "$4" --device cuda --threads 8 \
      > "$LOG/light_${2}_${3}_${4}.log" 2>&1 && echo "DONE light $2 $3 $4 $(date +%T)" || echo "FAILED light $2 $3 $4"
  else
    nice -n 10 "$PY" experiments/cotracker3/occlusion_test.py --run "$2" --device cuda \
      > "$LOG/occl_${2}.log" 2>&1 && echo "DONE occl $2 $(date +%T)" || echo "FAILED occl $2"
  fi
}
export -f run_job
export PY LOG

echo "START $(date +%T)"
jobs_list | awk -v s="${SHARD:-0}" -v n="${NSHARD:-1}" '(NR - 1) % n == s' | xargs -P "${PAR:-6}" -I{} bash -c 'run_job "{}"'
echo "ALL FINISHED $(date +%T)"
