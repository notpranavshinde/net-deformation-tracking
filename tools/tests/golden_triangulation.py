"""Regenerate and compare the real-data triangulation golden outputs.

The pre-refactor baseline was captured with this exact command from the repo root::

    python triangulation/points_to_3d.py \
      --left "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/run_001_v_0.05mps_a/sam2/left/tracks_2d.csv" \
      --right "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/run_001_v_0.05mps_a/sam2/right/tracks_2d.csv" \
      --left-video "C:/Users/pshinte/Documents/net deformation tracking/raw videos/trial15left.MP4" \
      --right-video "C:/Users/pshinte/Documents/net deformation tracking/raw videos/trial15right.MP4" \
      --start-frame 1769 --end-frame 1972 \
      --stereo "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/calibration/stereo.npz" \
      --sync-json "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/calibration/sync.json" \
      --out-csv work/golden/tri_golden.csv --out-summary work/golden/tri_golden_summary.json
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "work" / "golden"
GOLDEN_CSV = GOLDEN_DIR / "tri_golden.csv"
GOLDEN_SUMMARY = GOLDEN_DIR / "tri_golden_summary.json"

COMMON_ARGS = [
    "--left", "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/run_001_v_0.05mps_a/sam2/left/tracks_2d.csv",
    "--right", "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/run_001_v_0.05mps_a/sam2/right/tracks_2d.csv",
    "--left-video", "C:/Users/pshinte/Documents/net deformation tracking/raw videos/trial15left.MP4",
    "--right-video", "C:/Users/pshinte/Documents/net deformation tracking/raw videos/trial15right.MP4",
    "--start-frame", "1769",
    "--end-frame", "1972",
    "--stereo", "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/calibration/stereo.npz",
    "--sync-json", "C:/Users/pshinte/Documents/net deformation tracking/work/trial15_baseline/calibration/sync.json",
]


def _compare_csv(expected_path: Path, actual_path: Path) -> list[str]:
    with expected_path.open(newline="", encoding="utf-8") as expected_file:
        expected = list(csv.reader(expected_file))
    with actual_path.open(newline="", encoding="utf-8") as actual_file:
        actual = list(csv.reader(actual_file))
    if expected == actual:
        return []
    if len(expected) != len(actual):
        return [f"CSV row count differs: expected {len(expected)}, got {len(actual)}"]
    for row_number, (expected_row, actual_row) in enumerate(zip(expected, actual), start=1):
        if len(expected_row) != len(actual_row):
            return [f"CSV column count differs on row {row_number}"]
        for column_number, (expected_cell, actual_cell) in enumerate(zip(expected_row, actual_row), start=1):
            if expected_cell != actual_cell:
                return [
                    f"CSV cell differs at row {row_number}, column {column_number}: "
                    f"expected {expected_cell!r}, got {actual_cell!r}"
                ]
    return ["CSV differs for an unknown reason"]


def _first_json_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"JSON type differs at {path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        if list(expected) != list(actual):
            return f"JSON keys/order differ at {path}: expected {list(expected)!r}, got {list(actual)!r}"
        for key in expected:
            difference = _first_json_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"JSON list length differs at {path}: expected {len(expected)}, got {len(actual)}"
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            difference = _first_json_difference(expected_value, actual_value, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"JSON value differs at {path}: expected {expected!r}, got {actual!r}"
    return None


def main() -> int:
    """Run the current implementation and return zero only for an exact match."""

    missing = [path for path in (GOLDEN_CSV, GOLDEN_SUMMARY) if not path.exists()]
    if missing:
        print("SKIP: triangulation golden files are missing:")
        for path in missing:
            print(f"  {path}")
        print("Create them by running the exact baseline command in this file's module docstring before refactoring.")
        return 0

    temp_dir = ROOT / "work" / f"triangulation-golden-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        actual_csv = temp_dir / "tri_golden.csv"
        actual_summary = temp_dir / "tri_golden_summary.json"
        command = [
            sys.executable,
            "triangulation/points_to_3d.py",
            *COMMON_ARGS,
            "--out-csv", str(actual_csv),
            "--out-summary", str(actual_summary),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            print(f"FAIL: triangulation command exited with code {completed.returncode}")
            return 1

        failures = _compare_csv(GOLDEN_CSV, actual_csv)
        expected_json = json.loads(GOLDEN_SUMMARY.read_text(encoding="utf-8"))
        actual_json = json.loads(actual_summary.read_text(encoding="utf-8"))
        json_difference = _first_json_difference(expected_json, actual_json)
        if json_difference:
            failures.append(json_difference)
        if failures:
            print("FAIL: golden triangulation outputs differ")
            for failure in failures:
                print(f"  {failure}")
            return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("PASS: CSV cells and JSON keys/values exactly match the pre-refactor golden outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
