"""Unit-style checks for stereo offset sweep trim semantics (no video I/O)."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "calibration" / "stereo_checker_debug.py"
SPEC = importlib.util.spec_from_file_location("stereo_checker_debug", MODULE_PATH)
stereo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stereo)


def test_pairing_shift_sign_convention():
    # The same board pose is source frame 4 later in LEFT than in RIGHT.
    left_detected = [10, 20, 30]
    right_detected = [6, 16, 26]

    assert stereo.build_paired_indices_from_stats(0, 0, left_detected, right_detected) == []
    trim_left, trim_right = stereo.shifted_trims_for_stereo_offset(0, 0, 4)
    assert (trim_left, trim_right) == (4, 0)
    assert stereo.build_paired_indices_from_stats(
        trim_left, trim_right, left_detected, right_detected
    ) == [(10, 6), (20, 16), (30, 26)]

    wrong_left, wrong_right = stereo.shifted_trims_for_stereo_offset(0, 0, -4)
    assert stereo.build_paired_indices_from_stats(
        wrong_left, wrong_right, left_detected, right_detected
    ) == []


def test_sync_sweep_update_round_trip():
    left_detected = [10, 20, 30]
    right_detected = [6, 16, 26]
    temp_root = ROOT / "tools" / "tests" / "_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    sync_path = temp_root / "stereo_offset_sweep_sync.json"
    try:
        sync_path.unlink(missing_ok=True)
        sync_path.write_text(json.dumps({"mode": "audio", "trim_left": 0, "trim_right": 0, "keep": "yes"}))

        stereo.update_sync_after_stereo_sweep(
            str(sync_path), 4,
            {"-4": 9.5, "0": 6.0, "4": 0.4},
            {"-4": 3, "0": 3, "4": 3},
        )
        assert stereo.load_sync(str(sync_path)) == (4, 0)
        updated = json.loads(sync_path.read_text())
        assert updated["keep"] == "yes"
        assert updated["stereo_sweep"]["applied_offset"] == 4
        assert updated["stereo_sweep"]["rms_by_offset"]["4"] == 0.4
        assert stereo.build_paired_indices_from_stats(
            *stereo.load_sync(str(sync_path)), left_detected, right_detected
        ) == [(10, 6), (20, 16), (30, 26)]
    finally:
        sync_path.unlink(missing_ok=True)


def main():
    test_pairing_shift_sign_convention()
    test_sync_sweep_update_round_trip()
    print("test_stereo_offset_sweep: PASS")


if __name__ == "__main__":
    main()
