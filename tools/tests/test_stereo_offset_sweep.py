"""Unit-style checks for stereo offset sweep trim semantics (no video I/O)."""

import importlib.util
import json
from pathlib import Path

import numpy as np


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
        sync_path.write_text(json.dumps({
            "mode": "audio", "trim_left": 0, "trim_right": 0, "keep": "yes",
            "subframe_offset": 0.35,
            "subframe_offset_sign_convention": "stale",
            "subframe_sweep": {"rms_by_tau": {"0.35": 0.4}},
        }))

        stereo.update_sync_after_stereo_sweep(
            str(sync_path), 4,
            {"-4": 9.5, "0": 6.0, "4": 0.4},
            {"-4": 3, "0": 3, "4": 3},
        )
        assert stereo.load_sync(str(sync_path)) == (4, 0)
        updated = json.loads(sync_path.read_text())
        assert updated["keep"] == "yes"
        assert "subframe_offset" not in updated
        assert "subframe_sweep" not in updated
        assert "subframe_offset_sign_convention" not in updated
        assert updated["stereo_sweep"]["applied_offset"] == 4
        assert updated["stereo_sweep"]["rms_by_offset"]["4"] == 0.4
        assert stereo.build_paired_indices_from_stats(
            *stereo.load_sync(str(sync_path)), left_detected, right_detected
        ) == [(10, 6), (20, 16), (30, 26)]

        stereo.update_sync_after_subframe_sweep(
            str(sync_path), 0.35,
            {"0.30": 0.5, "0.35": 0.4, "0.40": 0.45},
            {"0.30": 8, "0.35": 8, "0.40": 7},
        )
        updated = json.loads(sync_path.read_text())
        assert updated["trim_left"] == 4 and updated["trim_right"] == 0
        assert updated["subframe_offset"] == 0.35
        assert updated["subframe_sweep"]["rms_by_tau"]["0.35"] == 0.4
        assert "samples LEFT corners at f+tau" in updated["subframe_offset_sign_convention"]
    finally:
        sync_path.unlink(missing_ok=True)


def sample(frame_idx, x):
    corners = np.array([[x, 0.0], [x + 10.0, 0.0], [x, 10.0], [x + 10.0, 10.0]], np.float32)
    obj = np.zeros((4, 3), np.float32)
    return stereo.Sample(frame_idx, corners, obj)


def test_subframe_interpolation_sign_and_planted_tau_recovery():
    cache = {idx: sample(idx, float(idx)) for idx in range(0, 10)}
    assert np.allclose(stereo.interpolate_corner_sample(cache, 4, 0.25).img, sample(4, 4.25).img)
    assert np.allclose(stereo.interpolate_corner_sample(cache, 4, -0.25).img, sample(4, 3.75).img)

    planted_tau = 0.35
    pairs = [(idx, idx) for idx in range(2, 8)]
    right_cache = {idx: sample(idx, idx + planted_tau) for idx in range(2, 8)}
    rms_by_tau = {}
    for tau in stereo.subframe_tau_values(-0.9, 0.9, 0.05):
        _base, interpolated, right = stereo.select_interpolated_pairs(
            pairs, cache, right_cache, tau, len(pairs)
        )
        rms_by_tau[tau] = float(np.mean([
            np.linalg.norm(left.img - right_sample.img, axis=1).mean()
            for left, right_sample in zip(interpolated, right)
        ]))
    recovered = min(rms_by_tau, key=rms_by_tau.get)
    assert np.isclose(recovered, planted_tau)
    assert rms_by_tau[recovered] < 1e-6


def test_subframe_interpolation_rejects_corner_order_flip():
    base = sample(4, 4.0)
    flipped = stereo.Sample(5, np.flip(base.img, axis=0).copy(), np.array(base.obj, copy=True))
    translated = stereo.Sample(
        5,
        (base.img + np.array([0.1, 0.0], np.float32)).copy(),
        np.array(base.obj, copy=True),
    )

    assert stereo.interpolate_corner_sample({4: base, 5: flipped}, 4, 0.5) is None
    interpolated = stereo.interpolate_corner_sample({4: base, 5: translated}, 4, 0.5)
    assert interpolated is not None
    assert np.allclose(
        interpolated.img, base.img + np.array([0.05, 0.0], np.float32)
    )


def test_still_pair_scoring_and_selection_order():
    positions = {
        0: 0.0, 1: 1.0, 2: 2.0,
        9: 5.0, 10: 5.1, 11: 5.2,
    }
    left_cache = {idx: sample(idx, x) for idx, x in positions.items()}
    right_cache = {idx: sample(idx, x + 3.0) for idx, x in positions.items()}
    moving = stereo.stereo_pair_motion_score(left_cache, right_cache, (1, 1))
    still = stereo.stereo_pair_motion_score(left_cache, right_cache, (10, 10))
    assert still < moving
    base, _interpolated, _right = stereo.select_interpolated_pairs(
        [(1, 1), (10, 10)], left_cache, right_cache, 0.0, 1, prefer_still=True
    )
    assert [item.frame_idx for item in base] == [10]


def main():
    test_pairing_shift_sign_convention()
    test_sync_sweep_update_round_trip()
    test_subframe_interpolation_sign_and_planted_tau_recovery()
    test_subframe_interpolation_rejects_corner_order_flip()
    test_still_pair_scoring_and_selection_order()
    print("test_stereo_offset_sweep: PASS")


if __name__ == "__main__":
    main()
