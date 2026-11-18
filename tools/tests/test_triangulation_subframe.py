import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "triangulation"))
from tri3d.matching import shift_track_subframe  # noqa: E402
from tri3d.syncmap import load_calibration_sync  # noqa: E402


def track(frames, obj_id=0):
    # Knot moving 10 px/frame in u and -4 px/frame in v.
    return pd.DataFrame({"frame": frames, "obj_id": obj_id, "u": [100.0 + 10 * f for f in frames],
                         "v": [50.0 - 4 * f for f in frames], "quality": 1.0, "valid": 1})


def test_shift_moves_along_the_track_both_ways():
    t = track(list(range(10)))
    ahead = shift_track_subframe(t, 0.4).set_index("frame")
    assert list(ahead.index) == list(range(9))  # frame 9 has no frame 10 to move toward
    assert abs(ahead.loc[3, "u"] - 134.0) < 1e-9 and abs(ahead.loc[3, "v"] - (50 - 4 * 3.4)) < 1e-9
    behind = shift_track_subframe(t, -0.5).set_index("frame")
    assert list(behind.index) == list(range(1, 10))
    assert abs(behind.loc[3, "u"] - 125.0) < 1e-9
    assert shift_track_subframe(t, 0.0) is t


def test_shift_drops_rows_next_to_a_gap_and_keeps_knots_apart():
    t = pd.concat([track([0, 1, 2, 5, 6]), track([0, 1, 2], obj_id=7)])
    out = shift_track_subframe(t, 0.5)
    assert sorted(map(tuple, out[["obj_id", "frame"]].to_numpy())) == [(0, 0), (0, 1), (0, 5), (7, 0), (7, 1)]


def test_sync_json_offset_is_read_only_when_set():
    tmp = ROOT / "tools" / "tests" / "_tmp" / "subframe_sync"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        path = tmp / "sync.json"
        path.write_text(json.dumps({"trim_left": 0, "trim_right": 3}))
        assert "subframe_offset" not in load_calibration_sync(str(path))[2]
        path.write_text(json.dumps({"trim_left": 0, "trim_right": 3, "subframe_offset": -0.45}))
        assert load_calibration_sync(str(path))[2]["subframe_offset"] == -0.45
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_shift_moves_along_the_track_both_ways()
    test_shift_drops_rows_next_to_a_gap_and_keeps_knots_apart()
    test_sync_json_offset_is_read_only_when_set()
    print("test_triangulation_subframe: PASS")


if __name__ == "__main__":
    main()
