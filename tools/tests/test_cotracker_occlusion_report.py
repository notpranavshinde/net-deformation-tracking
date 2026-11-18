import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("occlusion_report", ROOT / "experiments" / "cotracker3" / "occlusion_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_tracks(folder, hidden, offset, start):
    """Two knots (0 and 12) moving 2 px/frame; ``hidden`` frames of knot 0 keep a guess 5 px off."""
    rows = []
    for f in range(30):
        for oid in (0, 12):
            u, v = 100.0 + 2 * f, 50.0 + 10 * oid
            vis = not (oid == 0 and f in hidden)
            uc = u + (5.0 if not vis else 1.0)
            rows.append({"frame": f, "obj_id": oid, "u": uc if not vis else u, "v": v, "u_coarse": uc, "v_coarse": v,
                         "visibility": 0.95 if vis else 0.2, "confidence": 0.9, "visible": int(vis),
                         "method": "cotracker+dtpeak" if vis else "cotracker_inferred"})
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "tracks_2d.csv", index=False)
    (folder / "run_meta.json").write_text(json.dumps({"frame_range": [start, start + 30], "refine_radius": 40}))


def test_hidden_stretch_and_3d_split():
    tmp = ROOT / "tools" / "tests" / "_tmp" / "occlusion_report"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        write_tracks(tmp / "tracks" / "left", hidden=range(5, 10), offset=0, start=1000)
        write_tracks(tmp / "tracks" / "right", hidden=(), offset=0, start=1003)
        result = tmp / "result"
        result.mkdir()
        tri = [{"gframe": g, "frame_L": g, "frame_R": g + 3, "obj_id": oid,
                "err_px": 9.0 if (oid == 0 and 5 <= g <= 9) else 1.0, "valid_3d": 1}
               for g in range(27) for oid in (0, 12)]
        pd.DataFrame(tri).to_csv(result / "triangulated_3d.csv", index=False)

        out = report.write_report(result, tmp / "tracks")
        st = pd.read_csv(out / "hidden_stretches.csv")
        assert len(st) == 1
        s = st.iloc[0]
        assert (s.camera, s.obj_id, s.first_frame, s.last_frame, s.n_frames) == ("left", 0, 5, 9, 5)
        assert (s.first_source_frame, s.last_source_frame) == (1005, 1009)
        assert s.reappeared and not s.at_search_limit
        assert np.isclose(s.reappear_gap_px, 3.0)  # guess at 9 was u+5; marker at 10 is 2 px further on
        assert np.isclose(s.model_miss_px, 1.0)
        assert s.err_px_median_hidden == 9.0 and s.err_px_median_nearby_both_visible == 1.0

        summary = json.loads((out / "summary.json").read_text())
        left = summary["cameras"]["left"]
        assert left["stretches"] == 1 and left["hidden_pct_by_row"][0] == round(100 * 5 / 30, 2)
        assert summary["cameras"]["right"]["stretches"] == 0
        assert summary["three_d_by_visibility"]["left_hidden_only"]["n"] == 5
        assert summary["three_d_by_visibility"]["left_hidden_only"]["reproj_err_median_px"] == 9.0
        kf = pd.read_csv(out / "knot_frames.csv")
        assert len(kf) == 54 and kf.visible_L.notna().all() and kf.visible_R.notna().all()
        assert (out / "occlusion_timeline.png").stat().st_size > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_hidden_stretch_and_3d_split()
    print("test_cotracker_occlusion_report: PASS")


if __name__ == "__main__":
    main()
