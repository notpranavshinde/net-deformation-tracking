"""Read, validate, filter, and serialize 2D/3D tracking table data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TRACK_COLUMNS = ("frame", "obj_id", "u", "v", "quality", "valid")
VIEWER_COLUMNS = ("frame_L", "obj_id", "uL", "vL", "X", "Y", "Z", "valid_3d")


def load_track_csv(path: str | Path, quality_min: float) -> pd.DataFrame:
    """Load one camera's observations and apply the legacy validity filters."""

    frame = pd.read_csv(path)
    for column in TRACK_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing column '{column}' in {frame}")
    frame["frame"] = frame["frame"].astype(int)
    frame["obj_id"] = frame["obj_id"].astype(int)
    frame["valid"] = frame["valid"].astype(int)
    frame["quality"] = frame["quality"].astype(float)
    return frame[
        (frame["valid"] == 1) & (frame["quality"] >= quality_min)
    ].copy()


def load_existing_triangulation_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a triangulation CSV for visualization-only runs."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"--viz-only requested but CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = [column for column in VIEWER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required visualization columns: {missing}")
    rows = frame.to_dict(orient="records")
    print(f"[VIS] Loaded existing triangulation CSV: {csv_path} rows={len(rows)}")
    return rows


def write_triangulation_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write triangulated observations using pandas' established formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output

