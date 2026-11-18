"""Match filtered left/right observations on the shared global frame timeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


Observation = tuple[float, float, float, int]
ObservationKey = tuple[int, int]


def shift_track_subframe(track: pd.DataFrame, tau: float) -> pd.DataFrame:
    """One camera's positions at ``frame + tau``, interpolated from the neighbouring tracked frame.

    The cameras run a fraction of a frame apart, so pairing whole frames compares the net at two
    slightly different times. The calibration sync's ``subframe_offset`` says LEFT frame f lines
    up with RIGHT at f + tau; moving each LEFT point that far along its own track puts both
    cameras at the same instant. Rows whose neighbouring frame was not tracked are dropped.
    """
    if abs(float(tau)) < 1e-9 or track.empty:
        return track
    diffs = track.sort_values(["obj_id", "frame"]).groupby("obj_id")["frame"].diff().dropna()
    step = max(1, int(diffs.median())) if len(diffs) else 1
    direction = 1 if tau > 0 else -1
    neighbour = track[["frame", "obj_id", "u", "v", "quality"]].copy()
    neighbour["frame"] = neighbour["frame"] - direction * step  # keyed by the frame it serves
    merged = track.merge(neighbour, on=["frame", "obj_id"], suffixes=("", "_nb"))
    weight = abs(float(tau)) / step
    merged["u"] = (1.0 - weight) * merged["u"] + weight * merged["u_nb"]
    merged["v"] = (1.0 - weight) * merged["v"] + weight * merged["v_nb"]
    merged["quality"] = np.minimum(merged["quality"], merged["quality_nb"])
    return merged[track.columns].copy()


def match_frame_observations(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_drop: int,
    right_drop: int,
) -> tuple[list[ObservationKey], dict[ObservationKey, Observation], dict[ObservationKey, Observation]]:
    """Return sorted common keys and camera-indexed observations."""

    left["gframe"] = left["frame"] + left_drop
    right["gframe"] = right["frame"] + right_drop
    left_keyed = {
        (int(row.gframe), int(row.obj_id)): (
            float(row.u), float(row.v), float(row.quality), int(row.frame)
        )
        for row in left.itertuples(index=False)
    }
    right_keyed = {
        (int(row.gframe), int(row.obj_id)): (
            float(row.u), float(row.v), float(row.quality), int(row.frame)
        )
        for row in right.itertuples(index=False)
    }
    keys = sorted(set(left_keyed) & set(right_keyed))
    return keys, left_keyed, right_keyed
