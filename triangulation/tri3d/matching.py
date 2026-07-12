"""Match filtered left/right observations on the shared global frame timeline."""

from __future__ import annotations

import pandas as pd


Observation = tuple[float, float, float, int]
ObservationKey = tuple[int, int]


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
