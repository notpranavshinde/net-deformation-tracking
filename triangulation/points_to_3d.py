r"""Compatibility entry point for stereo track triangulation.

The implementation lives in :mod:`triangulation.tri3d`; this path remains
stable because pipeline scripts and existing lab workflows invoke it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triangulation.tri3d.cli import main  # noqa: E402
from triangulation.tri3d.scene_geometry import (  # noqa: E402, F401
    _attach_displacements,
    _build_net_reference_frame,
    _build_reference_edges,
    _choose_iso_azim,
    _to_local_xyz,
)


if __name__ == "__main__":
    main()
