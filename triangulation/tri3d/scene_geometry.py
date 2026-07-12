"""Reference-frame, displacement, grid, and camera geometry for scene views."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d


def _reference_positions(valid_rows):
    refs = {}
    for r in sorted(valid_rows, key=lambda row: (int(row["frame_L"]), int(row["obj_id"]))):
        obj_id = int(r["obj_id"])
        if obj_id not in refs:
            refs[obj_id] = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])], dtype=np.float64)
    return refs


def _attach_displacements(valid_rows):
    refs = _reference_positions(valid_rows)
    by_obj = {}
    disp_max = 0.0
    for r in valid_rows:
        obj_id = int(r["obj_id"])
        xyz = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])], dtype=np.float64)
        disp = float(np.linalg.norm(xyz - refs[obj_id]))
        r["_disp_m"] = disp
        by_obj.setdefault(obj_id, []).append(disp)
        disp_max = max(disp_max, disp)

    motion = {obj_id: float(np.percentile(vals, 95)) for obj_id, vals in by_obj.items()}
    if not motion:
        return refs, set(), max(disp_max, 1e-9)
    cutoff = float(np.percentile(list(motion.values()), 15))
    fixed_ids = {obj_id for obj_id, value in motion.items() if value <= cutoff}
    return refs, fixed_ids, max(disp_max, 1e-9)


def _build_net_reference_frame(refs, fixed_ids):
    ref_items = sorted(refs.items())
    ref_xyz = np.array([xyz for _obj_id, xyz in ref_items], dtype=np.float64)
    origin = ref_xyz.mean(axis=0)
    centered = ref_xyz - origin
    if len(ref_xyz) >= 3:
        _vals, vecs = np.linalg.eigh(np.cov(centered.T))
        normal = vecs[:, 0]
        plane_y = vecs[:, 2]
        plane_z = vecs[:, 1]
    else:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        plane_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        plane_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    fixed_xyz = np.array(
        [refs[obj_id] for obj_id in fixed_ids if obj_id in refs],
        dtype=np.float64,
    )
    if fixed_xyz.size == 0:
        fixed_xyz = ref_xyz

    ref_local = np.column_stack([
        centered @ normal,
        centered @ plane_y,
        centered @ plane_z,
    ])
    fixed_centered = fixed_xyz - origin
    fixed_local = np.column_stack([
        fixed_centered @ normal,
        fixed_centered @ plane_y,
        fixed_centered @ plane_z,
    ])

    # Put the low-motion fixed end toward small y and high z in the local net plane.
    if float(np.mean(fixed_local[:, 1])) > float(np.mean(ref_local[:, 1])):
        plane_y = -plane_y
        ref_local[:, 1] *= -1.0
        fixed_local[:, 1] *= -1.0
    if float(np.mean(fixed_local[:, 2])) < float(np.mean(ref_local[:, 2])):
        plane_z = -plane_z
        ref_local[:, 2] *= -1.0
        fixed_local[:, 2] *= -1.0

    basis = np.vstack([normal, plane_y, plane_z])
    return origin, basis, ref_local, fixed_local


def _infer_grid_shape(refs, ref_local, grid_cols=0, grid_rows=0):
    obj_ids = sorted(int(obj_id) for obj_id in refs)
    if not obj_ids:
        return 0, 0
    count = len(obj_ids)
    if grid_cols and grid_rows:
        return int(grid_cols), int(grid_rows)
    if grid_cols:
        return int(grid_cols), max(1, count // int(grid_cols))
    if grid_rows:
        return max(1, count // int(grid_rows)), int(grid_rows)

    factor_pairs = [
        (cols, count // cols)
        for cols in range(1, count + 1)
        if count % cols == 0
    ]
    if not factor_pairs:
        return count, 1

    yz_span = np.ptp(np.asarray(ref_local[:, 1:3], dtype=np.float64), axis=0)
    ref_ratio = max(float(yz_span[0]), float(yz_span[1])) / max(min(float(yz_span[0]), float(yz_span[1])), 1e-9)
    candidates = [
        pair for pair in factor_pairs
        if pair[0] > 1 and pair[1] > 1
    ] or factor_pairs

    def score(pair):
        cols, rows = pair
        grid_ratio = max(cols - 1, rows - 1) / max(min(cols - 1, rows - 1), 1)
        orientation_penalty = 0
        if yz_span[0] >= yz_span[1] and cols < rows:
            orientation_penalty = 0.25
        elif yz_span[1] > yz_span[0] and rows < cols:
            orientation_penalty = 0.25
        skinny_penalty = abs(cols - rows) / max(count, 1)
        return abs(grid_ratio - ref_ratio) + orientation_penalty + skinny_penalty

    return min(candidates, key=score)


def _build_reference_edges(refs, ref_local, grid_cols=0, grid_rows=0):
    """Infer net edges from the two dominant lattice directions."""
    ref_items = sorted(refs.items())
    obj_ids = [int(obj_id) for obj_id, _xyz in ref_items]
    if len(obj_ids) < 2:
        return [], (0, 0)

    cols, rows = _infer_grid_shape(refs, ref_local, grid_cols=grid_cols, grid_rows=grid_rows)
    if cols <= 0 or rows <= 0:
        return [], (0, 0)

    yz = np.asarray(ref_local[:, 1:3], dtype=np.float64)
    pairs = []
    nearest_by_point = []
    for i in range(len(obj_ids)):
        best = None
        for j in range(len(obj_ids)):
            if i == j:
                continue
            dist = float(np.linalg.norm(yz[j] - yz[i]))
            if best is None or dist < best:
                best = dist
        if best is not None and best > 1e-9:
            nearest_by_point.append(best)

    if not nearest_by_point:
        return [], (cols, rows)
    nearest = float(np.median(nearest_by_point))

    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            vec = yz[j] - yz[i]
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-9 or dist > nearest * 1.55:
                continue
            angle = float(np.mod(np.arctan2(vec[1], vec[0]), np.pi))
            pairs.append((i, j, dist, angle))

    if not pairs:
        return [], (cols, rows)

    def angle_delta(a, b):
        return abs(float((a - b + np.pi / 2.0) % np.pi - np.pi / 2.0))

    bins = np.linspace(0.0, np.pi, 37)
    scores = np.zeros(len(bins) - 1, dtype=np.float64)
    for _i, _j, dist, angle in pairs:
        bin_idx = min(int(angle / np.pi * len(scores)), len(scores) - 1)
        scores[bin_idx] += 1.0 / max(dist, 1e-9)

    first_idx = int(np.argmax(scores))
    first_angle = float(0.5 * (bins[first_idx] + bins[first_idx + 1]))
    second_idx = None
    second_score = -1.0
    for idx, score_value in enumerate(scores):
        angle = float(0.5 * (bins[idx] + bins[idx + 1]))
        sep = angle_delta(angle, first_angle)
        if sep < np.deg2rad(35.0) or sep > np.deg2rad(145.0):
            continue
        if score_value > second_score:
            second_idx = idx
            second_score = float(score_value)
    if second_idx is None:
        return [], (cols, rows)
    second_angle = float(0.5 * (bins[second_idx] + bins[second_idx + 1]))

    def refine_angle(seed_angle):
        nearby = [
            (angle, 1.0 / max(dist, 1e-9))
            for _i, _j, dist, angle in pairs
            if angle_delta(angle, seed_angle) <= np.deg2rad(18.0)
        ]
        if not nearby:
            return seed_angle
        doubled = np.array([2.0 * angle for angle, _weight in nearby], dtype=np.float64)
        weights = np.array([weight for _angle, weight in nearby], dtype=np.float64)
        mean_angle = 0.5 * np.arctan2(
            float(np.sum(weights * np.sin(doubled))),
            float(np.sum(weights * np.cos(doubled))),
        )
        return float(np.mod(mean_angle, np.pi))

    lattice_angles = [refine_angle(first_angle), refine_angle(second_angle)]
    edges = set()
    cone = np.deg2rad(18.0)
    for lattice_angle in lattice_angles:
        dir_pairs = [
            (i, j, dist)
            for i, j, dist, angle in pairs
            if angle_delta(angle, lattice_angle) <= cone
        ]
        if not dir_pairs:
            continue
        step = float(np.median([dist for _i, _j, dist in dir_pairs]))
        max_step = step * 1.45
        for i, j, dist in dir_pairs:
            if dist <= max_step:
                edges.add(tuple(sorted((obj_ids[i], obj_ids[j]))))

    return sorted(edges), (cols, rows)


def _to_local_xyz(xyz, origin, basis):
    return (np.asarray(xyz, dtype=np.float64) - origin) @ basis.T


def _choose_iso_azim(ref_local, fixed_local):
    """Pick an azimuth that projects the low-motion fixed end to top-left."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    # Avoid exactly edge-on camera angles. The net should still look like a
    # mostly-flat plane, just rotated enough that x deformation comes forward.
    candidates = [-60, -45, 45, 60]
    x_lim = (float(ref_local[:, 0].min()), float(ref_local[:, 0].max()))
    y_lim = (float(ref_local[:, 1].min()), float(ref_local[:, 1].max()))
    z_lim = (float(ref_local[:, 2].min()), float(ref_local[:, 2].max()))
    best = (-1e18, -120)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
    all_center = ref_local.mean(axis=0)
    fixed_center = fixed_local.mean(axis=0)
    for azim in candidates:
        ax.view_init(elev=30, azim=azim)
        fig.canvas.draw()
        fx, fy, _ = proj3d.proj_transform(*fixed_center, ax.get_proj())
        axc, ayc, _ = proj3d.proj_transform(*all_center, ax.get_proj())
        # Match the rendered image: fixed end should project left and high.
        score = (fx - axc) + (fy - ayc)
        if score > best[0]:
            best = (float(score), int(azim))
    plt.close(fig)
    return best[1]
