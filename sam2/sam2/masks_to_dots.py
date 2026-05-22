"""
Convert SAM2 per-frame binary masks (per object) into stable 2D points with quality metrics.

Input:
  out/masks/          per-object masks (frame-by-frame)
  out/prompts/        frame0_points.json (reference points for order)
  frames_root/        (optional) original frames for validation

Output:
  tracks_2d.csv       u, v, method, area_px, core_radius_px, solidity, quality, valid per frame/obj
  summary.json        processing stats
  debug_overlay/      (optional) visualization frames
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import ndimage
import skimage.morphology as morph


def resolve_coordinate_offset(masks_root: Path, crop_meta_path: Path = None, restore_original_coords: bool = True):
    """Resolve (x,y) offset to convert cropped-frame points back to original frame coordinates."""
    if not restore_original_coords:
        return 0, 0, None

    candidate = Path(crop_meta_path) if crop_meta_path else (Path(masks_root).parent / "prompts" / "crop_roi.json")
    if not candidate.exists():
        return 0, 0, None

    try:
        payload = json.loads(candidate.read_text())
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        return x, y, candidate
    except Exception as e:
        print(f"[WARN] Failed reading crop metadata from {candidate}: {e}")
        return 0, 0, None


def _process_frame_worker(args):
    """
    Worker function for multiprocessing.
    Takes (frame_idx, converter_config) and returns (frame_rows, frame_stats).
    """
    frame_idx, converter_config = args
    converter = MaskToDotConverter(**converter_config)
    return converter.process_frame(frame_idx)


class MaskToDotConverter:
    def __init__(
        self,
        masks_root: Path,
        out_csv: Path,
        out_summary: Path,
        frames_root: Path = None,
        debug_overlay_dir: Path = None,
        overlay_dir: Path = None,
        visualization_results: bool = True,
        visualization_output: Path = None,
        num_workers: int = None,
        use_skeleton: bool = False,
        area_min: int = 10,
        area_max: int = 1_000_000,
        core_radius_min: float = 1.0,
        solidity_min: float = 0.5,
        junction_min_pixels: int = 3,
        junction_radius_min: float = 2.0,
        erode_kernel_size: int = 3,
        coord_offset_x: int = 0,
        coord_offset_y: int = 0,
    ):
        """
        Args:
            masks_root: Path to masks/ directory (contains 00/, 01/, ... subdirs)
            out_csv: Path to output tracks_2d.csv
            out_summary: Path to output summary.json
            frames_root: (optional) Path to frames/ directory for validation
            debug_overlay_dir: (optional) Path to save debug overlay frames
            area_min, area_max: bounds for valid area
            core_radius_min: min distance-transform value at peak
            solidity_min: min area/convex_hull_area ratio
            junction_min_pixels: min number of junction pixels to use junction method
            junction_radius_min: min DT value at best junction
            erode_kernel_size: kernel size for eroded-centroid fallback
        """
        self.masks_root = Path(masks_root)
        self.out_csv = Path(out_csv)
        self.out_summary = Path(out_summary)
        self.frames_root = Path(frames_root) if frames_root else None
        self.debug_overlay_dir = Path(debug_overlay_dir) if debug_overlay_dir else None
        self.overlay_dir = Path(overlay_dir) if overlay_dir else None
        self.visualization_results = visualization_results
        self.visualization_output = Path(visualization_output) if visualization_output else Path("tracks_visualization.mp4")
        self.num_workers = num_workers if num_workers else 8
        self.use_skeleton = use_skeleton

        self.area_min = area_min
        self.area_max = area_max
        self.core_radius_min = core_radius_min
        self.solidity_min = solidity_min
        self.junction_min_pixels = junction_min_pixels
        self.junction_radius_min = junction_radius_min
        self.erode_kernel_size = erode_kernel_size
        self.coord_offset_x = int(coord_offset_x)
        self.coord_offset_y = int(coord_offset_y)

        # Stats
        self.stats = {
            "frames_processed": 0,
            "missing_masks": 0,
            "invalid_frames": 0,
            "per_object_dropout": defaultdict(lambda: {"total": 0, "invalid": 0}),
        }

        # Results
        self.rows = []  # CSV rows

        if self.debug_overlay_dir:
            self.debug_overlay_dir.mkdir(parents=True, exist_ok=True)
        if self.visualization_output:
            self.visualization_output.parent.mkdir(parents=True, exist_ok=True)

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask from file, return bool array or None if invalid."""
        if not mask_path.exists():
            return None
        try:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None or m.size == 0:
                return None
            return (m > 127).astype(np.uint8).astype(bool)
        except Exception as e:
            print(f"[WARN] Failed to load {mask_path}: {e}")
            return None

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Cleanup: keep largest connected component, optional morphological close.
        """
        if not np.any(mask):
            return mask

        # Label connected components
        labeled, n_labels = ndimage.label(mask)
        if n_labels == 0:
            return np.zeros_like(mask, dtype=bool)

        # Keep largest foreground component (exclude label 0 background)
        sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
        largest_label = int(np.argmax(sizes)) + 1
        cleaned = (labeled == largest_label).astype(bool)

        return cleaned

    def compute_centroid(self, mask: np.ndarray) -> tuple:
        """Compute centroid (v, u) in pixel coordinates. Returns None if mask is empty."""
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            return None
        u = np.mean(x_coords)
        v = np.mean(y_coords)
        return u, v

    def compute_area_and_solidity(self, mask: np.ndarray) -> tuple:
        """Compute area and solidity. Returns (area_px, solidity) or (0, 0)."""
        area = np.sum(mask)
        if area == 0:
            return 0, 0.0

        # Convex hull
        y_coords, x_coords = np.where(mask)
        if len(x_coords) < 3:
            return area, 1.0  # degenerate, assume solid

        pts = np.column_stack([x_coords, y_coords]).astype(np.float32)
        hull = cv2.convexHull(pts)
        if hull is None or len(hull) < 3:
            return area, 1.0

        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        return area, solidity

    def distance_transform_peak(self, mask: np.ndarray) -> tuple:
        """
        Compute distance transform and return (u, v, core_radius).
        Returns (None, None, 0) if mask is empty.
        """
        if not np.any(mask):
            return None, None, 0

        try:
            mask_u8 = (mask.astype(np.uint8) * 255)
            dt = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
            peak_idx = np.unravel_index(np.argmax(dt), dt.shape)
            v, u = peak_idx
            core_radius = dt[v, u]
            return float(u), float(v), float(core_radius)
        except cv2.error as e:
            if "Insufficient memory" in str(e):
                return None, None, 0
            raise

    def _effective_worker_count(self, frame_shape: tuple, requested_workers: int) -> int:
        """
        Cap workers for very large masks to avoid OpenCV OutOfMemory in parallel DT.
        """
        h, w = frame_shape
        pixels = h * w

        # Heuristic caps by resolution
        if pixels >= 8_000_000:      # ~4K
            return min(requested_workers, 8)
        if pixels >= 4_000_000:      # ~1440p
            return min(requested_workers, 12)
        return requested_workers

    def find_skeleton_junctions(self, mask: np.ndarray) -> tuple:
        """
        Skeletonize mask and find junction pixels (degree ≥ 3).
        Return (skeleton, junction_pixels_list) where junction_pixels = [(u, v), ...]
        Returns (None, []) if no skeleton.
        """
        if not np.any(mask):
            return None, []

        # Skeletonize
        try:
            import time
            start = time.time()
            skeleton = morph.skeletonize(mask).astype(np.uint8)
            elapsed = time.time() - start
            if elapsed > 0.5:
                print(f"[WARN] Skeletonize took {elapsed:.2f}s")
        except Exception as e:
            print(f"[WARN] Skeletonize failed: {e}")
            return None, []

        if not np.any(skeleton):
            return None, []

        # Find junctions (degree >= 3)
        junctions = []
        vy, vx = np.where(skeleton > 0)
        for v, u in zip(vy, vx):
            # 8-connectivity neighbors
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = v + dy, u + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx] > 0:
                            neighbors += 1
            if neighbors >= 3:
                junctions.append((float(u), float(v)))

        return skeleton, junctions

    def pick_junction_point(
        self, junctions: list, dt: np.ndarray, skeleton: np.ndarray
    ) -> tuple:
        """
        Pick best junction: max DT value or nearest to skeleton center-of-mass.
        Return (u, v, dt_value) or (None, None, 0) if no suitable junction.
        """
        if not junctions:
            return None, None, 0

        # Compute skeleton center-of-mass
        vy, vx = np.where(skeleton > 0)
        skel_center_u = np.mean(vx) if len(vx) > 0 else 0
        skel_center_v = np.mean(vy) if len(vy) > 0 else 0

        # Pick junction with max DT
        best_u, best_v = junctions[0]
        best_dt = dt[int(round(best_v)), int(round(best_u))]
        for u, v in junctions[1:]:
            iv, iu = int(round(v)), int(round(u))
            if 0 <= iv < dt.shape[0] and 0 <= iu < dt.shape[1]:
                dt_val = dt[iv, iu]
                if dt_val > best_dt:
                    best_dt = dt_val
                    best_u, best_v = u, v

        return best_u, best_v, best_dt

    def eroded_centroid(self, mask: np.ndarray, kernel_size: int = 3) -> tuple:
        """Centroid of eroded mask. Fallback when main methods fail."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return self.compute_centroid(eroded.astype(bool))

    def compute_quality(
        self,
        area: int,
        core_radius: float,
        solidity: float,
        bbox_h: int,
        bbox_w: int,
    ) -> float:
        """
        Compute quality score (0–1).
        Penalize for suspicious features.
        """
        quality = 1.0

        # Area bounds
        if area < self.area_min or area > self.area_max:
            quality *= 0.5

        # Core radius
        if core_radius < self.core_radius_min:
            quality *= 0.8

        # Solidity
        if solidity < self.solidity_min:
            quality *= 0.7

        # Aspect ratio (very extreme = suspicious)
        if bbox_h > 0 and bbox_w > 0:
            aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)
            if aspect > 5:
                quality *= 0.6

        return np.clip(quality, 0.0, 1.0)

    def process_frame(
        self, frame_idx: int, frame_bgr: np.ndarray = None
    ) -> dict:
        """
        Process all objects in frame_idx.
        Returns (rows, stats_update) for this frame.
        """
        frame_rows = []
        frame_stats = {
            "frames_processed": 1,
            "missing_masks": 0,
            "invalid_frames": 0,
            "per_object_dropout": defaultdict(lambda: {"total": 0, "invalid": 0}),
        }

        # Find all object directories
        obj_dirs = sorted(
            [d for d in self.masks_root.iterdir() if d.is_dir()],
            key=lambda d: int(d.name)
        )

        for obj_dir in obj_dirs:
            obj_id = int(obj_dir.name)
            mask_file = obj_dir / f"{frame_idx:06d}.jpg"

            frame_stats["per_object_dropout"][obj_id]["total"] += 1

            # Load and validate mask
            mask = self.load_mask(mask_file)
            if mask is None or not np.any(mask):
                frame_stats["per_object_dropout"][obj_id]["invalid"] += 1
                if mask is None:
                    frame_stats["missing_masks"] += 1
                frame_rows.append({
                    "frame": frame_idx,
                    "obj_id": obj_id,
                    "u": None,
                    "v": None,
                    "u_local": None,
                    "v_local": None,
                    "method": "invalid",
                    "area_px": 0,
                    "core_radius_px": 0.0,
                    "solidity": 0.0,
                    "quality": 0.0,
                    "valid": 0,
                })
                continue

            # Step 3.2: Clean mask
            mask = self.clean_mask(mask)
            if not np.any(mask):
                frame_stats["per_object_dropout"][obj_id]["invalid"] += 1
                frame_rows.append({
                    "frame": frame_idx,
                    "obj_id": obj_id,
                    "u": None,
                    "v": None,
                    "u_local": None,
                    "v_local": None,
                    "method": "invalid_after_cleanup",
                    "area_px": 0,
                    "core_radius_px": 0.0,
                    "solidity": 0.0,
                    "quality": 0.0,
                    "valid": 0,
                })
                continue

            # Step 3.3: Distance-transform peak
            u_dt, v_dt, core_radius = self.distance_transform_peak(mask)

            # Step 3.4: Skeleton junctions (optional)
            u_junc, v_junc, junc_dt = None, None, 0
            junctions = []
            if self.use_skeleton:
                skeleton, junctions = self.find_skeleton_junctions(mask)
                if skeleton is not None and len(junctions) >= self.junction_min_pixels:
                    try:
                        mask_u8 = (mask.astype(np.uint8) * 255)
                        dt = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                        u_junc, v_junc, junc_dt = self.pick_junction_point(junctions, dt, skeleton)
                    except cv2.error as e:
                        if "Insufficient memory" not in str(e):
                            raise

            # Step 3.5: Decision rule
            method = "dt_peak"
            u, v = u_dt, v_dt
            radius = core_radius

            if (
                u_junc is not None
                and len(junctions) >= self.junction_min_pixels
                and junc_dt >= self.junction_radius_min
            ):
                method = "skeleton_junction"
                u, v = u_junc, v_junc
                radius = junc_dt

            # Step 3.6: Fallbacks
            if u is None or v is None:
                fallback = self.eroded_centroid(mask, self.erode_kernel_size)
                if fallback is not None:
                    u, v = fallback
                    method = "fallback_eroded_centroid"
                else:
                    centroid = self.compute_centroid(mask)
                    if centroid is not None:
                        u, v = centroid
                        method = "fallback_centroid"
                    else:
                        frame_stats["per_object_dropout"][obj_id]["invalid"] += 1
                        frame_rows.append({
                            "frame": frame_idx,
                            "obj_id": obj_id,
                            "u": None,
                            "v": None,
                            "u_local": None,
                            "v_local": None,
                            "method": "no_fallback",
                            "area_px": 0,
                            "core_radius_px": 0.0,
                            "solidity": 0.0,
                            "quality": 0.0,
                            "valid": 0,
                        })
                        continue

            # Compute metrics
            area, solidity = self.compute_area_and_solidity(mask)
            vy, vx = np.where(mask)
            if len(vx) > 0 and len(vy) > 0:
                bbox_h = np.max(vy) - np.min(vy) + 1
                bbox_w = np.max(vx) - np.min(vx) + 1
            else:
                bbox_h, bbox_w = 1, 1

            # Step 4: Quality score
            quality = self.compute_quality(area, radius, solidity, bbox_h, bbox_w)
            valid = 1 if quality > 0 and area > 0 else 0
            u_local = float(u)
            v_local = float(v)
            u_out = float(u_local + self.coord_offset_x)
            v_out = float(v_local + self.coord_offset_y)

            frame_rows.append({
                "frame": frame_idx,
                "obj_id": obj_id,
                "u": u_out,
                "v": v_out,
                "u_local": u_local,
                "v_local": v_local,
                "method": method,
                "area_px": int(area),
                "core_radius_px": radius,
                "solidity": float(solidity),
                "quality": float(quality),
                "valid": valid,
            })

        # Convert defaultdict to regular dict for pickling
        frame_stats["per_object_dropout"] = dict(frame_stats["per_object_dropout"])
        return frame_rows, frame_stats

    def process_all_frames(self):
        """Process all frames in masks_root using multiprocessing."""
        # Find all frame indices from first object dir
        obj_dirs = sorted(
            [d for d in self.masks_root.iterdir() if d.is_dir()],
            key=lambda d: int(d.name)
        )
        if not obj_dirs:
            print("[WARN] No object directories found.")
            return

        first_obj = obj_dirs[0]
        mask_files = sorted(
            [f for f in first_obj.glob("*.jpg")],
            key=lambda f: int(f.stem)
        )

        frame_indices = [int(f.stem) for f in mask_files]

        # Estimate safe worker count from first mask resolution
        first_mask_img = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        effective_workers = self.num_workers
        if first_mask_img is not None and first_mask_img.size > 0:
            effective_workers = self._effective_worker_count(first_mask_img.shape[:2], self.num_workers)
            if effective_workers < self.num_workers:
                print(
                    f"[WARN] Capping workers from {self.num_workers} to {effective_workers} "
                    f"for mask size {first_mask_img.shape[1]}x{first_mask_img.shape[0]} to prevent OOM."
                )

        print(f"[INFO] Found {len(frame_indices)} frames to process.")
        print(f"[INFO] Using {effective_workers} workers for parallel processing.")

        # Create config dict for worker processes
        converter_config = {
            "masks_root": self.masks_root,
            "out_csv": self.out_csv,
            "out_summary": self.out_summary,
            "frames_root": self.frames_root,
            "debug_overlay_dir": self.debug_overlay_dir,
            "overlay_dir": self.overlay_dir,
            "visualization_results": self.visualization_results,
            "visualization_output": self.visualization_output,
            "num_workers": 1,  # Workers don't spawn more workers
            "use_skeleton": self.use_skeleton,
            "area_min": self.area_min,
            "area_max": self.area_max,
            "core_radius_min": self.core_radius_min,
            "solidity_min": self.solidity_min,
            "junction_min_pixels": self.junction_min_pixels,
            "junction_radius_min": self.junction_radius_min,
            "erode_kernel_size": self.erode_kernel_size,
            "coord_offset_x": self.coord_offset_x,
            "coord_offset_y": self.coord_offset_y,
        }

        # Process frames in parallel
        with Pool(effective_workers, maxtasksperchild=25) as pool:
            worker_args = [(frame_idx, converter_config) for frame_idx in frame_indices]
            results = pool.imap_unordered(_process_frame_worker, worker_args, chunksize=1)
            
            for i, (frame_rows, frame_stats) in enumerate(results):
                self.rows.extend(frame_rows)
                
                # Merge stats
                self.stats["frames_processed"] += frame_stats["frames_processed"]
                self.stats["missing_masks"] += frame_stats["missing_masks"]
                self.stats["invalid_frames"] += frame_stats["invalid_frames"]
                for obj_id, counts in frame_stats["per_object_dropout"].items():
                    if obj_id not in self.stats["per_object_dropout"]:
                        self.stats["per_object_dropout"][obj_id] = {"total": 0, "invalid": 0}
                    self.stats["per_object_dropout"][obj_id]["total"] += counts["total"]
                    self.stats["per_object_dropout"][obj_id]["invalid"] += counts["invalid"]
                
                if (i + 1) % max(1, len(frame_indices) // 10) == 0:
                    print(f"[INFO] Processed {i + 1}/{len(frame_indices)} frames")

    def write_csv(self):
        """Write tracks_2d.csv."""
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w", newline="") as f:
            fieldnames = [
                "frame",
                "obj_id",
                "u",
                "v",
                "u_local",
                "v_local",
                "method",
                "area_px",
                "core_radius_px",
                "solidity",
                "quality",
                "valid",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[OK] Wrote {self.out_csv}")

    def write_summary(self, frame_width: int = None, frame_height: int = None):
        """Write summary.json."""
        self.out_summary.parent.mkdir(parents=True, exist_ok=True)

        # Compute per-object dropout %
        dropout = {}
        for obj_id, counts in self.stats["per_object_dropout"].items():
            total = counts["total"]
            invalid = counts["invalid"]
            dropout_pct = (invalid / max(total, 1)) * 100
            dropout[str(obj_id)] = {
                "total_frames": total,
                "invalid_frames": invalid,
                "dropout_pct": dropout_pct,
            }

        summary = {
            "frames_processed": self.stats["frames_processed"],
            "missing_masks": self.stats["missing_masks"],
            "invalid_frames": self.stats["invalid_frames"],
            "per_object_dropout": dropout,
            "params": {
                "area_min": self.area_min,
                "area_max": self.area_max,
                "core_radius_min": self.core_radius_min,
                "solidity_min": self.solidity_min,
                "junction_min_pixels": self.junction_min_pixels,
                "junction_radius_min": self.junction_radius_min,
                "coord_offset_x": self.coord_offset_x,
                "coord_offset_y": self.coord_offset_y,
            },
        }

        if frame_width is not None and frame_height is not None:
            summary["frame_width"] = frame_width
            summary["frame_height"] = frame_height

        with open(self.out_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Wrote {self.out_summary}")

    def create_visualization_video(self):
        """Create side-by-side video: overlay (left) + tracked points (right)."""
        if not self.visualization_results:
            print("[INFO] Visualization disabled, skipping...")
            return

        if not self.overlay_dir or not self.overlay_dir.exists():
            print("[WARN] overlay_dir not found, skipping visualization.")
            return

        # Load CSV
        if not self.out_csv.exists():
            print(f"[WARN] {self.out_csv} not found, skipping visualization.")
            return

        rows = []
        with open(self.out_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("[WARN] CSV is empty, skipping visualization.")
            return

        # Group by frame
        frames_data = defaultdict(list)
        for row in rows:
            frame_idx = int(row["frame"])
            frames_data[frame_idx].append(row)

        # Get frame dimensions
        first_overlay = sorted(self.overlay_dir.glob("*.png"))[0]
        overlay_img = cv2.imread(str(first_overlay))
        if overlay_img is None:
            print("[WARN] Could not read overlay image, skipping visualization.")
            return

        h, w = overlay_img.shape[:2]
        frame_size = (w * 2, h)  # side-by-side

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(
            str(self.visualization_output),
            fourcc,
            30.0,
            frame_size,
        )

        # Process each frame
        for frame_idx in sorted(frames_data.keys()):
            # Load overlay
            overlay_path = self.overlay_dir / f"{frame_idx:06d}.png"
            overlay = cv2.imread(str(overlay_path))
            if overlay is None:
                print(f"[WARN] Could not read overlay frame {frame_idx}")
                continue

            # Create dark canvas for tracked points
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Draw tracked points
            for row in frames_data[frame_idx]:
                if row["valid"] == "0":
                    continue

                try:
                    u = float(row.get("u_local", row["u"]))
                    v = float(row.get("v_local", row["v"]))
                    obj_id = int(row["obj_id"])
                    quality = float(row["quality"])
                except (ValueError, KeyError):
                    continue

                # Color by object ID
                rng = np.random.RandomState(obj_id)
                color = tuple(int(c) for c in rng.randint(50, 255, size=3))

                # Draw circle
                cv2.circle(
                    canvas,
                    (int(round(u)), int(round(v))),
                    radius=8,
                    color=color,
                    thickness=-1,
                )

                # Draw ID label
                cv2.putText(
                    canvas,
                    str(obj_id),
                    (int(round(u)) + 10, int(round(v)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Combine overlay and canvas side-by-side
            combined = np.hstack([overlay, canvas])

            # Add frame number
            cv2.putText(
                combined,
                f"Frame {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            out_video.write(combined)

        out_video.release()
        print(f"[OK] Wrote visualization video: {self.visualization_output}")

    def run(self):
        """Main processing pipeline."""
        print("[INFO] Starting mask-to-dots conversion...")
        self.process_all_frames()
        self.write_csv()
        
        # Try to get frame size from first frame
        frame_w, frame_h = None, None
        if self.frames_root:
            first_frame_path = self.frames_root / "000000.jpg"
            if first_frame_path.exists():
                img = cv2.imread(str(first_frame_path))
                if img is not None:
                    frame_h, frame_w = img.shape[:2]

        self.write_summary(frame_width=frame_w, frame_height=frame_h)

        # Create visualization if enabled
        if self.visualization_results:
            self.create_visualization_video()

        print("[OK] Conversion complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM2 masks to 2D points with quality metrics."
    )
    parser.add_argument(
        "--masks",
        required=True,
        help="Path to masks/ directory (from SAM2 output)"
    )
    parser.add_argument(
        "--out-csv",
        default="tracks_2d.csv",
        help="Output tracks_2d.csv"
    )
    parser.add_argument(
        "--out-summary",
        default="summary.json",
        help="Output summary.json"
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Path to frames/ directory (optional, for validation)"
    )
    parser.add_argument(
        "--debug-overlay",
        default=None,
        help="Path to save debug overlay frames (optional)"
    )
    parser.add_argument(
        "--area-min",
        type=int,
        default=10,
        help="Minimum mask area (px²)"
    )
    parser.add_argument(
        "--area-max",
        type=int,
        default=1_000_000,
        help="Maximum mask area (px²)"
    )
    parser.add_argument(
        "--core-radius-min",
        type=float,
        default=1.0,
        help="Minimum distance-transform value"
    )
    parser.add_argument(
        "--solidity-min",
        type=float,
        default=0.5,
        help="Minimum solidity (area / convex hull)"
    )
    parser.add_argument(
        "--junction-min-pixels",
        type=int,
        default=3,
        help="Min skeleton junctions to use junction method"
    )
    parser.add_argument(
        "--junction-radius-min",
        type=float,
        default=2.0,
        help="Min DT value at best junction"
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Path to overlay/ directory (from SAM2 output, for visualization)"
    )
    parser.add_argument(
        "--visualization-results",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Generate side-by-side visualization video (true/false)"
    )
    parser.add_argument(
        "--visualization-output",
        default="tracks_visualization.mp4",
        help="Output path for visualization video"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: 8)"
    )
    parser.add_argument(
        "--use-skeleton",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Enable skeleton junction method (slow)"
    )
    parser.add_argument(
        "--restore-original-coords",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Add crop offsets back so output u,v are in original full-frame coordinates",
    )
    parser.add_argument(
        "--crop-meta",
        default=None,
        help="Path to crop ROI metadata JSON (defaults to <masks_parent>/prompts/crop_roi.json)",
    )
    parser.add_argument(
        "--coord-offset-x",
        type=int,
        default=None,
        help="Manual x offset to add to u (overrides crop metadata)",
    )
    parser.add_argument(
        "--coord-offset-y",
        type=int,
        default=None,
        help="Manual y offset to add to v (overrides crop metadata)",
    )

    args = parser.parse_args()

    auto_x, auto_y, auto_meta_path = resolve_coordinate_offset(
        masks_root=Path(args.masks),
        crop_meta_path=Path(args.crop_meta) if args.crop_meta else None,
        restore_original_coords=args.restore_original_coords,
    )
    offset_x = int(args.coord_offset_x) if args.coord_offset_x is not None else int(auto_x)
    offset_y = int(args.coord_offset_y) if args.coord_offset_y is not None else int(auto_y)
    if args.restore_original_coords:
        if auto_meta_path is not None:
            print(f"[INFO] Restoring original coordinates using crop metadata: {auto_meta_path}")
        else:
            print("[INFO] No crop metadata found. Using zero coordinate offset.")
    print(f"[INFO] Coordinate offset applied: x={offset_x}, y={offset_y}")

    converter = MaskToDotConverter(
        masks_root=args.masks,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        frames_root=args.frames,
        debug_overlay_dir=args.debug_overlay,
        overlay_dir=args.overlay,
        visualization_results=args.visualization_results,
        visualization_output=args.visualization_output,
        num_workers=args.num_workers,
        use_skeleton=args.use_skeleton,
        area_min=args.area_min,
        area_max=args.area_max,
        core_radius_min=args.core_radius_min,
        solidity_min=args.solidity_min,
        junction_min_pixels=args.junction_min_pixels,
        junction_radius_min=args.junction_radius_min,
        coord_offset_x=offset_x,
        coord_offset_y=offset_y,
    )

    converter.run()


if __name__ == "__main__":
    main()
