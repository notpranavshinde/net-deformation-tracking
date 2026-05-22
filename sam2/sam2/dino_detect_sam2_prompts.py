"""
DINO exemplar matching for initial SAM2 prompt candidates.

This is for local, interactive prompt proposal only. It does not run SAM2,
does not track masks, and does not triangulate. It uses a few positive/negative
example clicks to build a dense DINO feature-similarity heatmap, then proposes
local maxima as SAM2 click points.

Paste-ready local workflow from sam2\\sam2:
    python .\\dino_detect_sam2_prompts.py --frame 150 --select-dino-crop --click-examples

After examples are saved, retune/rerun without clicking:
    python .\\dino_detect_sam2_prompts.py --frame 150 --examples .\\work\\dino_prompts\\examples.json

Default backend:
    DINOv2 via the public Meta repo in external\\dinov2. This does not need
    Hugging Face approval.

Optional DINOv3 backend:
    DINOv3 weights are gated by Meta. Once approved, put a token in .env:
        HF_TOKEN=your_token_here
    Then run with --backend huggingface.

Outputs:
    work\\dino_prompts\\left_candidates.csv/json/preview.png/heatmap.png
    work\\dino_prompts\\right_candidates.csv/json/preview.png/heatmap.png
    work\\dino_prompts\\examples.json

Click controls for examples:
    LMB = positive painted-node example
    RMB = negative/background/net example
    mouse wheel / +/- = zoom around cursor
    middle mouse drag = pan
    x   = undo last click
    q   = accept examples for that side
    Esc = cancel

Coordinates:
    x/y are full-frame pixels.
    local_x/local_y are DINO-crop-local pixels.
    DINO crops are search crops only and are saved separately from SAM2 crops.
"""

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from auto_detect_sam2_prompts import (  # reuse the SAM2-consistent crop helpers
    DEFAULT_OUT_ROOT,
    clip_crop,
    crop_to_dict,
    ensure_parent,
    open_image,
    parse_crop_arg,
    read_existing_crop,
    robust_read_frame,
    save_crop_meta,
)


DEFAULT_WORK_DIR = Path("work") / "dino_prompts"

warnings.filterwarnings("ignore", message="xFormers is not available.*")


def save_peak_params(path: Path, args):
    ensure_parent(path)
    payload = {
        "min_score": float(args.min_score),
        "min_distance": float(args.min_distance),
        "max_candidates": int(args.max_candidates),
        "target_count": int(args.target_count),
        "left_target_count": int(args.left_target_count),
        "right_target_count": int(args.right_target_count),
        "feature_max_size": int(args.feature_max_size),
        "negative_weight": float(args.negative_weight),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Saved DINO peak params: {path}")


def load_peak_params(path: Path, args):
    if not path.exists() or args.ignore_saved_params:
        return
    payload = json.loads(path.read_text())
    for key in (
        "min_score",
        "min_distance",
        "max_candidates",
        "target_count",
        "left_target_count",
        "right_target_count",
        "feature_max_size",
        "negative_weight",
    ):
        if key in payload:
            current = getattr(args, key)
            if isinstance(current, int):
                setattr(args, key, int(payload[key]))
            else:
                setattr(args, key, float(payload[key]))
    print(f"[INFO] Loaded saved DINO peak params: {path}")


@dataclass
class Candidate:
    x: float
    y: float
    local_x: float
    local_y: float
    score: float
    area: float = 0.0
    method: str = "dino"


def load_dotenv(path: Path):
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dino_model(args, device):
    if args.backend == "dinov2":
        repo = args.dinov2_repo or os.environ.get("DINOV2_REPO") or (Path(__file__).resolve().parent / "external" / "dinov2")
        repo_path = Path(repo)
        if not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv2 repo path not found: {repo_path}. Clone with: "
                "git clone https://github.com/facebookresearch/dinov2.git external/dinov2"
            )
        args.patch_size = 14
        print(f"[INFO] Loading DINOv2 model {args.model} from {repo_path}")
        model = torch.hub.load(str(repo_path), args.model, source="local")
        model.eval().to(device)
        return model

    if args.backend == "huggingface":
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise RuntimeError("Missing transformers. Install with: python -m pip install \"transformers>=4.56.0\"") from e

        print(f"[INFO] Loading DINOv3 Hugging Face model {args.hf_model}")
        try:
            token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            model = AutoModel.from_pretrained(args.hf_model, token=token)
        except Exception as e:
            raise RuntimeError(
                "Could not load the Hugging Face DINOv3 model. Meta's DINOv3 checkpoints are gated.\n"
                "Do this once:\n"
                "  1) Open https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
                "  2) Accept the model access terms while logged into Hugging Face\n"
                "  3) Run: huggingface-cli login\n"
                "     or pass --hf-token YOUR_TOKEN\n"
                f"Original error: {e}"
            ) from e

        args.patch_size = int(getattr(model.config, "patch_size", args.patch_size))
        model.eval().to(device)
        return model

    repo = args.dinov3_repo or os.environ.get("DINOV3_REPO") or (Path(__file__).resolve().parent / "external" / "dinov3")
    repo_path = Path(repo)
    if not repo_path.exists():
        raise FileNotFoundError(
            f"DINOv3 repo path not found: {repo_path}. Clone with: "
            "git clone https://github.com/facebookresearch/dinov3.git external/dinov3"
        )

    kwargs = {"source": "local"}
    if args.weights:
        kwargs["weights"] = str(args.weights)

    print(f"[INFO] Loading DINOv3 torchhub model {args.model} from {repo_path}")
    model = torch.hub.load(str(repo_path), args.model, **kwargs)
    model.eval().to(device)
    return model


def select_dino_crop_on_frame(video_path: Path, frame_idx: int, side: str):
    """Select a DINO search crop on the exact prompt frame; no video-wide review."""
    frame, actual_idx, _ = robust_read_frame(video_path, frame_idx)
    h, w = frame.shape[:2]
    scale = min(1600.0 / float(w), 900.0 / float(h), 1.0)
    disp = frame if scale == 1.0 else cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    win = f"Select {side.upper()} DINO search crop on frame {actual_idx}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    x, y, rw, rh = [int(v) for v in roi]
    if rw <= 0 or rh <= 0:
        raise RuntimeError(f"No {side} DINO crop selected.")
    if scale != 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        rw = int(round(rw / scale))
        rh = int(round(rh / scale))

    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    rw = max(1, min(w - x, rw))
    rh = max(1, min(h - y, rh))
    crop = (x, y, rw, rh)
    print(f"[INFO][{side}] Selected DINO crop on frame {actual_idx}: {crop}")
    return crop


def preprocess_bgr_for_dino(frame_bgr: np.ndarray, max_size: int, patch_size: int, device):
    h, w = frame_bgr.shape[:2]
    scale = min(float(max_size) / float(max(h, w)), 1.0)
    in_w = max(patch_size, int(math.floor(w * scale / patch_size) * patch_size))
    in_h = max(patch_size, int(math.floor(h * scale / patch_size) * patch_size))
    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device), (in_w, in_h), (w, h)


def extract_patch_features(model, image_tensor, patch_size: int):
    with torch.inference_mode():
        if image_tensor.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_features(image_tensor) if hasattr(model, "forward_features") else model(image_tensor)
        else:
            out = model.forward_features(image_tensor) if hasattr(model, "forward_features") else model(image_tensor)

    if hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        for key in ("x_norm_patchtokens", "x_prenorm", "patch_tokens", "patchtokens"):
            if key in out:
                tokens = out[key]
                break
        else:
            raise RuntimeError(f"Could not find patch tokens in DINO output keys: {list(out.keys())}")
    elif torch.is_tensor(out):
        tokens = out
        if tokens.ndim == 3 and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
    else:
        raise RuntimeError(f"Unsupported DINO output type: {type(out)}")

    if tokens.ndim != 3:
        raise RuntimeError(f"Expected patch tokens shape B,N,C, got {tuple(tokens.shape)}")

    _, _, h, w = image_tensor.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    n_expected = grid_h * grid_w
    if tokens.shape[1] != n_expected:
        # Some ViT APIs include class/register tokens. Keep the last patch grid.
        if tokens.shape[1] > n_expected:
            tokens = tokens[:, -n_expected:, :]
        else:
            raise RuntimeError(f"Patch token count {tokens.shape[1]} does not match grid {grid_w}x{grid_h}")

    feats = tokens[0].float().reshape(grid_h, grid_w, -1)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy()


def click_examples(frame_bgr: np.ndarray, side: str, existing=None):
    positives = list((existing or {}).get("positive", []))
    negatives = list((existing or {}).get("negative", []))
    events = [("positive", p) for p in positives] + [("negative", p) for p in negatives]

    h, w = frame_bgr.shape[:2]
    base_scale = min(1500.0 / float(w), 900.0 / float(h), 1.0)
    disp_w = int(round(w * base_scale))
    disp_h = int(round(h * base_scale))
    view = {
        "zoom": 1.0,
        "min_zoom": 1.0,
        "max_zoom": 10.0,
        "center_x": w / 2.0,
        "center_y": h / 2.0,
        "last_mouse": (disp_w // 2, disp_h // 2),
        "dragging": False,
        "last_drag": (disp_w // 2, disp_h // 2),
    }
    win = f"DINO examples [{side}]"

    def clamp_center(cx, cy):
        view_w = w / view["zoom"]
        view_h = h / view["zoom"]
        half_w = view_w / 2.0
        half_h = view_h / 2.0
        cx = max(half_w, min(w - half_w, cx))
        cy = max(half_h, min(h - half_h, cy))
        return cx, cy

    def get_view_rect():
        view_w = w / view["zoom"]
        view_h = h / view["zoom"]
        view["center_x"], view["center_y"] = clamp_center(view["center_x"], view["center_y"])
        x0 = int(round(view["center_x"] - view_w / 2.0))
        y0 = int(round(view["center_y"] - view_h / 2.0))
        x1 = int(round(x0 + view_w))
        y1 = int(round(y0 + view_h))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return x0, y0, x1, y1

    def map_disp_to_orig(x, y):
        x0, y0, x1, y1 = get_view_rect()
        ox = x0 + (x / max(disp_w - 1, 1)) * (x1 - x0)
        oy = y0 + (y / max(disp_h - 1, 1)) * (y1 - y0)
        return [float(np.clip(ox, 0, w - 1)), float(np.clip(oy, 0, h - 1))]

    def redraw():
        x0, y0, x1, y1 = get_view_rect()
        disp = cv2.resize(frame_bgr[y0:y1, x0:x1], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        for label, pt in events:
            x, y = pt
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            dx = int(round((x - x0) * disp_w / max(x1 - x0, 1)))
            dy = int(round((y - y0) * disp_h / max(y1 - y0, 1)))
            color = (0, 255, 255) if label == "positive" else (0, 0, 255)
            cv2.circle(disp, (dx, dy), 7, color, -1)
            cv2.putText(disp, "P" if label == "positive" else "N", (dx + 9, dy - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 84), (0, 0, 0), -1)
        cv2.putText(disp, f"{side.upper()} positives={sum(e[0]=='positive' for e in events)} negatives={sum(e[0]=='negative' for e in events)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "LMB positive  RMB negative  x undo  q accept  Esc cancel",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(disp, f"wheel/+/- zoom  MMB drag pan  zoom={view['zoom']:.1f}x",
                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        return disp

    def on_mouse(event, x, y, flags, param):
        view["last_mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            events.append(("positive", map_disp_to_orig(x, y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            events.append(("negative", map_disp_to_orig(x, y)))
        elif event == cv2.EVENT_MBUTTONDOWN:
            view["dragging"] = True
            view["last_drag"] = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            view["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and view["dragging"]:
            x0, y0, x1, y1 = get_view_rect()
            ddx = x - view["last_drag"][0]
            ddy = y - view["last_drag"][1]
            view["center_x"] -= ddx * (x1 - x0) / max(disp_w - 1, 1)
            view["center_y"] -= ddy * (y1 - y0) / max(disp_h - 1, 1)
            view["last_drag"] = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            ox, oy = map_disp_to_orig(x, y)
            factor = 1.15 if flags > 0 else (1.0 / 1.15)
            view["zoom"] = max(view["min_zoom"], min(view["max_zoom"], view["zoom"] * factor))
            view["center_x"], view["center_y"] = ox, oy

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Example clicking canceled by user.")
        if key == ord("x") and events:
            events.pop()
        if key in (ord("+"), ord("="), ord("-"), ord("_")):
            ox, oy = map_disp_to_orig(*view["last_mouse"])
            factor = 1.15 if key in (ord("+"), ord("=")) else (1.0 / 1.15)
            view["zoom"] = max(view["min_zoom"], min(view["max_zoom"], view["zoom"] * factor))
            view["center_x"], view["center_y"] = ox, oy
        if key == ord("q"):
            cv2.destroyWindow(win)
            positives = [pt for label, pt in events if label == "positive"]
            negatives = [pt for label, pt in events if label == "negative"]
            if not positives:
                raise RuntimeError(f"No positive examples selected for {side}.")
            return {"positive": positives, "negative": negatives}


def load_examples(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_examples(path: Path, payload: dict):
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Saved examples: {path}")


def read_existing_dino_crop(side: str, work_dir: Path):
    path = work_dir / f"{side}_dino_crop_roi.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if not payload.get("crop_applied", False):
            return None
        crop = (
            int(payload.get("x", 0)),
            int(payload.get("y", 0)),
            int(payload.get("w", 0)),
            int(payload.get("h", 0)),
        )
        if crop[2] > 0 and crop[3] > 0:
            print(f"[INFO][{side}] Using existing DINO crop from {path}: {crop}")
            return crop
    except Exception as e:
        print(f"[WARN][{side}] Could not read existing DINO crop {path}: {e}")
    return None


def sample_feature(feats: np.ndarray, local_xy, crop_size, input_size, patch_size: int):
    crop_w, crop_h = crop_size
    in_w, in_h = input_size
    x, y = local_xy
    px = int(np.clip((x / max(crop_w, 1)) * in_w / patch_size, 0, feats.shape[1] - 1))
    py = int(np.clip((y / max(crop_h, 1)) * in_h / patch_size, 0, feats.shape[0] - 1))
    return feats[py, px]


def build_similarity_map(feats: np.ndarray, examples: dict, crop_size, input_size, patch_size: int, negative_weight: float):
    pos_feats = [sample_feature(feats, pt, crop_size, input_size, patch_size) for pt in examples["positive"]]
    pos_proto = np.mean(np.stack(pos_feats, axis=0), axis=0)
    pos_proto = pos_proto / max(np.linalg.norm(pos_proto), 1e-9)
    score = np.tensordot(feats, pos_proto, axes=([2], [0]))

    negs = examples.get("negative", [])
    if negs:
        neg_feats = [sample_feature(feats, pt, crop_size, input_size, patch_size) for pt in negs]
        neg_proto = np.mean(np.stack(neg_feats, axis=0), axis=0)
        neg_proto = neg_proto / max(np.linalg.norm(neg_proto), 1e-9)
        neg_score = np.tensordot(feats, neg_proto, axes=([2], [0]))
        score = score - float(negative_weight) * neg_score

    lo = float(np.percentile(score, 2))
    hi = float(np.percentile(score, 99))
    if hi <= lo:
        return np.zeros(score.shape, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def heatmap_to_candidates(score_map, crop, crop_size, min_score, min_distance, max_candidates, target_count=0):
    crop_x, crop_y, crop_w, crop_h = crop
    grid_h, grid_w = score_map.shape
    min_dist_grid = max(1, int(round(min_distance * min(grid_w / crop_w, grid_h / crop_h))))

    score_u8 = (score_map * 255).astype(np.uint8)
    k = max(3, min_dist_grid * 2 + 1)
    dilated = cv2.dilate(score_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    score_floor = 0.0 if target_count > 0 else float(min_score)
    peaks = (score_u8 == dilated) & (score_map >= score_floor)
    ys, xs = np.where(peaks)
    order = sorted(range(len(xs)), key=lambda i: float(score_map[ys[i], xs[i]]), reverse=True)
    keep_limit = int(target_count) if target_count > 0 else int(max_candidates)

    candidates: List[Candidate] = []
    for i in order:
        gx, gy = int(xs[i]), int(ys[i])
        local_x = (gx + 0.5) * crop_w / float(grid_w)
        local_y = (gy + 0.5) * crop_h / float(grid_h)
        if any(math.hypot(local_x - c.local_x, local_y - c.local_y) < min_distance for c in candidates):
            continue
        candidates.append(
            Candidate(
                x=crop_x + local_x,
                y=crop_y + local_y,
                local_x=local_x,
                local_y=local_y,
                score=float(score_map[gy, gx]),
                area=0.0,
            )
        )
        if keep_limit > 0 and len(candidates) >= keep_limit:
            break
    return sorted(candidates, key=lambda c: (c.local_y, c.local_x))


def draw_preview(frame_crop, candidates: List[Candidate]):
    preview = frame_crop.copy()
    text_scale = max(0.45, min(1.0, preview.shape[1] / 1600.0))
    thickness = max(1, int(round(text_scale * 2)))
    for idx, c in enumerate(candidates):
        px, py = int(round(c.local_x)), int(round(c.local_y))
        cv2.circle(preview, (px, py), 9, (0, 0, 0), -1)
        cv2.circle(preview, (px, py), 7, (0, 255, 255), -1)
        cv2.putText(preview, str(idx), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(preview, str(idx), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return preview


def save_candidates_csv(path: Path, candidates: List[Candidate]):
    ensure_parent(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "x", "y", "area", "method", "score", "local_x", "local_y"])
        writer.writeheader()
        for idx, c in enumerate(candidates):
            writer.writerow({
                "candidate_id": idx,
                "x": f"{c.x:.3f}",
                "y": f"{c.y:.3f}",
                "area": f"{c.area:.3f}",
                "method": c.method,
                "score": f"{c.score:.4f}",
                "local_x": f"{c.local_x:.3f}",
                "local_y": f"{c.local_y:.3f}",
            })


def save_candidates_json(path: Path, side, input_path, frame_idx, crop, args, candidates):
    ensure_parent(path)
    payload = {
        "side": side,
        "input": str(input_path),
        "frame": int(frame_idx),
        "crop": crop_to_dict(crop),
        "method": f"{args.backend}_exemplar",
        "model": args.model,
        "min_score": float(args.min_score),
        "min_distance": float(args.min_distance),
        "max_candidates": int(args.max_candidates),
        "target_count": int(get_target_count(args, side)),
        "coordinate_note": "x/y are full-frame pixels; local_x/local_y are cropped SAM2 prompt pixels.",
        "candidates": [
            {
                "candidate_id": idx,
                "x": round(c.x, 3),
                "y": round(c.y, 3),
                "local_x": round(c.local_x, 3),
                "local_y": round(c.local_y, 3),
                "area": round(c.area, 3),
                "method": c.method,
                "score": round(c.score, 4),
            }
            for idx, c in enumerate(candidates)
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def get_target_count(args, side):
    specific = args.left_target_count if side == "left" else args.right_target_count
    return int(specific or args.target_count or 0)


def compute_side_data(side, input_path, crop_arg, examples, model, device, args):
    frame_full, frame_idx, video_info = robust_read_frame(input_path, args.frame)
    crop_applied = crop_arg is not None
    crop = clip_crop(crop_arg, frame_full.shape)
    crop_x, crop_y, crop_w, crop_h = crop
    frame_crop = frame_full[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    image_tensor, input_size, crop_size = preprocess_bgr_for_dino(frame_crop, args.feature_max_size, args.patch_size, device)
    feats = extract_patch_features(model, image_tensor, args.patch_size)
    score_map = build_similarity_map(feats, examples, crop_size, input_size, args.patch_size, args.negative_weight)
    return {
        "side": side,
        "input_path": input_path,
        "frame_full": frame_full,
        "frame_idx": frame_idx,
        "video_info": video_info,
        "crop_applied": crop_applied,
        "crop": crop,
        "frame_crop": frame_crop,
        "score_map": score_map,
        "crop_size": crop_size,
    }


def candidates_from_side_data(data, args):
    target_count = get_target_count(args, data["side"])
    return heatmap_to_candidates(
        data["score_map"],
        data["crop"],
        data["crop_size"],
        args.min_score,
        args.min_distance,
        args.max_candidates,
        target_count=target_count,
    )


def write_side_outputs(data, candidates, args):
    side = data["side"]
    input_path = data["input_path"]
    frame_idx = data["frame_idx"]
    crop = data["crop"]
    frame_full = data["frame_full"]
    frame_crop = data["frame_crop"]
    crop_applied = data["crop_applied"]
    crop_x, crop_y, crop_w, crop_h = crop
    score_map = data["score_map"]

    heat = cv2.resize(score_map, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    heat_overlay = cv2.addWeighted(frame_crop, 0.55, heat_color, 0.45, 0)
    preview = draw_preview(frame_crop, candidates)

    csv_path = args.work_dir / f"{side}_candidates.csv"
    json_path = args.work_dir / f"{side}_candidates.json"
    preview_path = args.work_dir / f"{side}_candidates_preview.png"
    heat_path = args.work_dir / f"{side}_heatmap.png"
    crop_path = args.work_dir / f"{side}_dino_crop_roi.json"

    save_candidates_csv(csv_path, candidates)
    save_candidates_json(json_path, side, input_path, frame_idx, crop, args, candidates)
    save_crop_meta(crop_path, side, input_path, crop, crop_applied, frame_full.shape)
    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(heat_path), heat_overlay)
    print(f"[OK][{side}] candidates={len(candidates)} csv={csv_path}")
    print(f"[OK][{side}] preview={preview_path}")
    print(f"[OK][{side}] heatmap={heat_path}")
    print(f"[OK][{side}] dino_crop={crop_path}")
    return len(candidates), preview_path, heat_path


def process_side(side, input_path, crop_arg, examples, model, device, args):
    data = compute_side_data(side, input_path, crop_arg, examples, model, device, args)
    candidates = candidates_from_side_data(data, args)
    target = get_target_count(args, side)
    if target > 0 and len(candidates) != target:
        print(f"[WARN][{side}] target_count={target}, but found {len(candidates)} spaced peaks.")
    return write_side_outputs(data, candidates, args)


def print_peak_params(args):
    print(
        "\nCurrent DINO peak params:\n"
        f"  min_score          = {args.min_score:g}\n"
        f"  min_distance       = {args.min_distance:g}\n"
        f"  max_candidates     = {args.max_candidates}  (0 keeps all)\n"
        f"  target_count       = {args.target_count}  (0 disables)\n"
        f"  left_target_count  = {args.left_target_count}  (0 uses target_count)\n"
        f"  right_target_count = {args.right_target_count}  (0 uses target_count)\n"
    )


def update_peak_param(args, key, value):
    key = key.strip().replace("-", "_")
    if key in ("min_score", "min_distance"):
        setattr(args, key, float(value))
    elif key in ("max_candidates", "target_count", "left_target_count", "right_target_count"):
        setattr(args, key, int(float(value)))
    else:
        raise ValueError(f"Unknown peak parameter: {key}")
    if args.min_score < 0 or args.min_score > 1:
        raise ValueError("min_score must be between 0 and 1")
    if args.min_distance < 0:
        raise ValueError("min_distance must be >= 0")
    if min(args.max_candidates, args.target_count, args.left_target_count, args.right_target_count) < 0:
        raise ValueError("counts must be >= 0")


def run_peak_tuning(side_data: Dict[str, dict], args):
    print(
        "\n[INFO] DINO peak tuning mode. DINO heatmaps are cached; apply does not rerun DINO.\n"
        "Commands:\n"
        "  show\n"
        "  min_distance 24\n"
        "  min_score 0.68\n"
        "  target_count 144\n"
        "  left_target_count 144\n"
        "  right_target_count 144\n"
        "  apply        write candidate previews with current params\n"
        "  open         open latest left/right previews\n"
        "  save         save params and continue\n"
        "  cancel\n"
    )
    latest_paths = {}
    print_peak_params(args)

    def apply_current():
        nonlocal latest_paths
        latest_paths = {}
        for side, data in side_data.items():
            candidates = candidates_from_side_data(data, args)
            target = get_target_count(args, side)
            print(f"[INFO][{side}] candidates={len(candidates)} target={target or 'off'}")
            _, preview_path, heat_path = write_side_outputs(data, candidates, args)
            latest_paths[side] = preview_path
        if not args.no_open_preview:
            for path in latest_paths.values():
                open_image(path)

    apply_current()
    while True:
        try:
            command = input("dino-peaks> ").strip()
        except EOFError:
            command = "save"
        if not command:
            continue
        lower = command.lower()
        if lower in ("show", "status", "?"):
            print_peak_params(args)
            continue
        if lower in ("apply", "a", "run"):
            apply_current()
            continue
        if lower in ("open", "preview"):
            for path in latest_paths.values():
                open_image(path)
            continue
        if lower in ("save", "s", "done", "q"):
            save_peak_params(args.params_file, args)
            return args
        if lower in ("cancel", "exit", "esc"):
            raise RuntimeError("DINO peak tuning canceled by user.")

        parts = command.split()
        try:
            if len(parts) >= 3 and parts[0].lower() == "set":
                update_peak_param(args, parts[1], " ".join(parts[2:]))
            elif len(parts) >= 2:
                update_peak_param(args, parts[0], " ".join(parts[1:]))
            else:
                print("[WARN] Could not parse command. Example: min_distance 24")
                continue
            print_peak_params(args)
            print("[INFO] Value changed. Type apply to update previews.")
        except Exception as e:
            print(f"[WARN] {e}")


def resolve_crops(args):
    both_crop = parse_crop_arg(args.dino_crop or args.crop, "--dino-crop")
    left_crop = parse_crop_arg(args.left_dino_crop or args.left_crop, "--left-dino-crop") or both_crop
    right_crop = parse_crop_arg(args.right_dino_crop or args.right_crop, "--right-dino-crop") or both_crop
    if args.select_crop:
        print("[WARN] --select-crop is deprecated here; using it as --select-dino-crop. SAM2 crop is unchanged.")
    if args.select_crop or args.select_dino_crop:
        left_crop = select_dino_crop_on_frame(args.left_input, args.frame, "left")
        right_crop = select_dino_crop_on_frame(args.right_input, args.frame, "right")
    elif args.examples and args.examples.exists():
        try:
            payload = json.loads(args.examples.read_text())
            if left_crop is None and (payload.get("left", {}).get("dino_crop") or payload.get("left", {}).get("crop")):
                c = payload["left"].get("dino_crop") or payload["left"]["crop"]
                left_crop = (int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"]))
                print(f"[INFO][left] Using DINO crop from examples: {left_crop}")
            if right_crop is None and (payload.get("right", {}).get("dino_crop") or payload.get("right", {}).get("crop")):
                c = payload["right"].get("dino_crop") or payload["right"]["crop"]
                right_crop = (int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"]))
                print(f"[INFO][right] Using DINO crop from examples: {right_crop}")
        except Exception as e:
            print(f"[WARN] Could not read crop from examples {args.examples}: {e}")
    if not args.ignore_existing_crops:
        if left_crop is None:
            left_crop = read_existing_dino_crop("left", args.work_dir)
            if left_crop is None:
                left_crop = read_existing_crop("left", args.out_root)
                if left_crop is not None:
                    print("[INFO][left] No DINO crop found; using SAM2 crop as fallback.")
        if right_crop is None:
            right_crop = read_existing_dino_crop("right", args.work_dir)
            if right_crop is None:
                right_crop = read_existing_crop("right", args.out_root)
                if right_crop is not None:
                    print("[INFO][right] No DINO crop found; using SAM2 crop as fallback.")
    return left_crop, right_crop


def prepare_examples(args, left_crop, right_crop):
    examples = load_examples(args.examples) if args.examples else None
    examples = examples or {"frame": int(args.frame), "left": {}, "right": {}}

    if args.click_examples:
        for side, input_path, crop in (("left", args.left_input, left_crop), ("right", args.right_input, right_crop)):
            frame_full, _, _ = robust_read_frame(input_path, args.frame)
            crop = clip_crop(crop, frame_full.shape)
            x, y, w, h = crop
            frame_crop = frame_full[y:y + h, x:x + w]
            existing = examples.get(side, {})
            examples[side] = click_examples(frame_crop, side, existing=existing)
            examples[side]["dino_crop"] = crop_to_dict(crop)
            examples[side]["input"] = str(input_path)
        save_examples(args.examples, examples)

    for side in ("left", "right"):
        if side not in examples or not examples[side].get("positive"):
            raise RuntimeError(f"No positive DINO examples found for {side}. Use --click-examples first.")
    return examples


def build_argparser():
    parser = argparse.ArgumentParser(description="Use DINOv3 exemplar matching to propose SAM2 prompt points.")
    parser.add_argument("--left-input", default=Path("in") / "left.mp4", type=Path)
    parser.add_argument("--right-input", default=Path("in") / "right.mp4", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--examples", type=Path, default=None)
    parser.add_argument("--params-file", type=Path, default=None)
    parser.add_argument("--ignore-saved-params", action="store_true")
    parser.add_argument("--click-examples", action="store_true")
    parser.add_argument("--dino-crop", default=None, help="Optional DINO search crop for both sides: x,y,w,h.")
    parser.add_argument("--left-dino-crop", default=None, help="Optional LEFT DINO search crop: x,y,w,h.")
    parser.add_argument("--right-dino-crop", default=None, help="Optional RIGHT DINO search crop: x,y,w,h.")
    parser.add_argument("--select-dino-crop", action="store_true", help="Interactively select DINO search crops on --frame only; no across-frame review.")
    parser.add_argument("--crop", default=None, help="Deprecated alias for --dino-crop.")
    parser.add_argument("--left-crop", default=None, help="Deprecated alias for --left-dino-crop.")
    parser.add_argument("--right-crop", default=None, help="Deprecated alias for --right-dino-crop.")
    parser.add_argument("--select-crop", action="store_true", help="Deprecated alias for --select-dino-crop.")
    parser.add_argument("--ignore-existing-crops", action="store_true")
    parser.add_argument("--backend", choices=["dinov2", "huggingface", "torchhub"], default="dinov2")
    parser.add_argument("--hf-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated DINOv3 weights.")
    parser.add_argument("--dinov2-repo", default=None)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--feature-max-size", type=int, default=896)
    parser.add_argument("--min-score", type=float, default=0.72)
    parser.add_argument("--negative-weight", type=float, default=0.35)
    parser.add_argument("--min-distance", type=float, default=28.0)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 keeps all")
    parser.add_argument("--target-count", type=int, default=0, help="Keep strongest N spaced peaks per side; 0 disables.")
    parser.add_argument("--left-target-count", type=int, default=0, help="Override target count for left side.")
    parser.add_argument("--right-target-count", type=int, default=0, help="Override target count for right side.")
    parser.add_argument("--tune-peaks", action="store_true", help="Cache DINO heatmaps once, then text-tune peak picking without rerunning DINO.")
    parser.add_argument("--no-open-preview", action="store_true")
    return parser


def validate_args(args):
    if args.examples is None:
        args.examples = args.work_dir / "examples.json"
    if args.params_file is None:
        args.params_file = args.work_dir / "peak_params.json"
    if not args.left_input.exists():
        raise FileNotFoundError(f"LEFT input not found: {args.left_input}")
    if not args.right_input.exists():
        raise FileNotFoundError(f"RIGHT input not found: {args.right_input}")
    if args.min_score < 0 or args.min_score > 1:
        raise ValueError("--min-score must be between 0 and 1")
    if args.min_distance < 0:
        raise ValueError("--min-distance must be >= 0")
    if args.target_count < 0 or args.left_target_count < 0 or args.right_target_count < 0:
        raise ValueError("target counts must be >= 0")
    if args.feature_max_size < args.patch_size:
        raise ValueError("--feature-max-size must be >= --patch-size")


def main():
    args = build_argparser().parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    validate_args(args)
    load_peak_params(args.params_file, args)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    left_crop, right_crop = resolve_crops(args)
    examples = prepare_examples(args, left_crop, right_crop)

    device = resolve_device(args.device)
    model = load_dino_model(args, device)

    if args.tune_peaks:
        side_data = {
            "left": compute_side_data("left", args.left_input, left_crop, examples["left"], model, device, args),
            "right": compute_side_data("right", args.right_input, right_crop, examples["right"], model, device, args),
        }
        args = run_peak_tuning(side_data, args)
        left_candidates = candidates_from_side_data(side_data["left"], args)
        right_candidates = candidates_from_side_data(side_data["right"], args)
        left_count, left_preview, left_heat = write_side_outputs(side_data["left"], left_candidates, args)
        right_count, right_preview, right_heat = write_side_outputs(side_data["right"], right_candidates, args)
    else:
        left_count, left_preview, left_heat = process_side(
            "left", args.left_input, left_crop, examples["left"], model, device, args
        )
        right_count, right_preview, right_heat = process_side(
            "right", args.right_input, right_crop, examples["right"], model, device, args
        )

    summary = {
        "frame": int(args.frame),
        "left_candidates": int(left_count),
        "right_candidates": int(right_count),
        "examples": str(args.examples),
        "model": args.model,
    }
    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] summary={summary_path}")

    if not args.no_open_preview:
        open_image(left_preview)
        open_image(left_heat)
        open_image(right_preview)
        open_image(right_heat)


if __name__ == "__main__":
    main()
