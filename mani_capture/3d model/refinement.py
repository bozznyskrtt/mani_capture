from __future__ import annotations

import os
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

# =========================
# User settings
# =========================
#DEFAULT_ROOT = "/home/bozznyskrtt/pcl_ws/teddybear/session17"
DEFAULT_ROOT = os.path.expandvars("$HOME/pcl_ws/teddybear/session17")
ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
DATASET_ROOT = ROOT + "/crop"
META_PATH = os.path.join(ROOT, "meta.txt")
CHECKPOINT_PATH = os.path.expandvars("$HOME/hebi_ws/src/snapshot/snapshot/3d model/checkpoints_multisession_640x480/best_multisession_640x480.pth")

OUTPUT_DIR = ROOT + "/test_outputs_mask_depth_refined_640x480"

# Output resolution — must match what the checkpoint was trained with
OUT_H = 480
OUT_W = 640

# Decoder square size used during training (power-of-2: 64/128/256/512)
DECODER_SIZE = 256

THRESHOLD = 0.5

# choose subtraction mode:
# "zero_out"     -> set predicted arm pixels to 0
# "depth_minus"  -> subtract predicted depth value from original depth
SUBTRACT_MODE = "depth_minus"

# depth normalization fallback
DEPTH_NORM_FALLBACK_MAX_MM = 1000.0

# Set to None to use checkpoint stored joints
SELECTED_JOINTS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# =========================
# Refinement settings
# =========================
ENABLE_REFINEMENT = True

# What to use as the observation mask during refinement:
# "full_nonzero" -> all non-zero pixels in observed depth (arm + object)
# "close_to_pred" -> only pixels near the raw predicted mask centroid/depth region
OBS_MASK_MODE = "full_nonzero"

# Coarse search — ranges scaled up ~5× relative to 128×128 notebook
COARSE_DX    = list(range(-50, 51, 10))
COARSE_DY    = list(range(-40, 41, 10))
COARSE_THETA = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
COARSE_SCALE = [1.0]

# Fine search around the best coarse result
USE_FINE_REFINEMENT = True
FINE_DXY_RADIUS  = 8
FINE_THETA_RADIUS = 2.0
FINE_THETA_STEP   = 0.5
FINE_SCALE = [1.0]

# Loss weights
LAMBDA_MASK   = 1.0
LAMBDA_DEPTH  = 0.5
LAMBDA_CENTER = 0.05

# Optional guard:
# require at least this much IoU improvement before replacing raw prediction
MIN_IOU_IMPROVEMENT_TO_ACCEPT = 0.0

# GPU batch size for affine warp (reduce if OOM)
WARP_CHUNK = 32

#v1

# =========================
# Utils
# =========================
@dataclass
class FrameMetrics:
    seq: int
    filename: str
    raw_iou: float
    refined_iou: float
    raw_dice: float
    refined_dice: float
    raw_depth_mae: float
    refined_depth_mae: float
    pixel_acc: float
    precision: float
    recall: float
    align_dx: float
    align_dy: float
    align_theta: float
    align_scale: float
    align_loss: float

def compute_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / (union + 1e-8))
    dice = float((2.0 * inter) / (pred.sum() + gt.sum() + 1e-8))

    total_pixels = gt.size
    pixel_acc = float(np.sum(pred == gt)) / total_pixels

    precision = float(inter / (pred.sum() + 1e-8))
    recall = float(inter / (gt.sum() + 1e-8))

    return {"iou": iou, "dice": dice, "pixel_acc": pixel_acc, "precision": precision, "recall": recall}

def compute_depth_mae(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> float:
    overlap = (pred_depth > 0) & (mask > 0)
    if overlap.sum() == 0:
        return 1e6
    return float(np.abs(pred_depth[overlap] - gt_depth[overlap]).mean())

def safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def flatten_matrix(mat: List[List[float]], expected_shape: Tuple[int, int] = (4, 4)) -> List[float]:
    if mat is None:
        return [0.0] * (expected_shape[0] * expected_shape[1])

    arr = np.asarray(mat, dtype=np.float32)
    if arr.shape != expected_shape:
        out = np.zeros(expected_shape, dtype=np.float32)
        h = min(expected_shape[0], arr.shape[0])
        w = min(expected_shape[1], arr.shape[1])
        out[:h, :w] = arr[:h, :w]
        arr = out

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.reshape(-1).tolist()


def load_yaml_meta(meta_path: str) -> Dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace(": nan", ": .nan")
    data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Failed to parse metadata file: {meta_path}")
    if "captures" not in data or not isinstance(data["captures"], list):
        raise ValueError("meta.txt must contain a top-level 'captures' list.")

    return data


def resize_nearest(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize to (out_w, out_h) using nearest-neighbour interpolation."""
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def resize_linear(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize to (out_w, out_h) using bilinear interpolation."""
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def depth_to_binary_mask(depth_img: np.ndarray) -> np.ndarray:
    if depth_img.ndim == 3:
        depth_img = depth_img[..., 0]
    return (depth_img > 0).astype(np.float32)


def save_mask_png(mask01: np.ndarray, out_path: str) -> None:
    img = (np.clip(mask01, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def save_depth_vis(depth01: np.ndarray, out_path: str) -> None:
    img = (np.clip(depth01, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def save_raw_depth_png(depth_raw: np.ndarray, out_path: str) -> None:
    if depth_raw.dtype not in [np.uint8, np.uint16]:
        if depth_raw.max() <= 255:
            depth_raw = depth_raw.astype(np.uint8)
        else:
            depth_raw = depth_raw.astype(np.uint16)
    Image.fromarray(depth_raw).save(out_path)


def normalize_depth(depth_img: np.ndarray, mode: str, global_max_mm: float) -> np.ndarray:
    depth = depth_img.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]

    valid = depth > 0
    out = np.zeros_like(depth, dtype=np.float32)

    if not np.any(valid):
        return out

    if mode == "per_image":
        dmax = depth[valid].max()
        if dmax > 0:
            out[valid] = depth[valid] / dmax
    elif mode == "global":
        out[valid] = depth[valid] / float(global_max_mm)
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown depth norm mode: {mode}")

    return out


def denormalize_depth(depth01: np.ndarray, ref_depth_raw: np.ndarray, mode: str, global_max_mm: float) -> np.ndarray:
    ref = ref_depth_raw.astype(np.float32)
    valid = ref > 0

    out = np.zeros_like(depth01, dtype=np.float32)

    if mode == "per_image":
        if np.any(valid):
            dmax = ref[valid].max()
            out = depth01 * dmax
    elif mode == "global":
        out = depth01 * float(global_max_mm)
    else:
        raise ValueError(f"Unknown depth norm mode: {mode}")

    return out


def make_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_b = gt.astype(bool)
    pr_b = pred.astype(bool)

    tp = gt_b & pr_b
    fn = gt_b & (~pr_b)
    fp = (~gt_b) & pr_b

    overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    overlay[tp] = [255, 255, 255]
    overlay[fn] = [255, 0, 0]
    overlay[fp] = [0, 255, 0]
    return overlay


def save_color_png(img_rgb: np.ndarray, out_path: str) -> None:
    Image.fromarray(img_rgb.astype(np.uint8)).save(out_path)


def add_label(img, text):
    img = img.copy()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # background box
    cv2.rectangle(img, (0, 0), (200, 24), (0, 0, 0), -1)

    cv2.putText(
        img,
        text,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img

# =========================
# Model  (640×480 variant)
# =========================
class CoordConv2d(nn.Module):
    """Conv2d that appends normalised (y, x) coordinate channels before the conv."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.conv(torch.cat([x, ys, xs], dim=1))


class JointToMaskDepthNet(nn.Module):
    """MLP → CoordConv decoder → (out_h, out_w) mask + depth.

    The decoder upsamples from 8×8 to decoder_size×decoder_size (a square
    power-of-2), then a final F.interpolate maps to the exact (out_h, out_w).
    """

    def __init__(self, input_dim: int, out_h: int = 480, out_w: int = 640,
                 decoder_size: int = 256) -> None:
        super().__init__()
        assert decoder_size in (64, 128, 256, 512), \
            "decoder_size must be one of [64, 128, 256, 512]."
        self.out_h        = out_h
        self.out_w        = out_w
        self.decoder_size = decoder_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 8 * 8 * 128),
            nn.ReLU(inplace=True),
        )

        layers: List[nn.Module] = []
        in_ch    = 128
        cur_size = 8
        while cur_size < decoder_size:
            out_ch = max(in_ch // 2, 16)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                CoordConv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch     = out_ch
            cur_size *= 2

        self.shared_decoder = nn.Sequential(*layers)

        self.mask_head = CoordConv2d(in_ch, 1, kernel_size=1)
        self.depth_head = nn.Sequential(
            CoordConv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        b    = x.shape[0]
        z    = self.mlp(x).view(b, 128, 8, 8)
        feat = self.shared_decoder(z)          # (B, C, decoder_size, decoder_size)

        mask_logits = self.mask_head(feat)
        depth_pred  = self.depth_head(feat)

        # Stretch to target resolution
        mask_logits = F.interpolate(mask_logits, size=(self.out_h, self.out_w),
                                    mode="bilinear", align_corners=False)
        depth_pred  = F.interpolate(depth_pred,  size=(self.out_h, self.out_w),
                                    mode="bilinear", align_corners=False)
        return mask_logits, depth_pred
    
    # =========================
# Feature builder
# =========================
def build_feature_vector(
    capture: Dict,
    selected_joints: List[str],
    use_velocities: bool = False,
    use_efforts: bool = False,
    use_t_cam_to_ee: bool = False,
    use_t_base_to_ee: bool = False,
) -> np.ndarray:
    joint_names     = capture.get("joint_names", [])
    joint_positions = capture.get("joint_positions", [])
    joint_velocities = capture.get("joint_velocities", [])
    joint_efforts   = capture.get("joint_efforts", [])

    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    feats: List[float] = []

    for jn in selected_joints:
        idx = name_to_idx.get(jn, None)
        feats.append(
            safe_float(joint_positions[idx])
            if idx is not None and idx < len(joint_positions)
            else 0.0
        )

    if use_velocities:
        for jn in selected_joints:
            idx = name_to_idx.get(jn, None)
            feats.append(
                safe_float(joint_velocities[idx])
                if idx is not None and idx < len(joint_velocities)
                else 0.0
            )

    if use_efforts:
        for jn in selected_joints:
            idx = name_to_idx.get(jn, None)
            feats.append(
                safe_float(joint_efforts[idx])
                if idx is not None and idx < len(joint_efforts)
                else 0.0
            )

    if use_t_cam_to_ee:
        feats.extend(flatten_matrix(capture.get("T_cam_to_ee")))

    if use_t_base_to_ee:
        feats.extend(flatten_matrix(capture.get("T_base_to_ee")))

    return np.asarray(feats, dtype=np.float32)

# =========================
# Depth subtraction
# =========================
def subtract_predicted_arm(
    original_depth_raw: np.ndarray,
    pred_mask01: np.ndarray,
    pred_depth01: np.ndarray,
    depth_norm_mode: str,
    depth_max_mm: float,
    tolerance_mm: float = 30.0,
) -> np.ndarray:
    """
    Occlusion-aware arm removal.

    Rule:
    - remove pixel only if predicted arm is actually visible in front
      of the observed depth at that pixel
    - if object is closer than predicted arm, keep the object pixel

    Assumes smaller depth = closer.
    """
    original     = original_depth_raw.astype(np.float32)
    pred_mask_bin = pred_mask01 > 0.5

    pred_depth_raw = denormalize_depth(
        pred_depth01,
        ref_depth_raw=original_depth_raw,
        mode=depth_norm_mode,
        global_max_mm=depth_max_mm,
    ).astype(np.float32)

    out = original.copy()

    valid_obs  = original > 0
    valid_pred = pred_mask_bin & (pred_depth_raw > 0)

    # arm is removable only if it is not behind the visible surface
    visible_arm = (
        valid_obs &
        valid_pred &
        (pred_depth_raw <= original + tolerance_mm)
    )

    out[visible_arm] = 0.0

    if original_depth_raw.dtype == np.uint16:
        out = np.clip(out, 0, 65535).astype(np.uint16)
    else:
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out

# =========================
# Refinement helpers
# =========================
def compute_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a     = a.astype(bool)
    b     = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + eps))


def mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ca = mask_centroid(mask_a)
    cb = mask_centroid(mask_b)
    if ca is None or cb is None:
        return 1e6
    return float(np.linalg.norm(ca - cb))


def compute_overlap_depth_mae(
    pred_depth: np.ndarray,
    obs_depth: np.ndarray,
    pred_mask: np.ndarray,
    obs_mask: np.ndarray,
) -> float:
    overlap = (pred_mask > 0) & (obs_mask > 0)
    if overlap.sum() == 0:
        return 1e6
    return float(np.abs(pred_depth[overlap] - obs_depth[overlap]).mean())


def warp_pair(
    mask: np.ndarray,
    depth: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    scale: float = 1.0,
):
    h, w = mask.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, theta_deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    warped_mask = cv2.warpAffine(
        mask.astype(np.float32), M, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_depth = cv2.warpAffine(
        depth.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_mask = (warped_mask > 0.5).astype(np.uint8)
    warped_depth[warped_mask == 0] = 0.0
    return warped_mask, warped_depth, M


def build_observation_mask(
    obs_depth01: np.ndarray,
    raw_pred_mask: np.ndarray,
    mode: str = "full_nonzero",
) -> np.ndarray:
    full_mask = (obs_depth01 > 0).astype(np.uint8)
    if mode == "full_nonzero":
        return full_mask
    if mode == "close_to_pred":
        kernel = np.ones((43, 43), np.uint8)
        roi    = cv2.dilate(raw_pred_mask.astype(np.uint8), kernel, iterations=1)
        return (full_mask * roi).astype(np.uint8)
    raise ValueError(f"Unknown OBS_MASK_MODE: {mode}")


def alignment_loss(
    warped_mask: np.ndarray,
    warped_depth: np.ndarray,
    obs_mask: np.ndarray,
    obs_depth: np.ndarray,
    lambda_mask: float = 1.0,
    lambda_depth: float = 0.5,
    lambda_center: float = 0.05,
):
    """Single-pair loss (kept for ad-hoc use; not called in the search loop)."""
    iou            = compute_iou(warped_mask, obs_mask)
    depth_mae      = compute_overlap_depth_mae(warped_depth, obs_depth, warped_mask, obs_mask)
    center_penalty = centroid_distance(warped_mask, obs_mask)
    loss = (
        lambda_mask   * (1.0 - iou)
        + lambda_depth  * depth_mae
        + lambda_center * center_penalty / max(OUT_H, OUT_W)
    )
    return float(loss), float(iou), float(depth_mae), float(center_penalty)


# -- GPU search helpers ---------------------------------------------------

def _build_affine_batch(
    combos: list,
    H: int, W: int,
    device,
) -> torch.Tensor:
    """Convert (dx, dy, theta_deg, scale) combos to normalised torch affine matrices."""
    N  = len(combos)
    sx = 2.0 / (W - 1)
    sy = 2.0 / (H - 1)
    r  = float(H - 1) / float(W - 1)

    thetas = torch.zeros(N, 2, 3, dtype=torch.float32)
    for i, (dx, dy, theta_deg, scale) in enumerate(combos):
        a     = math.radians(theta_deg)
        cos_a = scale * math.cos(a)
        sin_a = scale * math.sin(a)
        thetas[i] = torch.tensor([
            [ cos_a,       r * sin_a,    sx * dx],
            [-sin_a / r,   cos_a,        sy * dy],
        ])
    return thetas.to(device)


def _search_alignment(
    pred_mask: np.ndarray,
    pred_depth: np.ndarray,
    obs_mask: np.ndarray,
    obs_depth: np.ndarray,
    dx_values,
    dy_values,
    theta_values,
    scale_values,
):
    """Grid search with GPU affine warps scored on GPU.

    Candidates are scored and compared *online* inside the chunk loop so no
    N x H x W intermediate arrays are ever allocated.  The previous version
    allocated all_wm (N,H,W) + all_wd (N,H,W) before scoring, which reached
    ~4 GB for the fine search and crashed the kernel.
    """
    H, W = pred_mask.shape
    combos = [
        (dx, dy, theta, scale)
        for dx    in dx_values
        for dy    in dy_values
        for theta in theta_values
        for scale in scale_values
    ]
    N = len(combos)

    thetas  = _build_affine_batch(combos, H, W, DEVICE)
    mask_t  = torch.from_numpy(pred_mask.astype(np.float32)).to(DEVICE).unsqueeze(0).unsqueeze(0)
    depth_t = torch.from_numpy(pred_depth.astype(np.float32)).to(DEVICE).unsqueeze(0).unsqueeze(0)

    om_gpu = torch.from_numpy(obs_mask.astype(np.float32)).to(DEVICE)
    od_gpu = torch.from_numpy(obs_depth.astype(np.float32)).to(DEVICE)
    om_b   = om_gpu > 0.5

    # Coordinate grids and obs centroid -- built once per search call
    ys_gpu  = torch.arange(H, dtype=torch.float32, device=DEVICE).view(H, 1)
    xs_gpu  = torch.arange(W, dtype=torch.float32, device=DEVICE).view(1, W)
    has_obs = bool(om_b.float().sum() > 0)
    if has_obs:
        om_cnt = om_b.float().sum().clamp(min=1)
        om_cx  = (om_b.float() * xs_gpu).sum() / om_cnt
        om_cy  = (om_b.float() * ys_gpu).sum() / om_cnt
    else:
        om_cx = om_cy = torch.tensor(0.0, device=DEVICE)

    best_loss  = float("inf")
    best_i_abs = 0
    best_wm_np = None
    best_wd_np = None
    best_iou   = 0.0
    best_dmae  = 1e6
    best_cp    = 1e6

    for start in range(0, N, WARP_CHUNK):
        end   = min(start + WARP_CHUNK, N)
        chunk = end - start

        grid     = F.affine_grid(thetas[start:end], (chunk, 1, H, W), align_corners=True)
        wm_chunk = F.grid_sample(mask_t.expand(chunk, -1, -1, -1),  grid,
                                  mode="nearest",  padding_mode="zeros", align_corners=True)
        wd_chunk = F.grid_sample(depth_t.expand(chunk, -1, -1, -1), grid,
                                  mode="bilinear", padding_mode="zeros", align_corners=True)

        wm   = wm_chunk[:, 0]          # (chunk, H, W)
        wd   = wd_chunk[:, 0]          # (chunk, H, W)
        wm_b = wm > 0.5
        wd   = wd * wm_b.float()       # zero non-mask depth

        # IoU
        inter = (wm_b & om_b.unsqueeze(0)).float().sum(dim=(1, 2))
        union = (wm_b | om_b.unsqueeze(0)).float().sum(dim=(1, 2))
        iou   = inter / (union + 1e-8)

        # Depth MAE over overlap
        overlap     = wm_b & om_b.unsqueeze(0)
        overlap_cnt = overlap.float().sum(dim=(1, 2))
        depth_mae   = (
            (wd - od_gpu.unsqueeze(0)).abs() * overlap.float()
        ).sum(dim=(1, 2)) / overlap_cnt.clamp(min=1)
        depth_mae   = torch.where(overlap_cnt > 0, depth_mae,
                                   torch.full_like(depth_mae, 1e6))

        # Centroid distance
        wm_cnt = wm_b.float().sum(dim=(1, 2)).clamp(min=1)
        wm_cx  = (wm_b.float() * xs_gpu.unsqueeze(0)).sum(dim=(1, 2)) / wm_cnt
        wm_cy  = (wm_b.float() * ys_gpu.unsqueeze(0)).sum(dim=(1, 2)) / wm_cnt
        cp     = torch.sqrt((wm_cx - om_cx) ** 2 + (wm_cy - om_cy) ** 2)
        has_wm = wm_b.float().sum(dim=(1, 2)) > 0
        if not has_obs:
            cp = torch.full_like(cp, 1e6)
        else:
            cp = torch.where(has_wm, cp, torch.full_like(cp, 1e6))

        loss = (
            LAMBDA_MASK   * (1.0 - iou)
            + LAMBDA_DEPTH  * depth_mae
            + LAMBDA_CENTER * cp / max(H, W)
        )

        local_best = int(loss.argmin())
        local_loss = loss[local_best].item()
        if local_loss < best_loss:
            best_loss  = local_loss
            best_i_abs = start + local_best
            best_wm_np = (wm[local_best] > 0.5).byte().cpu().numpy()
            best_wd_np = wd[local_best].cpu().numpy()
            best_iou   = float(iou[local_best].item())
            best_dmae  = float(depth_mae[local_best].item())
            best_cp    = float(cp[local_best].item())

    dx, dy, theta, scale = combos[best_i_abs]
    _, _, M = warp_pair(pred_mask, pred_depth, dx, dy, theta, scale)

    return {
        "loss"           : best_loss,
        "iou"            : best_iou,
        "depth_mae"      : best_dmae,
        "center_penalty" : best_cp,
        "dx"             : float(dx),
        "dy"             : float(dy),
        "theta"          : float(theta),
        "scale"          : float(scale),
        "mask"           : best_wm_np,
        "depth"          : best_wd_np,
        "M"              : M,
    }


def refine_alignment(
    pred_mask: np.ndarray,
    pred_depth: np.ndarray,
    obs_mask: np.ndarray,
    obs_depth: np.ndarray,
):
    coarse_best = _search_alignment(
        pred_mask=pred_mask, pred_depth=pred_depth,
        obs_mask=obs_mask, obs_depth=obs_depth,
        dx_values=COARSE_DX, dy_values=COARSE_DY,
        theta_values=COARSE_THETA, scale_values=COARSE_SCALE,
    )

    if not USE_FINE_REFINEMENT:
        return coarse_best

    best_dx    = int(round(coarse_best["dx"]))
    best_dy    = int(round(coarse_best["dy"]))
    best_theta = float(coarse_best["theta"])

    fine_dx = range(best_dx - FINE_DXY_RADIUS, best_dx + FINE_DXY_RADIUS + 1, 1)
    fine_dy = range(best_dy - FINE_DXY_RADIUS, best_dy + FINE_DXY_RADIUS + 1, 1)

    fine_theta = []
    t = best_theta - FINE_THETA_RADIUS
    while t <= best_theta + FINE_THETA_RADIUS + 1e-8:
        fine_theta.append(round(t, 3))
        t += FINE_THETA_STEP

    fine_best = _search_alignment(
        pred_mask=pred_mask, pred_depth=pred_depth,
        obs_mask=obs_mask, obs_depth=obs_depth,
        dx_values=fine_dx, dy_values=fine_dy,
        theta_values=fine_theta, scale_values=FINE_SCALE,
    )

    return fine_best if fine_best["loss"] < coarse_best["loss"] else coarse_best

# =========================
# Frame prefetch helper
# =========================
def _load_frame_data(args):
    """Load and preprocess one frame's depth image on a background thread."""
    cap, dataset_root, out_w, out_h, depth_norm_mode, depth_max_mm = args
    depth_png  = cap.get("depth_png", "")
    depth_path = os.path.join(dataset_root, depth_png)
    if not depth_png or not os.path.exists(depth_path):
        return None
    depth_raw = np.array(Image.open(depth_path))
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    gt_mask    = depth_to_binary_mask(depth_raw)
    gt_mask    = resize_nearest(gt_mask, out_w, out_h).astype(np.uint8)
    gt_depth01 = normalize_depth(depth_raw, mode=depth_norm_mode, global_max_mm=depth_max_mm)
    gt_depth01 = resize_nearest(gt_depth01, out_w, out_h)
    return depth_raw, gt_mask, gt_depth01


def _save_frame_outputs(
    fname,
    pred_mask_dir, pred_depth_dir, gt_mask_dir,
    subtract_dir, compare_dir, refined_overlay_dir,
    refined_pred_mask, gt_mask, refined_pred_depth,
    subtracted_depth_raw, gt_depth01,
    raw_pred_mask, raw_pred_depth,
    depth_norm_mode, depth_max_mm,
):
    """Save all outputs for one frame (runs on a background thread)."""
    save_mask_png(refined_pred_mask.astype(np.float32),  os.path.join(pred_mask_dir,  fname))
    save_mask_png(gt_mask.astype(np.float32),            os.path.join(gt_mask_dir,    fname))
    save_depth_vis(refined_pred_depth,                   os.path.join(pred_depth_dir, fname))
    save_raw_depth_png(subtracted_depth_raw,             os.path.join(subtract_dir,   fname))

    raw_overlay     = make_overlay(gt_mask, raw_pred_mask)
    refined_overlay = make_overlay(gt_mask, refined_pred_mask)
    save_color_png(refined_overlay, os.path.join(refined_overlay_dir, fname))

    gt_mask_vis            = (gt_mask * 255).astype(np.uint8)
    raw_pred_mask_vis      = (raw_pred_mask * 255).astype(np.uint8)
    refined_pred_mask_vis  = (refined_pred_mask * 255).astype(np.uint8)
    gt_depth_vis           = (np.clip(gt_depth01,         0.0, 1.0) * 255).astype(np.uint8)
    raw_pred_depth_vis     = (np.clip(raw_pred_depth,     0.0, 1.0) * 255).astype(np.uint8)
    refined_pred_depth_vis = (np.clip(refined_pred_depth, 0.0, 1.0) * 255).astype(np.uint8)

    sub_depth01   = normalize_depth(subtracted_depth_raw.astype(np.float32),
                                     mode=depth_norm_mode, global_max_mm=depth_max_mm)
    sub_depth_vis = (np.clip(sub_depth01, 0.0, 1.0) * 255).astype(np.uint8)

    top = np.concatenate([
        add_label(cv2.cvtColor(gt_mask_vis,           cv2.COLOR_GRAY2RGB), "GT Mask"),
        add_label(cv2.cvtColor(raw_pred_mask_vis,     cv2.COLOR_GRAY2RGB), "Pred Raw"),
        add_label(raw_overlay,                                             "Overlay Raw"),
        add_label(cv2.cvtColor(refined_pred_mask_vis, cv2.COLOR_GRAY2RGB), "Pred Refined"),
        add_label(refined_overlay,                                         "Overlay Refined"),
    ], axis=1)

    bottom = np.concatenate([
        add_label(cv2.cvtColor(gt_depth_vis,            cv2.COLOR_GRAY2RGB), "GT Depth"),
        add_label(cv2.cvtColor(raw_pred_depth_vis,      cv2.COLOR_GRAY2RGB), "Depth Raw"),
        add_label(cv2.cvtColor(refined_pred_depth_vis,  cv2.COLOR_GRAY2RGB), "Depth Refined"),
        add_label(cv2.cvtColor(sub_depth_vis,           cv2.COLOR_GRAY2RGB), "Subtract"),
        add_label(np.zeros_like(refined_overlay),                            "Empty"),
    ], axis=1)

    panel = np.concatenate([top, bottom], axis=0)
    Image.fromarray(panel).save(os.path.join(compare_dir, fname))


# =========================
# Main
# =========================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_mask_dir       = os.path.join(OUTPUT_DIR, "pred_masks")
    pred_depth_dir      = os.path.join(OUTPUT_DIR, "pred_depth")
    gt_mask_dir         = os.path.join(OUTPUT_DIR, "gt_masks")
    subtract_dir        = os.path.join(OUTPUT_DIR, "subtracted_depth_raw")
    compare_dir         = os.path.join(OUTPUT_DIR, "comparisons")
    refined_overlay_dir = os.path.join(OUTPUT_DIR, "refined_overlays")

    for d in [pred_mask_dir, pred_depth_dir, gt_mask_dir,
              subtract_dir, compare_dir, refined_overlay_dir]:
        os.makedirs(d, exist_ok=True)

    meta     = load_yaml_meta(META_PATH)
    captures = meta["captures"]
    if len(captures) == 0:
        raise ValueError("No captures found in meta.txt")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    selected_joints  = SELECTED_JOINTS if SELECTED_JOINTS is not None else ckpt["selected_joints"]
    use_velocities   = ckpt.get("use_velocities",   False)
    use_efforts      = ckpt.get("use_efforts",      False)
    use_t_cam_to_ee  = ckpt.get("use_t_cam_to_ee",  False)
    use_t_base_to_ee = ckpt.get("use_t_base_to_ee", False)
    input_dim        = ckpt["input_dim"]
    depth_norm_mode  = ckpt.get("depth_norm_mode",  "per_image")
    depth_max_mm     = ckpt.get("depth_max_mm",     DEPTH_NORM_FALLBACK_MAX_MM)

    out_h        = ckpt.get("out_h",        OUT_H)
    out_w        = ckpt.get("out_w",        OUT_W)
    decoder_size = ckpt.get("decoder_size", DECODER_SIZE)

    model = JointToMaskDepthNet(
        input_dim=input_dim, out_h=out_h, out_w=out_w, decoder_size=decoder_size,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint : {CHECKPOINT_PATH}")
    print(f"Output resolution : {out_w}x{out_h}")
    print(f"Decoder size      : {decoder_size}")
    print(f"Using joints      : {selected_joints}")
    print(f"Number of frames  : {len(captures)}")
    print(f"Depth norm mode   : {depth_norm_mode}")
    print(f"Subtract mode     : {SUBTRACT_MODE}")
    print(f"Refinement enabled: {ENABLE_REFINEMENT}")
    print(f"Observation mode  : {OBS_MASK_MODE}")
    print(f"Warp chunk size   : {WARP_CHUNK}")

    # -- Batched model inference -----------------------------------------
    # Build all feature vectors upfront, then run one batched forward pass
    # instead of one per frame (avoids Python loop overhead + per-frame
    # CUDA launch latency).
    INFER_BATCH = 64
    all_feats: List[np.ndarray] = []
    for cap in captures:
        feat = build_feature_vector(
            capture=cap,
            selected_joints=selected_joints,
            use_velocities=use_velocities,
            use_efforts=use_efforts,
            use_t_cam_to_ee=use_t_cam_to_ee,
            use_t_base_to_ee=use_t_base_to_ee,
        )
        if feat.shape[0] != input_dim:
            seq = int(cap.get("seq", -1))
            raise ValueError(
                f"Input dim mismatch at seq={seq}: got {feat.shape[0]}, expected {input_dim}"
            )
        all_feats.append(feat)

    pred_mask_probs_list: List[np.ndarray] = []
    pred_depth01s_list:   List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(all_feats), INFER_BATCH):
            batch = torch.from_numpy(
                np.stack(all_feats[start:start + INFER_BATCH])
            ).float().to(DEVICE)
            ml, dp = model(batch)
            pred_mask_probs_list.append(torch.sigmoid(ml)[:, 0].cpu().numpy())
            pred_depth01s_list.append(dp[:, 0].cpu().numpy())

    pred_mask_probs_all = np.concatenate(pred_mask_probs_list, axis=0)  # (N, H, W)
    pred_depth01s_all   = np.concatenate(pred_depth01s_list,   axis=0)  # (N, H, W)
    print(f"Batched inference done ({len(all_feats)} frames, batch={INFER_BATCH})")

    rows: List[FrameMetrics] = []
    save_futures = []

    load_args = [
        (cap, DATASET_ROOT, out_w, out_h, depth_norm_mode, depth_max_mm)
        for cap in captures
    ]

    # io_pool   -- prefetches depth PNGs from disk
    # save_pool -- writes output PNGs in the background so disk I/O overlaps
    #              with GPU refinement of the next frame
    with ThreadPoolExecutor(max_workers=4) as io_pool, \
         ThreadPoolExecutor(max_workers=4) as save_pool:

        futures = [io_pool.submit(_load_frame_data, a) for a in load_args]

        for idx, (cap, future) in enumerate(tqdm(zip(captures, futures), total=len(captures), desc="Refining frames")):
            loaded = future.result()
            if loaded is None:
                seq = int(cap.get("seq", -1))
                print(f"[WARN] Missing or unreadable depth for seq={seq}")
                continue

            depth_raw, gt_mask, gt_depth01 = loaded
            seq       = int(cap.get("seq", -1))
            depth_png = cap.get("depth_png", "")

            pred_mask_prob = pred_mask_probs_all[idx]
            pred_depth01   = pred_depth01s_all[idx]

            raw_pred_mask  = (pred_mask_prob > THRESHOLD).astype(np.uint8)
            raw_pred_depth = pred_depth01.copy()
            raw_pred_depth[raw_pred_mask == 0] = 0.0

            obs_mask  = build_observation_mask(
                obs_depth01=gt_depth01, raw_pred_mask=raw_pred_mask, mode=OBS_MASK_MODE,
            )
            obs_depth = gt_depth01.astype(np.float32)

            # Raw metrics
            raw_m         = compute_mask_metrics(raw_pred_mask, gt_mask)
            raw_depth_mae = compute_depth_mae(raw_pred_depth, gt_depth01, gt_mask)

            # Refinement
            refined_pred_mask  = raw_pred_mask.copy()
            refined_pred_depth = raw_pred_depth.copy()
            best_align = {
                "dx": 0.0, "dy": 0.0, "theta": 0.0, "scale": 1.0,
                "loss": 0.0, "iou": raw_m["iou"], "depth_mae": raw_depth_mae,
            }

            if ENABLE_REFINEMENT:
                candidate = refine_alignment(
                    pred_mask=raw_pred_mask.astype(np.uint8),
                    pred_depth=raw_pred_depth.astype(np.float32),
                    obs_mask=obs_mask.astype(np.uint8),
                    obs_depth=obs_depth.astype(np.float32),
                )
                raw_vs_obs_iou = compute_iou(raw_pred_mask, obs_mask)
                if candidate["iou"] >= raw_vs_obs_iou + MIN_IOU_IMPROVEMENT_TO_ACCEPT:
                    refined_pred_mask  = candidate["mask"]
                    refined_pred_depth = candidate["depth"]
                    best_align         = candidate

            # Subtraction
            depth_raw_resized = resize_nearest(depth_raw.astype(np.float32), out_w, out_h)
            if depth_raw.dtype == np.uint16:
                depth_raw_resized = np.clip(depth_raw_resized, 0, 65535).astype(np.uint16)
            else:
                depth_raw_resized = np.clip(depth_raw_resized, 0, 255).astype(np.uint8)

            subtracted_depth_raw = subtract_predicted_arm(
                original_depth_raw=depth_raw_resized,
                pred_mask01=refined_pred_mask.astype(np.float32),
                pred_depth01=refined_pred_depth,
                depth_norm_mode=depth_norm_mode,
                depth_max_mm=depth_max_mm,
            )

            # Refined metrics
            refined_m         = compute_mask_metrics(refined_pred_mask, gt_mask)
            refined_depth_mae = compute_depth_mae(refined_pred_depth, gt_depth01, gt_mask)

            rows.append(FrameMetrics(
                seq=seq, filename=depth_png,
                raw_iou=raw_m["iou"], refined_iou=refined_m["iou"],
                raw_dice=raw_m["dice"], refined_dice=refined_m["dice"],
                raw_depth_mae=raw_depth_mae, refined_depth_mae=refined_depth_mae,
                pixel_acc=refined_m["pixel_acc"],
                precision=refined_m["precision"], recall=refined_m["recall"],
                align_dx=float(best_align["dx"]), align_dy=float(best_align["dy"]),
                align_theta=float(best_align["theta"]), align_scale=float(best_align["scale"]),
                align_loss=float(best_align["loss"]),
            ))

            # Dispatch PNG/comparison writes to the background pool so saves
            # overlap with GPU refinement of the next frame.
            fname = os.path.basename(depth_png)
            save_futures.append(save_pool.submit(
                _save_frame_outputs,
                fname,
                pred_mask_dir, pred_depth_dir, gt_mask_dir,
                subtract_dir, compare_dir, refined_overlay_dir,
                refined_pred_mask.copy(), gt_mask.copy(),
                refined_pred_depth.copy(), subtracted_depth_raw.copy(),
                gt_depth01.copy(), raw_pred_mask.copy(), raw_pred_depth.copy(),
                depth_norm_mode, depth_max_mm,
            ))

            print(
                f"[seq {seq:04d}] {depth_png} | "
                f"raw IoU={raw_m['iou']:.4f} -> refined IoU={refined_m['iou']:.4f} | "
                f"raw depth MAE={raw_depth_mae:.4f} -> refined depth MAE={refined_depth_mae:.4f} | "
                f"dx={best_align['dx']:.1f} dy={best_align['dy']:.1f} "
                f"theta={best_align['theta']:.2f} scale={best_align['scale']:.3f}"
            )

        # Drain the save queue before writing CSV/summary
        for sf in save_futures:
            sf.result()

    if len(rows) == 0:
        print("No valid frames processed.")
        return

    mean_raw_iou           = float(np.mean([r.raw_iou           for r in rows]))
    mean_refined_iou       = float(np.mean([r.refined_iou       for r in rows]))
    mean_raw_dice          = float(np.mean([r.raw_dice          for r in rows]))
    mean_refined_dice      = float(np.mean([r.refined_dice      for r in rows]))
    mean_raw_depth_mae     = float(np.mean([r.raw_depth_mae     for r in rows]))
    mean_refined_depth_mae = float(np.mean([r.refined_depth_mae for r in rows]))
    mean_acc               = float(np.mean([r.pixel_acc         for r in rows]))
    mean_prec              = float(np.mean([r.precision         for r in rows]))
    mean_rec               = float(np.mean([r.recall            for r in rows]))

    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq", "filename",
            "raw_iou", "refined_iou",
            "raw_dice", "refined_dice",
            "raw_depth_mae", "refined_depth_mae",
            "pixel_acc", "precision", "recall",
            "align_dx", "align_dy", "align_theta", "align_scale", "align_loss",
        ])
        for r in rows:
            writer.writerow([
                r.seq, r.filename,
                r.raw_iou, r.refined_iou,
                r.raw_dice, r.refined_dice,
                r.raw_depth_mae, r.refined_depth_mae,
                r.pixel_acc, r.precision, r.recall,
                r.align_dx, r.align_dy, r.align_theta, r.align_scale, r.align_loss,
            ])

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Testing summary (mask + depth subtraction + refinement) -- 640x480\n")
        f.write("================================================================\n")
        f.write(f"Frames evaluated    : {len(rows)}\n")
        f.write(f"Output resolution   : {out_w}x{out_h}\n")
        f.write(f"Subtract mode       : {SUBTRACT_MODE}\n")
        f.write(f"Refinement enabled  : {ENABLE_REFINEMENT}\n")
        f.write(f"Observation mode    : {OBS_MASK_MODE}\n")
        f.write(f"Mean raw IoU        : {mean_raw_iou:.6f}\n")
        f.write(f"Mean refined IoU    : {mean_refined_iou:.6f}\n")
        f.write(f"Mean raw Dice       : {mean_raw_dice:.6f}\n")
        f.write(f"Mean refined Dice   : {mean_refined_dice:.6f}\n")
        f.write(f"Mean raw Depth MAE  : {mean_raw_depth_mae:.6f}\n")
        f.write(f"Mean refined Depth  : {mean_refined_depth_mae:.6f}\n")
        f.write(f"Mean Pixel Acc      : {mean_acc:.6f}\n")
        f.write(f"Mean Precision      : {mean_prec:.6f}\n")
        f.write(f"Mean Recall         : {mean_rec:.6f}\n")

    print("\n===== FINAL SUMMARY =====")
    print(f"Frames evaluated   : {len(rows)}")
    print(f"Output resolution  : {out_w}x{out_h}")
    print(f"Mean raw IoU       : {mean_raw_iou:.4f}")
    print(f"Mean refined IoU   : {mean_refined_iou:.4f}")
    print(f"Mean raw Dice      : {mean_raw_dice:.4f}")
    print(f"Mean refined Dice  : {mean_refined_dice:.4f}")
    print(f"Mean raw Depth MAE : {mean_raw_depth_mae:.4f}")
    print(f"Mean refined Depth : {mean_refined_depth_mae:.4f}")
    print(f"Saved CSV          : {csv_path}")
    print(f"Saved summary      : {summary_path}")
    print(f"Saved compare      : {compare_dir}")


main()

# claude --resume f5cd7c49-752c-41e7-96f7-6a69a13e0c53


'''                                                                        
  How to use:                                                               
                    
  Use the default path:                                                     
  python "snapshot/3d model/refinement.py"
  python "snapshot/3d model/filter.py"                                      
                                                                            
  Specify a custom ROOT path:                                               
  python "snapshot/3d model/refinement.py" /path/to/your/session            
  python "snapshot/3d model/filter.py" /path/to/your/session  

'''