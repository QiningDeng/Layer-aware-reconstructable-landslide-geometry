#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated Section 4.4 exporter for tabular latent features, reconstruction
evaluation, and publication-style reconstruction overlays.

This script DOES NOT call the separate latent-export scripts:
- ★3.19B策略后处理_生成表格式变量VAE.py
- ★3.19B策略后处理_生成表格式变量autodecoder.py

It directly implements:
1. VAE latent-table export using the encoder mean vector mu.
2. Auto-decoder latent-table export using stored training embeddings and/or
   sample-specific latent optimization with a fixed decoder.
3. Train/validation/test reconstruction evaluation for every detected model and
   latent dimension.
4. Stage-A-like advanced overlay figures for selected dimensions, e.g.
   VAE d=16 and Auto-decoder d=64.

Main expected input structure
-----------------------------
stageb_root/
    stageB_vae--latent_dim_1/
        checkpoints/best_model.pt
    ...
    stageB_autodecoder--latent_dim_64/
        checkpoints/best_model.pt

feature_root/
    features/*_features.npz

Typical command
---------------
python stageB_section4p4_integrated_tabular_reconstruction.py ^
    --stageb_root "D:\\...\\★3.18B步骤处理_主模型1+" ^
    --feature_root "D:\\...\\★3.18A步骤处理ALL" ^
    --output_root "D:\\...\\★4.4表格变量重建输出_integrated" ^
    --selected_vae_dim 16 ^
    --selected_autodecoder_dim 64 ^
    --splits train val test ^
    --run_latent_export ^
    --run_evaluation ^
    --run_visual_examples

If you only want to test the selected visual examples, run with:
    --only_selected_models --run_evaluation --run_visual_examples

Important interpretation note
-----------------------------
For the auto-decoder, validation/test latent vectors are obtained by optimizing
a sample-specific latent vector with the shared decoder fixed. Therefore, the
results represent inverse reconstruction after latent fitting, rather than
one-pass encoder-based inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Runtime safeguards for Windows / Anaconda mixed-library environments.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not enabled here because
# it is not supported on some Windows/PyTorch builds and only produces warnings.

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Publication style
# =============================================================================

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

RUN_NAME_PREFIX = {
    "vae": "stageB_vae--latent_dim_",
    "autodecoder": "stageB_autodecoder--latent_dim_",
}

MODEL_ORDER = ["vae", "autodecoder"]
SPLIT_ORDER = ["train", "val", "test"]

METRIC_COLUMNS = [
    "mask_iou",
    "mask_dice",
    "area_relative_error",
    "centroid_distance_normalized",
    "boundary_chamfer_normalized",
    "boundary_hausdorff_normalized",
    "polygon_count_abs_diff",
]

FALLBACK_LAYER_COLORS = {
    "top": (0.82, 0.00, 0.00),
    "middle": (0.00, 0.65, 0.00),
    "bottom": (0.00, 0.00, 0.72),
}

# Color names used in the Stage A polygon / feature database. These are not
# top-middle-bottom defaults. They follow the layer color identity stored in
# ★3.18A步骤处理ALL, e.g., layer_names = ["red", "green", "blue"] or
# ["red", "yellow", "blue"].
DATABASE_COLOR_NAME_RGB = {
    "red": (1.0, 0.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}

DEFAULT_REPRESENTATIVE_SAMPLE_NAMES = ["59", "358", "573", "768", "997", "1173"]

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def build_image_index(root: str | Path | None) -> Dict[str, Path]:
    """Recursively index original database images by sample-name stem."""
    index: Dict[str, Path] = {}
    if root is None:
        return index
    root_path = Path(root)
    if not root_path.exists():
        return index
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            stem = infer_sample_name(p)
            index.setdefault(stem, p)
    return index


def polygon_json_sample_name(path: str | Path) -> str:
    """Infer sample name from a *_layer_polygons.json file."""
    stem = Path(path).stem
    if stem.endswith("_layer_polygons"):
        stem = stem[: -len("_layer_polygons")]
    return stem


def build_polygon_json_index(root: str | Path | None) -> Dict[str, Path]:
    """Recursively index Stage A boundary-extraction JSON files by sample name."""
    index: Dict[str, Path] = {}
    if root is None:
        return index
    root_path = Path(root)
    if not root_path.exists():
        return index
    for p in root_path.rglob("*_layer_polygons.json"):
        if p.is_file():
            index.setdefault(polygon_json_sample_name(p), p)
    return index


def read_polygon_json_color_identity(
    json_path: str | Path,
) -> Tuple[List[str], List[str]]:
    """Read layer color names and vertical ranks from a *_layer_polygons.json file.

    Supports both list-style and dict-style layer records.
    Returns:
        layer_names: color names such as red / yellow / green / blue
        vertical_ranks: top / middle / bottom
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_layers = data.get("layers", [])
    parsed: List[Dict[str, Any]] = []

    if isinstance(raw_layers, list):
        for item in raw_layers:
            if not isinstance(item, dict):
                continue
            parsed.append({
                "color_name": str(item.get("color_name", item.get("layer_name", ""))).lower(),
                "vertical_rank": str(item.get("vertical_rank", "")).lower(),
            })
    elif isinstance(raw_layers, dict):
        for key, item in raw_layers.items():
            if not isinstance(item, dict):
                continue
            parsed.append({
                "color_name": str(item.get("color_name", item.get("layer_name", key))).lower(),
                "vertical_rank": str(item.get("vertical_rank", key)).lower(),
            })

    rank_order = {"top": 0, "middle": 1, "bottom": 2}
    if any(p.get("vertical_rank") in rank_order for p in parsed):
        parsed.sort(key=lambda p: rank_order.get(p.get("vertical_rank", ""), 999))

    layer_names: List[str] = []
    vertical_ranks: List[str] = []
    for item in parsed:
        color_name = item.get("color_name", "").lower()
        vertical_rank = item.get("vertical_rank", "").lower()
        if color_name:
            layer_names.append(color_name)
        if vertical_rank:
            vertical_ranks.append(vertical_rank)

    # Fall back to detected_color_order_top_to_bottom if per-layer records are incomplete.
    if len(layer_names) < 3:
        detected = data.get("detected_color_order_top_to_bottom", [])
        if isinstance(detected, list) and len(detected) >= 3:
            layer_names = [str(v).lower() for v in detected[:3]]

    if len(vertical_ranks) < 3:
        vertical_ranks = ["top", "middle", "bottom"][:len(layer_names)]

    return layer_names, vertical_ranks


def resolve_polygon_json_path(
    sample_name: str,
    polygon_json_root: str | Path | None = None,
    polygon_json_index: Optional[Dict[str, Path]] = None,
    feature_file: str | Path | None = None,
) -> Optional[Path]:
    """Resolve a sample's *_layer_polygons.json file."""
    sample_name = str(sample_name)
    if polygon_json_index is not None and sample_name in polygon_json_index:
        return polygon_json_index[sample_name]

    if polygon_json_root:
        root = Path(polygon_json_root)
        if root.exists():
            direct = root / f"{sample_name}_layer_polygons.json"
            if direct.is_file():
                return direct

    if feature_file is not None:
        feature_path = Path(feature_file)
        search_dirs = [
            feature_path.parent,
            feature_path.parent.parent,
            feature_path.parent.parent / "metadata",
            feature_path.parent.parent / "polygons",
            feature_path.parent.parent / "selfcheck_polygons",
            feature_path.parent.parent.parent / "★3.18边界提取ALL",
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            direct = d / f"{sample_name}_layer_polygons.json"
            if direct.is_file():
                return direct
    return None


def get_stageA_color_identity(
    sample_name: str,
    layer_names: Optional[Sequence[str]] = None,
    vertical_ranks: Optional[Sequence[str]] = None,
    feature_file: str | Path | None = None,
    polygon_json_root: str | Path | None = None,
    polygon_json_index: Optional[Dict[str, Path]] = None,
) -> Tuple[List[str], List[str], str]:
    """Resolve layer color identity for visualization.

    Priority:
    1. color names already stored in Stage A feature npz as layer_names
    2. *_layer_polygons.json color_name and vertical_rank
    3. fallback top/middle/bottom
    """
    ln = [str(v).lower() for v in layer_names] if layer_names is not None else []
    vr = [str(v).lower() for v in vertical_ranks] if vertical_ranks is not None else []

    if len(ln) >= 3 and any(v in DATABASE_COLOR_NAME_RGB for v in ln):
        if len(vr) < 3:
            vr = ["top", "middle", "bottom"]
        return ln[:3], vr[:3], "feature_npz_layer_names"

    json_path = resolve_polygon_json_path(
        sample_name=sample_name,
        polygon_json_root=polygon_json_root,
        polygon_json_index=polygon_json_index,
        feature_file=feature_file,
    )
    if json_path is not None:
        json_ln, json_vr = read_polygon_json_color_identity(json_path)
        if len(json_ln) >= 3:
            if len(json_vr) < 3:
                json_vr = ["top", "middle", "bottom"]
            return json_ln[:3], json_vr[:3], str(json_path)

    if len(ln) >= 3:
        if len(vr) < 3:
            vr = ["top", "middle", "bottom"]
        return ln[:3], vr[:3], "feature_npz_layer_names_noncolor"

    return ["top", "middle", "bottom"], ["top", "middle", "bottom"], "fallback_default"


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load Stage B checkpoints containing full Python/NumPy objects."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_sample_name(path: str | Path) -> str:
    name = Path(str(path)).stem
    if name.endswith("_features"):
        name = name[: -len("_features")]
    return name


def discover_stagea_feature_files(feature_root: str | Path) -> List[Path]:
    root = Path(feature_root)
    feature_dir = root / "features"
    if feature_dir.is_dir():
        files = sorted(feature_dir.glob("*_features.npz"))
        if files:
            return files
    return sorted(root.glob("*_features.npz"))


def discover_runs(stageb_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(stageb_root)
    rows: List[Dict[str, Any]] = []
    for model_type, prefix in RUN_NAME_PREFIX.items():
        for run_dir in sorted(root.glob(f"{prefix}*")):
            if not run_dir.is_dir():
                continue
            try:
                latent_dim = int(run_dir.name.split(prefix, 1)[1])
            except Exception:
                continue
            ckpt = run_dir / "checkpoints" / "best_model.pt"
            if not ckpt.is_file():
                continue
            rows.append({
                "model_type": model_type,
                "latent_dim": latent_dim,
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt),
            })
    rows.sort(key=lambda r: (0 if r["model_type"] == "vae" else 1, int(r["latent_dim"])))
    return rows


def filter_selected_runs(
    rows: Sequence[Dict[str, Any]],
    selected_vae_dim: int,
    selected_autodecoder_dim: int,
    only_selected_models: bool,
) -> List[Dict[str, Any]]:
    if not only_selected_models:
        return list(rows)
    selected = []
    for row in rows:
        if row["model_type"] == "vae" and int(row["latent_dim"]) == int(selected_vae_dim):
            selected.append(dict(row))
        elif row["model_type"] == "autodecoder" and int(row["latent_dim"]) == int(selected_autodecoder_dim):
            selected.append(dict(row))
    return selected


def split_files_from_checkpoint(checkpoint: str | Path) -> Dict[str, List[str]]:
    ckpt = safe_torch_load(checkpoint, map_location="cpu")
    split_dict = ckpt.get("split_dict", {})
    if not split_dict:
        return {}
    return {str(k): list(v) for k, v in split_dict.items()}


def split_map_from_checkpoint(checkpoint: str | Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    split_dict = split_files_from_checkpoint(checkpoint)
    for split_name, files in split_dict.items():
        for f in files:
            mapping[infer_sample_name(f)] = str(split_name)
    return mapping


def file_map_by_sample(feature_files: Sequence[Path]) -> Dict[str, Path]:
    return {infer_sample_name(f): f for f in feature_files}


# =============================================================================
# Dataset
# =============================================================================

class StageAFeatureDataset(Dataset):
    """Dataset wrapper for Stage A feature files."""

    def __init__(self, feature_files: Sequence[str | Path], encoder_input_size: int = 128) -> None:
        self.feature_files = [Path(f) for f in feature_files]
        self.encoder_input_size = int(encoder_input_size)

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.feature_files[index]
        with np.load(path, allow_pickle=True) as data:
            feature_tensor = data["feature_tensor"].astype(np.float32)
            sdf = data["sdf_normalized"].astype(np.float32)
            corner = data["corner_field"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
            source_image = (
                str(data["source_image"].tolist())
                if "source_image" in data
                else infer_sample_name(path)
            )
            layer_names = (
                [str(v) for v in data["layer_names"].tolist()]
                if "layer_names" in data
                else ["top", "middle", "bottom"]
            )
            vertical_ranks = (
                [str(v) for v in data["vertical_ranks"].tolist()]
                if "vertical_ranks" in data
                else ["top", "middle", "bottom"]
            )

        feature_small = cv2.resize(
            np.transpose(feature_tensor, (1, 2, 0)),
            (self.encoder_input_size, self.encoder_input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        feature_small = np.transpose(feature_small, (2, 0, 1)).astype(np.float32)

        return {
            "sample_name": infer_sample_name(path),
            "feature_file": str(path),
            "source_image": source_image,
            "feature_small": feature_small,
            "sdf": sdf,
            "corner": corner,
            "mask": mask,
            "layer_names": layer_names,
            "vertical_ranks": vertical_ranks,
        }


def collate_stageA(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    feature_small = torch.from_numpy(np.stack([b["feature_small"] for b in batch], axis=0))
    sdf = torch.from_numpy(np.stack([b["sdf"] for b in batch], axis=0))
    corner = torch.from_numpy(np.stack([b["corner"] for b in batch], axis=0))
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch], axis=0))
    return {
        "sample_name": [b["sample_name"] for b in batch],
        "feature_file": [b["feature_file"] for b in batch],
        "source_image": [b["source_image"] for b in batch],
        "feature_small": feature_small,
        "sdf": sdf,
        "corner": corner,
        "mask": mask,
        "layer_names": [b["layer_names"] for b in batch],
        "vertical_ranks": [b["vertical_ranks"] for b in batch],
    }


def dataloader_for_files(
    feature_files: Sequence[str | Path],
    encoder_input_size: int,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = StageAFeatureDataset(feature_files, encoder_input_size=encoder_input_size)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_stageA,
        pin_memory=torch.cuda.is_available(),
    )


# =============================================================================
# Model definitions matching Stage B
# =============================================================================

class ConvEncoder(nn.Module):
    """Convolutional encoder for 9-channel Stage A feature tensors."""

    def __init__(self, in_channels: int = 9, latent_dim: int = 64) -> None:
        super().__init__()
        channels = [32, 64, 128, 256, 256]
        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(channels[-1], latent_dim)
        self.fc_logvar = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class FourierFeatureEncoder(nn.Module):
    def __init__(self, num_levels: int = 8) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        freqs = 2.0 ** torch.arange(self.num_levels, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def output_dim(self) -> int:
        return 2 + 4 * self.num_levels

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        emb = [coords]
        x = coords[..., None] * self.freqs[None, None, None, :] * math.pi
        emb.append(torch.sin(x).reshape(coords.shape[0], coords.shape[1], -1))
        emb.append(torch.cos(x).reshape(coords.shape[0], coords.shape[1], -1))
        return torch.cat(emb, dim=-1)


class JointImplicitDecoder(nn.Module):
    """Continuous implicit decoder for 3 SDF, 3 corner, and 3 mask channels."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 6,
        fourier_levels: int = 8,
    ) -> None:
        super().__init__()
        self.coord_encoder = FourierFeatureEncoder(fourier_levels)
        input_dim = self.coord_encoder.output_dim + int(latent_dim)

        layers: List[nn.Module] = []
        d_in = input_dim
        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(d_in, int(hidden_dim)))
            layers.append(nn.SiLU(inplace=True))
            d_in = int(hidden_dim)
        layers.append(nn.Linear(d_in, 9))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, npts, _ = coords.shape
        coord_feat = self.coord_encoder(coords)
        z_expand = z[:, None, :].expand(-1, npts, -1)
        y = self.net(torch.cat([coord_feat, z_expand], dim=-1))
        return {
            "sdf": y[..., 0:3],
            "corner_logits": y[..., 3:6],
            "mask_logits": y[..., 6:9],
        }


def build_coordinate_grid(height: int, width: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=int(height), device=device)
    xs = torch.linspace(-1.0, 1.0, steps=int(width), device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def build_encoder_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> ConvEncoder:
    cfg = ckpt.get("config", {})
    latent_dim = int(cfg.get("latent_dim", ckpt.get("latent_dim", 64)))
    encoder = ConvEncoder(in_channels=9, latent_dim=latent_dim).to(device)
    state_dict = ckpt.get("encoder_state_dict", None)
    if state_dict is None:
        raise ValueError("Checkpoint does not contain encoder_state_dict.")
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def build_decoder_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> JointImplicitDecoder:
    cfg = ckpt.get("config", {})
    latent_dim = int(cfg.get("latent_dim", ckpt.get("latent_dim", 64)))
    decoder = JointImplicitDecoder(
        latent_dim=latent_dim,
        hidden_dim=int(cfg.get("decoder_hidden_dim", 256)),
        num_layers=int(cfg.get("decoder_num_layers", 6)),
        fourier_levels=int(cfg.get("decoder_fourier_levels", 8)),
    ).to(device)
    state_dict = ckpt.get("decoder_state_dict", None)
    if state_dict is None:
        raise ValueError("Checkpoint does not contain decoder_state_dict.")
    decoder.load_state_dict(state_dict)
    decoder.eval()
    return decoder


# =============================================================================
# Losses for auto-decoder latent optimization
# =============================================================================

def sample_random_points(
    sdf: torch.Tensor,
    corner: torch.Tensor,
    mask: torch.Tensor,
    coords_flat: torch.Tensor,
    points_per_sample: int,
) -> Dict[str, torch.Tensor]:
    bsz, _, h, w = sdf.shape
    hw = h * w
    device = sdf.device
    idx = torch.randint(0, hw, (bsz, int(points_per_sample)), device=device)

    sdf_flat = sdf.view(bsz, 3, hw).permute(0, 2, 1)
    corner_flat = corner.view(bsz, 3, hw).permute(0, 2, 1)
    mask_flat = mask.view(bsz, 3, hw).permute(0, 2, 1)

    idx_expand = idx.unsqueeze(-1).expand(-1, -1, 3)
    gt_sdf = torch.gather(sdf_flat, 1, idx_expand)
    gt_corner = torch.gather(corner_flat, 1, idx_expand)
    gt_mask = torch.gather(mask_flat, 1, idx_expand)

    coords = coords_flat[idx]
    return {"coords": coords, "sdf": gt_sdf, "corner": gt_corner, "mask": gt_mask}


def weighted_sdf_loss(pred_sdf: torch.Tensor, gt_sdf: torch.Tensor, delta: float, boost: float) -> torch.Tensor:
    weight = 1.0 + float(boost) * torch.exp(-torch.abs(gt_sdf) / max(float(delta), 1e-6))
    return torch.mean(weight * F.smooth_l1_loss(pred_sdf, gt_sdf, reduction="none"))


def reconstruction_loss(
    outputs: Dict[str, torch.Tensor],
    gt_sdf: torch.Tensor,
    gt_corner: torch.Tensor,
    gt_mask: torch.Tensor,
    latent: Optional[torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    loss_sdf = weighted_sdf_loss(
        outputs["sdf"],
        gt_sdf,
        float(cfg.get("boundary_delta", 0.10)),
        float(cfg.get("boundary_boost", 4.0)),
    )
    loss_corner = F.mse_loss(torch.sigmoid(outputs["corner_logits"]), gt_corner)
    loss_mask = F.binary_cross_entropy_with_logits(outputs["mask_logits"], gt_mask)

    p = torch.sigmoid(outputs["mask_logits"])
    loss_sep = (p[..., 0] * p[..., 1] + p[..., 0] * p[..., 2] + p[..., 1] * p[..., 2]).mean()

    sdf_prob = torch.sigmoid(-outputs["sdf"] / max(float(cfg.get("mask_temperature", 0.08)), 1e-6))
    mask_prob = torch.sigmoid(outputs["mask_logits"])
    loss_cons = F.mse_loss(mask_prob, sdf_prob)

    if latent is None:
        loss_latent = torch.tensor(0.0, device=gt_sdf.device)
    else:
        loss_latent = torch.mean(latent ** 2)

    total = (
        float(cfg.get("weight_sdf", 1.0)) * loss_sdf
        + float(cfg.get("weight_corner", 0.2)) * loss_corner
        + float(cfg.get("weight_mask", 0.2)) * loss_mask
        + float(cfg.get("weight_sep", 0.05)) * loss_sep
        + float(cfg.get("weight_consistency", 0.1)) * loss_cons
        + float(cfg.get("weight_latent_reg", 1e-4)) * loss_latent
    )
    return {
        "total": total,
        "sdf": loss_sdf,
        "corner": loss_corner,
        "mask": loss_mask,
        "sep": loss_sep,
        "consistency": loss_cons,
        "latent_reg": loss_latent,
    }


def optimize_latent_for_sample(
    decoder: JointImplicitDecoder,
    sdf: torch.Tensor,
    corner: torch.Tensor,
    mask: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    optimization_steps: Optional[int] = None,
    optimization_lr: Optional[float] = None,
    points_per_sample: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    latent_dim = int(cfg.get("latent_dim", 64))
    steps = int(optimization_steps if optimization_steps is not None else cfg.get("eval_latent_optimization_steps", cfg.get("latent_optimization_steps", 200)))
    lr = float(optimization_lr if optimization_lr is not None else cfg.get("eval_latent_optimization_lr", cfg.get("latent_learning_rate", 5e-3)))
    pts = int(points_per_sample if points_per_sample is not None else cfg.get("points_per_sample", 4096))

    decoder.eval()
    sdf = sdf.to(device=device, dtype=torch.float32)
    corner = corner.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.float32)

    h, w = sdf.shape[-2:]
    coords_flat = build_coordinate_grid(h, w, device=device).view(-1, 2)

    latent = torch.zeros((1, latent_dim), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    final_loss = float("nan")

    t0 = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        sampled = sample_random_points(sdf, corner, mask, coords_flat, pts)
        outputs = decoder(latent, sampled["coords"])
        loss_dict = reconstruction_loss(outputs, sampled["sdf"], sampled["corner"], sampled["mask"], latent, cfg)
        # Only latent is optimized. This avoids accumulating decoder gradients.
        latent_grad = torch.autograd.grad(
            loss_dict["total"],
            latent,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]
        latent.grad = latent_grad
        optimizer.step()
        final_loss = float(loss_dict["total"].detach().cpu())
        del sampled, outputs, loss_dict, latent_grad

    elapsed = time.time() - t0
    return latent.detach(), {
        "latent_source": "optimized",
        "latent_optimization_steps": int(steps),
        "latent_optimization_time_seconds": float(elapsed),
        "latent_optimization_final_loss": float(final_loss),
    }


# =============================================================================
# Dense reconstruction and metrics
# =============================================================================

def stable_sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for dense decoder logits.

    Large positive/negative logits are expected for confident mask/corner
    predictions. Clipping avoids harmless NumPy overflow warnings while leaving
    the resulting probabilities practically unchanged.
    """
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


@torch.no_grad()
def predict_dense_fields(
    decoder: JointImplicitDecoder,
    latent: torch.Tensor,
    grid_size: int,
    device: torch.device,
    chunk_size: int = 65536,
) -> Dict[str, np.ndarray]:
    decoder.eval()
    coords = build_coordinate_grid(grid_size, grid_size, device=device).view(-1, 2)
    latent = latent.to(device=device, dtype=torch.float32)
    chunks = []
    for start in range(0, coords.shape[0], int(chunk_size)):
        part = coords[start:start + int(chunk_size)][None, ...]
        outputs = decoder(latent, part)
        chunks.append(torch.cat([
            outputs["sdf"],
            outputs["corner_logits"],
            outputs["mask_logits"],
        ], dim=-1).detach().cpu())
    dense = torch.cat(chunks, dim=1).view(grid_size, grid_size, 9).numpy()
    sdf = dense[..., 0:3].transpose(2, 0, 1).astype(np.float32)
    corner_prob = stable_sigmoid_numpy(dense[..., 3:6].transpose(2, 0, 1))
    mask_prob = stable_sigmoid_numpy(dense[..., 6:9].transpose(2, 0, 1))
    pred_mask = (mask_prob >= 0.5).astype(np.uint8)
    return {
        "sdf": sdf,
        "corner_prob": corner_prob.astype(np.float32),
        "mask_prob": mask_prob.astype(np.float32),
        "pred_mask": pred_mask,
    }


def resize_mask_stack(mask: np.ndarray, target_size: int) -> np.ndarray:
    mask = mask.astype(np.float32)
    tensor = torch.from_numpy(mask[None, ...])
    out = F.interpolate(tensor, size=(target_size, target_size), mode="nearest")
    return out.squeeze(0).numpy().astype(np.uint8)


def boundary_mask_from_binary(mask: np.ndarray) -> np.ndarray:
    u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(u8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 255, 1)
    return (boundary > 0).astype(np.uint8)


def contour_points_from_mask(mask: np.ndarray, min_area: float = 3.0) -> List[np.ndarray]:
    u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for cnt in contours:
        if abs(cv2.contourArea(cnt)) >= float(min_area):
            polys.append(cnt[:, 0, :].astype(np.float32))
    return polys


def centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter / union) if union > 0 else 1.0


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    denom = (mask_a > 0).sum() + (mask_b > 0).sum()
    return float(2.0 * inter / denom) if denom > 0 else 1.0


def compute_area_relative_error(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    area_a = float((mask_a > 0).sum())
    area_b = float((mask_b > 0).sum())
    if area_a <= 1e-6 and area_b <= 1e-6:
        return 0.0
    return float(abs(area_b - area_a) / max(area_a, 1.0))


def compute_centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray, diagonal: float) -> float:
    c1 = centroid_from_mask(mask_a)
    c2 = centroid_from_mask(mask_b)
    if c1 is None and c2 is None:
        return 0.0
    if c1 is None or c2 is None:
        return 1.0
    return float(math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / max(diagonal, 1e-6))


def compute_boundary_distances(mask_a: np.ndarray, mask_b: np.ndarray, diagonal: float) -> Tuple[float, float]:
    b1 = boundary_mask_from_binary(mask_a)
    b2 = boundary_mask_from_binary(mask_b)
    n1, n2 = int(b1.sum()), int(b2.sum())
    if n1 == 0 and n2 == 0:
        return 0.0, 0.0
    if n1 == 0 or n2 == 0:
        return 1.0, 1.0
    dt_to_b2 = cv2.distanceTransform((1 - b2).astype(np.uint8), cv2.DIST_L2, 5)
    dt_to_b1 = cv2.distanceTransform((1 - b1).astype(np.uint8), cv2.DIST_L2, 5)
    d12 = dt_to_b2[b1 > 0]
    d21 = dt_to_b1[b2 > 0]
    chamfer = 0.5 * (float(d12.mean()) + float(d21.mean())) / max(diagonal, 1e-6)
    hausdorff = max(float(d12.max()), float(d21.max())) / max(diagonal, 1e-6)
    return float(chamfer), float(hausdorff)


def evaluate_sample_masks(gt_mask: np.ndarray, pred_mask: np.ndarray, layer_names: Sequence[str], vertical_ranks: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    assert gt_mask.shape[0] == 3 and pred_mask.shape[0] == 3
    h, w = gt_mask.shape[-2:]
    diagonal = math.sqrt(h * h + w * w)
    rows: List[Dict[str, Any]] = []

    for i in range(3):
        chamfer, hausdorff = compute_boundary_distances(gt_mask[i], pred_mask[i], diagonal)
        ref_polys = contour_points_from_mask(gt_mask[i])
        pred_polys = contour_points_from_mask(pred_mask[i])
        rows.append({
            "layer_index": i,
            "layer_name": str(layer_names[i]) if i < len(layer_names) else ["top", "middle", "bottom"][i],
            "vertical_rank": str(vertical_ranks[i]) if i < len(vertical_ranks) else ["top", "middle", "bottom"][i],
            "mask_iou": compute_iou(gt_mask[i], pred_mask[i]),
            "mask_dice": compute_dice(gt_mask[i], pred_mask[i]),
            "area_relative_error": compute_area_relative_error(gt_mask[i], pred_mask[i]),
            "centroid_distance_normalized": compute_centroid_distance(gt_mask[i], pred_mask[i], diagonal),
            "boundary_chamfer_normalized": chamfer,
            "boundary_hausdorff_normalized": hausdorff,
            "polygon_count_abs_diff": abs(len(pred_polys) - len(ref_polys)),
        })

    summary = {
        key: float(np.mean([r[key] for r in rows]))
        for key in METRIC_COLUMNS
    }
    return rows, summary


def summarize_numeric_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        else:
            out[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return out


# =============================================================================
# Latent sources and reconstruction engines
# =============================================================================

def get_train_embedding_from_checkpoint(ckpt: Dict[str, Any], sample_name: str, split_mapping: Dict[str, str]) -> Optional[torch.Tensor]:
    if split_mapping.get(sample_name) != "train":
        return None

    split_dict = ckpt.get("split_dict", {})
    train_files = list(split_dict.get("train", []))
    train_names = [infer_sample_name(f) for f in train_files]
    if sample_name not in train_names:
        return None
    idx = train_names.index(sample_name)

    # Preferred direct NumPy copy saved by training script.
    if "latent_embeddings" in ckpt:
        arr = np.asarray(ckpt["latent_embeddings"], dtype=np.float32)
        if arr.ndim == 2 and idx < arr.shape[0]:
            return torch.from_numpy(arr[idx:idx + 1].copy())

    # Alternative embedding state dict.
    emb_state = ckpt.get("embeddings_state_dict", None)
    if isinstance(emb_state, dict):
        for key in ["weight", "embeddings.weight"]:
            if key in emb_state:
                weight = emb_state[key]
                if isinstance(weight, torch.Tensor) and idx < weight.shape[0]:
                    return weight[idx:idx + 1].detach().cpu()

    return None


@torch.no_grad()
def vae_latent_from_batch(encoder: ConvEncoder, feature_small: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = feature_small.to(device=device, dtype=torch.float32, non_blocking=True)
    mu, logvar = encoder(x)
    return mu.detach(), logvar.detach()


# =============================================================================
# Export latent tables and reconstruction evaluation
# =============================================================================

def export_and_evaluate_run(
    row: Dict[str, Any],
    feature_root: Path,
    output_root: Path,
    splits: Sequence[str],
    run_latent_export: bool,
    run_evaluation: bool,
    run_predictions: bool,
    optimize_all_autodecoder: bool,
    auto_optimization_steps: Optional[int],
    auto_optimization_lr: Optional[float],
    batch_size_vae: int,
    device: torch.device,
    eval_grid_size_override: Optional[int] = None,
    eval_chunk_size: int = 65536,
) -> Dict[str, Any]:
    model_type = row["model_type"]
    latent_dim = int(row["latent_dim"])
    checkpoint = Path(row["checkpoint"])
    run_dir = Path(row["run_dir"])
    ckpt = safe_torch_load(checkpoint, map_location=device)
    cfg = dict(ckpt.get("config", {}))
    cfg["latent_dim"] = int(cfg.get("latent_dim", latent_dim))
    encoder_input_size = int(cfg.get("encoder_input_size", 128))
    eval_grid_size = int(eval_grid_size_override or cfg.get("eval_grid_size", 256))

    split_dict = split_files_from_checkpoint(checkpoint)
    split_mapping = split_map_from_checkpoint(checkpoint)

    # Fallback if split_dict is not present.
    all_feature_files = discover_stagea_feature_files(feature_root)
    feature_by_name = file_map_by_sample(all_feature_files)

    if split_dict:
        files_by_split: Dict[str, List[Path]] = {}
        for split in splits:
            files_by_split[split] = [
                feature_by_name[infer_sample_name(f)]
                for f in split_dict.get(split, [])
                if infer_sample_name(f) in feature_by_name
            ]
    else:
        files_by_split = {split: list(all_feature_files) if split == "all" else [] for split in splits}

    decoder = build_decoder_from_checkpoint(ckpt, device)

    if model_type == "vae":
        encoder = build_encoder_from_checkpoint(ckpt, device)
    else:
        encoder = None

    latent_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    per_layer_rows: List[Dict[str, Any]] = []

    run_output_root = ensure_dir(output_root / "runs" / model_type / f"latent_dim_{latent_dim}")
    prediction_root = ensure_dir(run_output_root / "predictions")

    for split in splits:
        feature_files = files_by_split.get(split, [])
        if not feature_files:
            print(f"[Warning] No feature files for {model_type}, d={latent_dim}, split={split}")
            continue

        batch_size = batch_size_vae if model_type == "vae" else 1
        loader = dataloader_for_files(
            feature_files,
            encoder_input_size=encoder_input_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        for batch in loader:
            bsz = len(batch["sample_name"])

            if model_type == "vae":
                assert encoder is not None
                mu, logvar = vae_latent_from_batch(encoder, batch["feature_small"], device)
                latents = mu
                logvars = logvar
                latent_infos = [
                    {
                        "latent_source": "encoder_mu",
                        "latent_optimization_steps": 0,
                        "latent_optimization_time_seconds": 0.0,
                        "latent_optimization_final_loss": float("nan"),
                    }
                    for _ in range(bsz)
                ]
            else:
                latents_list = []
                latent_infos = []
                logvars = None
                for i in range(bsz):
                    sample_name = batch["sample_name"][i]
                    stored = None if optimize_all_autodecoder else get_train_embedding_from_checkpoint(ckpt, sample_name, split_mapping)
                    if stored is not None:
                        latent = stored.to(device=device, dtype=torch.float32)
                        info = {
                            "latent_source": "stored_train_embedding",
                            "latent_optimization_steps": 0,
                            "latent_optimization_time_seconds": 0.0,
                            "latent_optimization_final_loss": float("nan"),
                        }
                    else:
                        latent, info = optimize_latent_for_sample(
                            decoder,
                            batch["sdf"][i:i + 1],
                            batch["corner"][i:i + 1],
                            batch["mask"][i:i + 1],
                            cfg,
                            device,
                            optimization_steps=auto_optimization_steps,
                            optimization_lr=auto_optimization_lr,
                        )
                    latents_list.append(latent)
                    latent_infos.append(info)
                latents = torch.cat(latents_list, dim=0)

            for i in range(bsz):
                sample_name = batch["sample_name"][i]
                latent = latents[i:i + 1]

                if run_latent_export:
                    row_latent: Dict[str, Any] = {
                        "sample_name": sample_name,
                        "split": split,
                        "source_image": batch["source_image"][i],
                        "feature_file": batch["feature_file"][i],
                        "model_type": model_type,
                        "latent_dim": latent_dim,
                    }
                    row_latent.update(latent_infos[i])
                    z_np = latent.detach().cpu().numpy()[0]
                    for j, value in enumerate(z_np):
                        row_latent[f"z_{j:03d}"] = float(value)
                    if model_type == "vae" and logvars is not None:
                        lv_np = logvars[i].detach().cpu().numpy()
                        for j, value in enumerate(lv_np):
                            row_latent[f"logvar_{j:03d}"] = float(value)
                    latent_rows.append(row_latent)

                if run_evaluation or run_predictions:
                    t0 = time.time()
                    dense = predict_dense_fields(
                        decoder,
                        latent,
                        eval_grid_size,
                        device,
                        chunk_size=int(eval_chunk_size),
                    )
                    reconstruction_time = time.time() - t0

                    gt_mask = batch["mask"][i].numpy().astype(np.uint8)
                    if gt_mask.shape[-1] != eval_grid_size or gt_mask.shape[-2] != eval_grid_size:
                        gt_mask = resize_mask_stack(gt_mask, eval_grid_size)

                    layer_rows, sample_summary = evaluate_sample_masks(
                        gt_mask,
                        dense["pred_mask"],
                        batch["layer_names"][i],
                        batch["vertical_ranks"][i],
                    )

                    sample_record: Dict[str, Any] = {
                        "sample_name": sample_name,
                        "split": split,
                        "model_type": model_type,
                        "latent_dim": latent_dim,
                        "source_image": batch["source_image"][i],
                        "feature_file": batch["feature_file"][i],
                        "reconstruction_time_seconds": float(reconstruction_time),
                    }
                    sample_record.update(latent_infos[i])
                    sample_record.update(sample_summary)
                    per_sample_rows.append(sample_record)

                    for layer_row in layer_rows:
                        layer_record = {
                            "sample_name": sample_name,
                            "split": split,
                            "model_type": model_type,
                            "latent_dim": latent_dim,
                        }
                        layer_record.update(layer_row)
                        per_layer_rows.append(layer_record)

                    if run_predictions:
                        pred_dir = ensure_dir(prediction_root / split / sample_name)
                        np.savez_compressed(
                            pred_dir / f"{sample_name}_predicted_fields.npz",
                            sdf=dense["sdf"],
                            corner_prob=dense["corner_prob"],
                            mask_prob=dense["mask_prob"],
                            pred_mask=dense["pred_mask"],
                            gt_mask=gt_mask,
                            source_image=batch["source_image"][i],
                            feature_file=batch["feature_file"][i],
                            layer_names=np.array(batch["layer_names"][i], dtype=object),
                            vertical_ranks=np.array(batch["vertical_ranks"][i], dtype=object),
                            sample_name=sample_name,
                            split=split,
                            model_type=model_type,
                            latent_dim=latent_dim,
                        )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Export latent tables.
    if run_latent_export and latent_rows:
        latent_headers = sorted({k for r in latent_rows for k in r.keys() if k.startswith("z_")})
        logvar_headers = sorted({k for r in latent_rows for k in r.keys() if k.startswith("logvar_")})
        base_headers = [
            "sample_name", "split", "source_image", "feature_file",
            "model_type", "latent_dim", "latent_source",
            "latent_optimization_steps", "latent_optimization_time_seconds",
            "latent_optimization_final_loss",
        ]
        fieldnames = base_headers + latent_headers + logvar_headers
        save_csv(
            output_root / "latent_tables_split_aware" / model_type / f"{model_type}_latent_dim_{latent_dim}_with_split.csv",
            latent_rows,
            fieldnames,
        )

    # Export evaluation tables.
    if run_evaluation and per_sample_rows:
        sample_fieldnames = [
            "sample_name", "split", "model_type", "latent_dim", "source_image", "feature_file",
            "latent_source", "latent_optimization_steps", "latent_optimization_time_seconds",
            "latent_optimization_final_loss", "reconstruction_time_seconds",
        ] + METRIC_COLUMNS
        save_csv(run_output_root / "evaluation_per_sample.csv", per_sample_rows, sample_fieldnames)

        layer_fieldnames = ["sample_name", "split", "model_type", "latent_dim", "layer_index", "layer_name", "vertical_rank"] + METRIC_COLUMNS
        save_csv(run_output_root / "evaluation_per_layer.csv", per_layer_rows, layer_fieldnames)

        summary_records = []
        for split in splits:
            subset = [r for r in per_sample_rows if r["split"] == split]
            if not subset:
                continue
            rec: Dict[str, Any] = {
                "model_type": model_type,
                "latent_dim": latent_dim,
                "split": split,
                "sample_count": len(subset),
            }
            metric_summary = summarize_numeric_rows(
                subset,
                METRIC_COLUMNS + [
                    "latent_optimization_time_seconds",
                    "reconstruction_time_seconds",
                ],
            )
            for key, stat_dict in metric_summary.items():
                for stat_name, value in stat_dict.items():
                    rec[f"{key}_{stat_name}"] = value
            summary_records.append(rec)

            # Also write split-specific JSON to match common Stage B structure.
            write_json(
                run_output_root / "evaluation" / split / "evaluation_summary.json",
                {
                    "model_type": model_type,
                    "latent_dim": latent_dim,
                    "split": split,
                    "sample_count": len(subset),
                    "metrics_per_sample": {
                        key: metric_summary[key]
                        for key in METRIC_COLUMNS
                    },
                    "timing_per_sample": {
                        key: metric_summary[key]
                        for key in [
                            "latent_optimization_time_seconds",
                            "reconstruction_time_seconds",
                        ]
                        if key in metric_summary
                    },
                },
            )
            pd.DataFrame(subset).to_csv(
                run_output_root / "evaluation" / split / "evaluation_per_sample.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame([r for r in per_layer_rows if r["split"] == split]).to_csv(
                run_output_root / "evaluation" / split / "evaluation_per_layer.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if summary_records:
            pd.DataFrame(summary_records).to_csv(
                run_output_root / "evaluation_summary_by_split.csv",
                index=False,
                encoding="utf-8-sig",
            )

    return {
        "model_type": model_type,
        "latent_dim": latent_dim,
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "output_dir": str(run_output_root),
        "latent_rows": len(latent_rows),
        "evaluation_sample_rows": len(per_sample_rows),
    }


# =============================================================================
# Advanced reconstruction overlay visualization
# =============================================================================

def resolve_image_path(
    source_image: str | Path,
    feature_file: str | Path | None = None,
    original_image_root: str | Path | None = None,
    image_index: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
    """Resolve the original layered-image path used by ★3.18A步骤处理ALL."""
    try:
        src = Path(str(source_image))
        if src.is_file():
            return src
    except Exception:
        pass

    sample_name = infer_sample_name(feature_file if feature_file is not None else source_image)

    if image_index is not None and sample_name in image_index:
        return image_index[sample_name]

    if original_image_root is not None:
        root = Path(original_image_root)
        if root.exists():
            for suf in SUPPORTED_IMAGE_SUFFIXES:
                cand = root / f"{sample_name}{suf}"
                if cand.is_file():
                    return cand

    if feature_file is not None:
        feature_path = Path(feature_file)
        search_dirs = [
            feature_path.parent,
            feature_path.parent.parent,
            feature_path.parent.parent / "images",
            feature_path.parent.parent / "source_images",
            feature_path.parent.parent / "cropped_images",
            feature_path.parent.parent / "valid_domain",
            feature_path.parent.parent / "continuous_masks",
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            for suf in SUPPORTED_IMAGE_SUFFIXES:
                cand = d / f"{sample_name}{suf}"
                if cand.is_file():
                    return cand
    return None

def load_image_rgb(
    source_image: str | Path,
    feature_file: str | Path | None = None,
    original_image_root: str | Path | None = None,
    image_index: Optional[Dict[str, Path]] = None,
) -> Optional[np.ndarray]:
    path = resolve_image_path(
        source_image,
        feature_file=feature_file,
        original_image_root=original_image_root,
        image_index=image_index,
    )
    if path is None:
        return None
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def infer_layer_colors_from_layer_names(
    layer_names: Optional[Sequence[str]] = None,
    vertical_ranks: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Build canonical top/middle/bottom colors from Stage A layer color names.

    The returned keys are top/middle/bottom because the overlay rendering draws
    by layer index. The RGB values, however, are derived from the actual Stage A
    layer_names/color_name entries, not from a fixed top-red/middle-green/bottom-blue
    template.
    """
    layer_names = list(layer_names) if layer_names is not None else []
    vertical_ranks = list(vertical_ranks) if vertical_ranks is not None else ["top", "middle", "bottom"]
    out: Dict[str, Tuple[float, float, float]] = {}

    for i in range(3):
        rank = str(vertical_ranks[i]).lower() if i < len(vertical_ranks) else ["top", "middle", "bottom"][i]
        name = str(layer_names[i]).lower() if i < len(layer_names) else rank

        if name in DATABASE_COLOR_NAME_RGB:
            out[rank] = DATABASE_COLOR_NAME_RGB[name]
        elif rank in DATABASE_COLOR_NAME_RGB:
            out[rank] = DATABASE_COLOR_NAME_RGB[rank]
        else:
            out[rank] = FALLBACK_LAYER_COLORS.get(rank, (0.5, 0.5, 0.5))

    for rank in ["top", "middle", "bottom"]:
        out.setdefault(rank, FALLBACK_LAYER_COLORS[rank])
    return out


def infer_layer_colors_from_source_image(
    source_image: str | Path,
    gt_mask: np.ndarray,
    layer_names: Optional[Sequence[str]] = None,
    vertical_ranks: Optional[Sequence[str]] = None,
    feature_file: str | Path | None = None,
    original_image_root: str | Path | None = None,
    image_index: Optional[Dict[str, Path]] = None,
    polygon_json_root: str | Path | None = None,
    polygon_json_index: Optional[Dict[str, Path]] = None,
    sample_name: str | None = None,
    strict: bool = True,
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Any]]:
    """Infer per-layer colors from Stage A database information.

    Color priority for final visualization:
    1. Stage A feature npz layer_names, if they contain color names;
    2. *_layer_polygons.json color_name + vertical_rank;
    3. original image median color sampled by gt_mask;
    4. fallback top/middle/bottom colors only when strict=False or no source is available.

    Returns:
        layer_colors: dict keyed by top/middle/bottom for plotting by layer rank
        color_info: diagnostic information printed before drawing
    """
    sample_name = str(sample_name or infer_sample_name(feature_file if feature_file is not None else source_image))

    resolved_layer_names, resolved_vertical_ranks, color_source = get_stageA_color_identity(
        sample_name=sample_name,
        layer_names=layer_names,
        vertical_ranks=vertical_ranks,
        feature_file=feature_file,
        polygon_json_root=polygon_json_root,
        polygon_json_index=polygon_json_index,
    )

    # Best source: Stage A / polygon JSON color identity.
    if any(str(v).lower() in DATABASE_COLOR_NAME_RGB for v in resolved_layer_names):
        colors = infer_layer_colors_from_layer_names(resolved_layer_names, resolved_vertical_ranks)
        info = {
            "sample_name": sample_name,
            "color_source": color_source,
            "layer_names": resolved_layer_names,
            "vertical_ranks": resolved_vertical_ranks,
            "colors": colors,
        }
        return colors, info

    # Optional image-based fallback.
    rgb = load_image_rgb(
        source_image,
        feature_file=feature_file,
        original_image_root=original_image_root,
        image_index=image_index,
    )
    if rgb is not None:
        h_img, w_img = rgb.shape[:2]
        h_m, w_m = gt_mask.shape[-2:]
        if h_img != h_m or w_img != w_m:
            rgb = cv2.resize(rgb, (w_m, h_m), interpolation=cv2.INTER_LINEAR)

        colors_by_rank: Dict[str, Tuple[float, float, float]] = {}
        for i in range(min(3, gt_mask.shape[0])):
            rank = resolved_vertical_ranks[i] if i < len(resolved_vertical_ranks) else ["top", "middle", "bottom"][i]
            mask = gt_mask[i] > 0
            if int(mask.sum()) == 0:
                colors_by_rank[rank] = FALLBACK_LAYER_COLORS.get(rank, (0.5, 0.5, 0.5))
                continue

            pixels = rgb[mask]
            valid = ~np.all(pixels >= 245, axis=1)
            if np.any(valid):
                pixels = pixels[valid]
            if pixels.shape[0] == 0:
                colors_by_rank[rank] = FALLBACK_LAYER_COLORS.get(rank, (0.5, 0.5, 0.5))
                continue

            median_rgb = np.median(pixels.astype(np.float32), axis=0) / 255.0
            median_rgb = np.clip(median_rgb, 0.0, 1.0)
            colors_by_rank[rank] = (float(median_rgb[0]), float(median_rgb[1]), float(median_rgb[2]))

        for rank in ["top", "middle", "bottom"]:
            colors_by_rank.setdefault(rank, FALLBACK_LAYER_COLORS[rank])

        info = {
            "sample_name": sample_name,
            "color_source": "original_image_median_color",
            "layer_names": resolved_layer_names,
            "vertical_ranks": resolved_vertical_ranks,
            "colors": colors_by_rank,
        }
        return colors_by_rank, info

    if strict:
        raise FileNotFoundError(
            f"Could not resolve Stage A color identity for sample '{sample_name}'. "
            f"Please provide --polygon_json_root pointing to the directory containing *_layer_polygons.json, "
            f"or ensure feature npz contains layer_names such as red/yellow/green/blue."
        )

    colors = infer_layer_colors_from_layer_names(resolved_layer_names, resolved_vertical_ranks)
    info = {
        "sample_name": sample_name,
        "color_source": color_source,
        "layer_names": resolved_layer_names,
        "vertical_ranks": resolved_vertical_ranks,
        "colors": colors,
    }
    return colors, info

def layer_name(index: int) -> str:
    return ["top", "middle", "bottom"][index]


def crop_bounds_from_union(masks: Sequence[np.ndarray], margin: int = 20) -> Tuple[int, int, int, int]:
    union = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        union = np.maximum(union, (mask > 0).astype(np.uint8))
    ys, xs = np.where(union > 0)
    if len(xs) == 0:
        return 0, union.shape[0], 0, union.shape[1]
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(union.shape[0], int(ys.max()) + margin + 1)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(union.shape[1], int(xs.max()) + margin + 1)
    return y1, y2, x1, x2


def export_reconstruction_overlay_advanced(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
    title: str = "",
    dpi: int = 800,
    crop_margin: int = 20,
    layer_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Dict[str, float]:
    """Create a Stage-A-like advanced overlay.

    Visual policy:
    1. Coloured fills use reconstructed masks, not reference masks.
    2. Black dashed reference boundaries are drawn last and above all layers.
    """
    assert gt_mask.shape[0] == 3 and pred_mask.shape[0] == 3

    h, w = gt_mask.shape[-2:]
    diagonal = math.sqrt(h * h + w * w)

    ious, dices, chamfers, hausdorffs = [], [], [], []
    for i in range(3):
        ious.append(compute_iou(gt_mask[i], pred_mask[i]))
        dices.append(compute_dice(gt_mask[i], pred_mask[i]))
        ch, hd = compute_boundary_distances(gt_mask[i], pred_mask[i], diagonal)
        chamfers.append(ch)
        hausdorffs.append(hd)

    metrics = {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_chamfer": float(np.mean(chamfers)),
        "mean_hausdorff": float(np.mean(hausdorffs)),
    }

    bounds = crop_bounds_from_union(
        [gt_mask[i] for i in range(3)] + [pred_mask[i] for i in range(3)],
        margin=crop_margin,
    )
    y1, y2, x1, x2 = bounds

    fig = plt.figure(figsize=(6.30, 3.25), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.85], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_frame_on(False)
    legend_ax.set_facecolor("white")

    draw_order = [2, 1, 0]  # bottom -> middle -> top
    if layer_colors is None:
        layer_colors = {name: FALLBACK_LAYER_COLORS[name] for name in ["top", "middle", "bottom"]}

    # 1. Reconstructed fills.
    for i in draw_order:
        lname = layer_name(i)
        color = layer_colors.get(lname, FALLBACK_LAYER_COLORS[lname])
        crop = pred_mask[i, y1:y2, x1:x2]
        rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=float)
        rgba[..., :3] = color
        rgba[..., 3] = 0.22 * (crop > 0).astype(float)
        ax.imshow(rgba, origin="upper", zorder=1 + i)

    # 2. Reconstructed colored boundaries.
    for i in draw_order:
        lname = layer_name(i)
        color = layer_colors.get(lname, FALLBACK_LAYER_COLORS[lname])
        pred_crop = pred_mask[i, y1:y2, x1:x2]
        if np.any(pred_crop > 0):
            ax.contour(
                pred_crop,
                levels=[0.5],
                colors=[color],
                linewidths=[1.15],
                linestyles=["solid"],
                zorder=20 + i,
            )

    # 3. Reference boundaries on top.
    for i in draw_order:
        ref_crop = gt_mask[i, y1:y2, x1:x2]
        if np.any(ref_crop > 0):
            ax.contour(
                ref_crop,
                levels=[0.5],
                colors=["#202020"],
                linewidths=[0.75],
                linestyles=["dashed"],
                zorder=100 + i,
            )

    if title:
        ax.set_title(title, pad=3)

    handles = [
        mlines.Line2D([], [], color="#202020", linewidth=0.95, linestyle="--", label="Reference boundary"),
        mlines.Line2D([], [], color=layer_colors.get("top", FALLBACK_LAYER_COLORS["top"]), linewidth=1.5, label="Top layer reconstruction"),
        mlines.Line2D([], [], color=layer_colors.get("middle", FALLBACK_LAYER_COLORS["middle"]), linewidth=1.5, label="Middle layer reconstruction"),
        mlines.Line2D([], [], color=layer_colors.get("bottom", FALLBACK_LAYER_COLORS["bottom"]), linewidth=1.5, label="Bottom layer reconstruction"),
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.0,
        labelspacing=0.72,
    )
    metric_text = "\n".join([
        f"Mean IoU: {metrics['mean_iou']:.4f}",
        f"Mean Dice: {metrics['mean_dice']:.4f}",
        f"Mean Chamfer: {metrics['mean_chamfer']:.5f}",
        f"Mean Hausdorff: {metrics['mean_hausdorff']:.5f}",
    ])
    legend_ax.text(
        0.00,
        0.34,
        metric_text,
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ensure_dir(out_path.parent)
    fig.savefig(
        out_path,
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)
    return metrics


def choose_example_sample_names(
    checkpoint: Path,
    split: str,
    examples_per_split: int,
    explicit_samples: Sequence[str],
) -> List[str]:
    """Choose visual examples for one split.

    If explicit sample names are provided, only samples that actually belong to
    the current split are returned. If explicit sample names are not provided,
    the manuscript representative samples 59, 358, 573, 768, 997, and 1173
    are used first. This prevents the script from accidentally selecting the
    first N samples in each split.
    """
    split_dict = split_files_from_checkpoint(checkpoint)
    available = [infer_sample_name(p) for p in split_dict.get(split, [])]
    available_set = set(available)

    explicit = [str(s).strip() for s in explicit_samples if str(s).strip()]
    target_names = explicit if explicit else DEFAULT_REPRESENTATIVE_SAMPLE_NAMES

    chosen = [s for s in target_names if s in available_set]
    if chosen:
        return chosen

    # Fallback only when none of the requested representative samples belongs
    # to this split.
    return available[: int(examples_per_split)]


def export_visual_examples(
    selected_rows: Sequence[Dict[str, Any]],
    output_root: Path,
    splits: Sequence[str],
    examples_per_split: int,
    sample_names: Sequence[str],
    dpi: int,
    original_image_root: str | Path | None = None,
    image_index: Optional[Dict[str, Path]] = None,
    polygon_json_root: str | Path | None = None,
    polygon_json_index: Optional[Dict[str, Path]] = None,
    strict_original_colors: bool = True,
) -> List[Path]:
    outputs: List[Path] = []
    for row in selected_rows:
        model_type = row["model_type"]
        latent_dim = int(row["latent_dim"])
        run_output = output_root / "runs" / model_type / f"latent_dim_{latent_dim}"
        checkpoint = Path(row["checkpoint"])

        for split in splits:
            chosen = choose_example_sample_names(checkpoint, split, examples_per_split, sample_names)
            for sample_name in chosen:
                pred_file = run_output / "predictions" / split / sample_name / f"{sample_name}_predicted_fields.npz"
                if not pred_file.is_file():
                    print(f"[Warning] Missing predicted field file, skip visual: {pred_file}")
                    continue
                with np.load(pred_file, allow_pickle=True) as data:
                    gt_mask = data["gt_mask"].astype(np.uint8)
                    pred_mask = data["pred_mask"].astype(np.uint8)
                    source_image = str(data["source_image"].tolist()) if "source_image" in data else sample_name
                    feature_file = str(data["feature_file"].tolist()) if "feature_file" in data else None
                    if "layer_names" in data:
                        layer_names = [str(v) for v in data["layer_names"].tolist()]
                    else:
                        layer_names = ["top", "middle", "bottom"]
                    if "vertical_ranks" in data:
                        vertical_ranks = [str(v) for v in data["vertical_ranks"].tolist()]
                    else:
                        vertical_ranks = ["top", "middle", "bottom"]

                layer_colors, color_info = infer_layer_colors_from_source_image(
                    source_image=source_image,
                    gt_mask=gt_mask,
                    layer_names=layer_names,
                    vertical_ranks=vertical_ranks,
                    feature_file=feature_file,
                    original_image_root=original_image_root,
                    image_index=image_index,
                    polygon_json_root=polygon_json_root,
                    polygon_json_index=polygon_json_index,
                    sample_name=sample_name,
                    strict=strict_original_colors,
                )

                out_path = (
                    output_root
                    / "selected_reconstruction_visuals"
                    / model_type
                    / f"latent_dim_{latent_dim}"
                    / split
                    / f"{sample_name}_{model_type}_d{latent_dim}_advanced_overlay.png"
                )
                title = f"{sample_name} ({model_type}, d={latent_dim}, {split})"
                print(
                    f"[Color] {sample_name} | {model_type} d={latent_dim} | "
                    f"source={color_info.get('color_source')} | "
                    f"layer_names={color_info.get('layer_names')} | "
                    f"vertical_ranks={color_info.get('vertical_ranks')} | "
                    f"colors={layer_colors}"
                )
                export_reconstruction_overlay_advanced(
                    gt_mask,
                    pred_mask,
                    out_path,
                    title=title,
                    dpi=dpi,
                    crop_margin=20,
                    layer_colors=layer_colors,
                )
                outputs.append(out_path)
    return outputs


# =============================================================================
# Master summary
# =============================================================================

def collect_global_evaluation_summary(output_root: Path, processed: Sequence[Dict[str, Any]], splits: Sequence[str]) -> pd.DataFrame:
    rows = []
    for item in processed:
        model = item["model_type"]
        dim = int(item["latent_dim"])
        path = output_root / "runs" / model / f"latent_dim_{dim}" / "evaluation_summary_by_split.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    ensure_dir(output_root / "tables")
    out.to_csv(output_root / "tables" / "evaluation_summary_by_model_dim_split.csv", index=False, encoding="utf-8-sig")
    return out


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrated Stage B Section 4.4 exporter without external latent-export scripts."
    )
    parser.add_argument("--stageb_root", type=str, required=True, help="Root folder containing Stage B model folders.")
    parser.add_argument("--feature_root", type=str, required=True, help="Stage A output root containing features/*_features.npz.")
    parser.add_argument("--output_root", type=str, required=True, help="Output folder for Section 4.4 materials.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--selected_vae_dim", type=int, default=16)
    parser.add_argument("--selected_autodecoder_dim", type=int, default=64)
    parser.add_argument("--only_selected_models", action="store_true", help="Only process VAE selected_dim and auto-decoder selected_dim.")
    parser.add_argument("--run_latent_export", action="store_true", help="Export latent feature tables.")
    parser.add_argument("--run_evaluation", action="store_true", help="Run integrated reconstruction evaluation.")
    parser.add_argument("--run_predictions", action="store_true", help="Save predicted dense fields. Automatically enabled by --run_visual_examples.")
    parser.add_argument("--run_visual_examples", action="store_true", help="Export selected advanced overlay examples.")
    parser.add_argument("--examples_per_split", type=int, default=2)
    parser.add_argument(
        "--sample_names",
        nargs="*",
        default=[],
        help=(
            "Representative sample names for visualization. If omitted, the "
            "manuscript representative samples 59, 358, 573, 768, 997, and "
            "1173 are used."
        ),
    )
    parser.add_argument("--optimize_all_autodecoder", action="store_true", help="Force latent optimization for all auto-decoder samples.")
    parser.add_argument("--auto_optimization_steps", type=int, default=None)
    parser.add_argument("--auto_optimization_lr", type=float, default=None)
    parser.add_argument("--batch_size_vae", type=int, default=16)
    parser.add_argument("--eval_grid_size", type=int, default=None, help="Override eval grid size. Default uses checkpoint config.")
    parser.add_argument("--eval_chunk_size", type=int, default=65536)
    parser.add_argument("--dpi", type=int, default=800)
    parser.add_argument(
        "--original_image_root",
        type=str,
        default="",
        help=(
            "Optional root folder of the original image database. This is only "
            "used as a fallback after Stage A layer color names and polygon JSON."
        ),
    )
    parser.add_argument(
        "--polygon_json_root",
        type=str,
        default="",
        help=(
            "Root folder containing *_layer_polygons.json from the Stage A boundary "
            "extraction stage, e.g. ★3.18边界提取ALL. Color names are read from "
            "these JSON files when feature npz layer_names are unavailable."
        ),
    )
    parser.add_argument(
        "--strict_original_colors",
        action="store_true",
        help="Abort visual-example export if original database colors cannot be resolved.",
    )
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or blank for auto.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    stageb_root = Path(args.stageb_root)
    feature_root = Path(args.feature_root)
    output_root = ensure_dir(args.output_root)

    original_image_root = args.original_image_root.strip() if str(args.original_image_root).strip() else None
    polygon_json_root = args.polygon_json_root.strip() if str(args.polygon_json_root).strip() else None
    image_index = build_image_index(original_image_root)
    polygon_json_index = build_polygon_json_index(polygon_json_root)

    if args.device:
        device = torch.device(args.device)
    else:
        device = default_device()
    print(f"Device: {device}")

    rows_all = discover_runs(stageb_root)
    if not rows_all:
        raise RuntimeError(f"No valid Stage B runs with checkpoints were found under: {stageb_root}")

    rows = filter_selected_runs(
        rows_all,
        selected_vae_dim=int(args.selected_vae_dim),
        selected_autodecoder_dim=int(args.selected_autodecoder_dim),
        only_selected_models=bool(args.only_selected_models),
    )
    if not rows:
        raise RuntimeError("No runs remained after filtering.")

    ensure_dir(output_root / "tables")
    pd.DataFrame(rows_all).to_csv(output_root / "tables" / "run_index_all_detected.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output_root / "tables" / "run_index_processed.csv", index=False, encoding="utf-8-sig")

    # Visual examples need saved prediction fields.
    run_predictions = bool(args.run_predictions or args.run_visual_examples)

    processed: List[Dict[str, Any]] = []
    for row in rows:
        print(f"\n[Process] {row['model_type']} | d={row['latent_dim']}")
        info = export_and_evaluate_run(
            row=row,
            feature_root=feature_root,
            output_root=output_root,
            splits=args.splits,
            run_latent_export=bool(args.run_latent_export),
            run_evaluation=bool(args.run_evaluation),
            run_predictions=run_predictions,
            optimize_all_autodecoder=bool(args.optimize_all_autodecoder),
            auto_optimization_steps=args.auto_optimization_steps,
            auto_optimization_lr=args.auto_optimization_lr,
            batch_size_vae=int(args.batch_size_vae),
            device=device,
            eval_grid_size_override=args.eval_grid_size,
            eval_chunk_size=int(args.eval_chunk_size),
        )
        processed.append(info)

    summary_df = collect_global_evaluation_summary(output_root, processed, args.splits)

    produced_visuals: List[Path] = []
    if args.run_visual_examples:
        selected_rows = filter_selected_runs(
            rows_all,
            selected_vae_dim=int(args.selected_vae_dim),
            selected_autodecoder_dim=int(args.selected_autodecoder_dim),
            only_selected_models=True,
        )
        produced_visuals = export_visual_examples(
            selected_rows,
            output_root,
            splits=args.splits,
            examples_per_split=int(args.examples_per_split),
            sample_names=args.sample_names,
            dpi=int(args.dpi),
            original_image_root=original_image_root,
            image_index=image_index,
            polygon_json_root=polygon_json_root,
            polygon_json_index=polygon_json_index,
            strict_original_colors=bool(args.strict_original_colors),
        )

    report = {
        "stageb_root": str(stageb_root),
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "device": str(device),
        "splits": args.splits,
        "processed_runs": processed,
        "summary_rows": int(len(summary_df)) if not summary_df.empty else 0,
        "visual_examples": [str(p) for p in produced_visuals],
        "default_representative_sample_names": DEFAULT_REPRESENTATIVE_SAMPLE_NAMES,
        "requested_sample_names": list(args.sample_names),
        "original_image_root": str(original_image_root) if original_image_root else "",
        "polygon_json_root": str(polygon_json_root) if polygon_json_root else "",
        "strict_original_colors": bool(args.strict_original_colors),
        "indexed_original_images": int(len(image_index)),
        "indexed_polygon_jsons": int(len(polygon_json_index)),
        "run_latent_export": bool(args.run_latent_export),
        "run_evaluation": bool(args.run_evaluation),
        "run_predictions": bool(run_predictions),
        "run_visual_examples": bool(args.run_visual_examples),
    }
    write_json(output_root / "reports" / "integrated_section4p4_report.json", report)

    print("\nIntegrated Section 4.4 export completed.")
    print(f"Processed runs: {len(processed)}")
    print(f"Output root: {output_root}")
    if produced_visuals:
        print(f"Visual examples: {len(produced_visuals)}")
    if not summary_df.empty:
        print(f"Summary table: {output_root / 'tables' / 'evaluation_summary_by_model_dim_split.csv'}")


if __name__ == "__main__":
    main()
