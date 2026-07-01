# -*- coding: utf-8 -*-
"""
05.1_export_all_latent_features_by_posterior_fitting.py

功能：
1. 加载已有 single-layer field 数据库。
2. 加载训练好的 best checkpoint。
3. 固定训练好的 decoder，对全集样本逐批执行 posterior latent fitting。
4. 对每个样本导出 latent vector、模型重建 IoU、Dice、Chamfer、Hausdorff 等指标。
5. 过滤 IoU > 0.85 的样本，导出用于下游分类的 Latent features 表格。

适配训练脚本：
02_train_single_layer_field_model_GUI.py

核心逻辑：
- AutoDecoder 的 checkpoint 中 latent.weight 只对应 train split。
- 全集样本的 latent features 需要通过固定 decoder 后验优化得到。
- 因此本脚本不再要求 checkpoint 中存在 8100 × d 的 latent matrix。
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


DEFAULT_DATASET_ROOT = r"D:\博士期间发表论文\2026.3.19_从层状沉积物边界到可重构的低维特征\Results第五章\1"


# ============================================================
# GUI helpers
# ============================================================

def make_root():
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def ask_directory(title: str, initialdir: Optional[str] = None) -> Optional[Path]:
    root = make_root()
    path = filedialog.askdirectory(title=title, initialdir=initialdir or "")
    root.destroy()
    return Path(path) if path else None


def ask_open_file(title: str, filetypes: List[Tuple[str, str]], initialdir: Optional[str] = None) -> Optional[Path]:
    root = make_root()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initialdir or "")
    root.destroy()
    return Path(path) if path else None


def ask_string(title: str, prompt: str, initialvalue: str = "") -> str:
    root = make_root()
    value = simpledialog.askstring(title=title, prompt=prompt, initialvalue=initialvalue)
    root.destroy()
    return "" if value is None else value.strip()


def show_info(msg: str):
    root = make_root()
    messagebox.showinfo("Finished", msg)
    root.destroy()


def show_warning(msg: str):
    root = make_root()
    messagebox.showwarning("Warning", msg)
    root.destroy()


# ============================================================
# Config and model architecture
# ============================================================

@dataclass
class Config:
    feature_root: str
    output_root: str
    model_type: str = "autodecoder"
    seed: int = 2026
    latent_dim: int = 64
    encoder_input_size: int = 128
    eval_grid_size: int = 256
    batch_size: int = 16
    epochs: int = 5000
    learning_rate: float = 5e-5
    latent_learning_rate: float = 5e-3
    weight_decay: float = 1e-6
    points_per_sample: int = 8192
    eval_chunk_size: int = 32768
    validate_every: int = 1
    patience: int = 50
    min_delta: float = 1e-5
    save_every: int = 10
    num_workers: int = 4
    beta_kl: float = 1e-4
    weight_sdf: float = 1.0
    weight_corner: float = 0.15
    weight_mask: float = 0.5
    weight_dice: float = 0.5
    weight_consistency: float = 0.05
    weight_latent_reg: float = 1e-4
    boundary_delta: float = 0.10
    boundary_boost: float = 4.0
    mask_temperature: float = 0.075
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 6
    decoder_fourier_levels: int = 8
    posterior_steps: int = 300
    posterior_lr: float = 2e-2
    val_posterior_steps: int = 100
    val_posterior_lr: float = 2e-2
    val_max_batches: int = 0
    show_val_progress: bool = True
    cache_in_memory: bool = False
    device: str = "cuda"


def config_from_checkpoint(ckpt: dict, dataset_root: Path, output_root: Path, device: str) -> Config:
    cfg_dict = dict(ckpt.get("config", {}))
    cfg_dict["feature_root"] = str(dataset_root)
    cfg_dict["output_root"] = str(output_root)
    cfg_dict["device"] = device

    valid = {f.name for f in fields(Config)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}

    return Config(**cfg_dict)


class Fourier(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.n = int(n)
        self.register_buffer(
            "freqs",
            2.0 ** torch.arange(self.n, dtype=torch.float32),
            persistent=False
        )

    @property
    def output_dim(self):
        return 2 + 4 * self.n

    def forward(self, c):
        x = c[..., None] * self.freqs[None, None, None, :] * math.pi
        return torch.cat(
            [
                c,
                torch.sin(x).reshape(c.shape[0], c.shape[1], -1),
                torch.cos(x).reshape(c.shape[0], c.shape[1], -1),
            ],
            dim=-1
        )


class ConvEncoder(nn.Module):
    def __init__(self, in_ch=3, latent_dim=64):
        super().__init__()
        ch = [32, 64, 128, 256, 256]
        layers = []
        ci = in_ch
        for co in ch:
            layers += [
                nn.Conv2d(ci, co, 4, 2, 1),
                nn.BatchNorm2d(co),
                nn.LeakyReLU(0.2, True),
            ]
            ci = co
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mu = nn.Linear(ch[-1], latent_dim)
        self.lv = nn.Linear(ch[-1], latent_dim)

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.mu(h), self.lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden, layers_n, fourier):
        super().__init__()
        self.ff = Fourier(fourier)
        d = self.ff.output_dim + latent_dim
        layers = []
        for _ in range(layers_n - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU(True)]
            d = hidden
        layers.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, z, coords):
        B, P, _ = coords.shape
        cf = self.ff(coords)
        ze = z[:, None, :].expand(-1, P, -1)
        y = self.net(torch.cat([cf, ze], dim=-1))
        return {
            "sdf": y[..., 0:1],
            "corner_logits": y[..., 1:2],
            "mask_logits": y[..., 2:3],
        }


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ConvEncoder(3, cfg.latent_dim)
        self.decoder = Decoder(
            cfg.latent_dim,
            cfg.decoder_hidden_dim,
            cfg.decoder_num_layers,
            cfg.decoder_fourier_levels
        )


class AutoDecoder(nn.Module):
    def __init__(self, n_train, cfg):
        super().__init__()
        self.latent = nn.Embedding(n_train, cfg.latent_dim)
        nn.init.normal_(self.latent.weight, 0, 0.01)
        self.decoder = Decoder(
            cfg.latent_dim,
            cfg.decoder_hidden_dim,
            cfg.decoder_num_layers,
            cfg.decoder_fourier_levels
        )


# ============================================================
# Dataset metadata and full dataset
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_splits(root: Path) -> Dict[str, List[str]]:
    split_path = Path(root) / "metadata" / "split_stratified.json"
    obj = json.loads(split_path.read_text(encoding="utf-8"))
    return obj["splits"] if "splits" in obj else obj


def extract_sample_order(sample_name: str) -> int:
    m = re.search(r"sample_(\d+)", str(sample_name))
    if m:
        return int(m.group(1))
    return 10**18


def load_full_metadata(dataset_root: Path) -> pd.DataFrame:
    meta_dir = Path(dataset_root) / "metadata"
    selected_path = meta_dir / "selected_indices.csv"
    shape_path = meta_dir / "shape_descriptors.csv"
    split_path = meta_dir / "split_stratified.json"

    if not selected_path.exists():
        raise FileNotFoundError(f"Cannot find selected_indices.csv: {selected_path}")

    selected = pd.read_csv(selected_path)
    selected["sample_name"] = selected["sample_name"].astype(str)

    if shape_path.exists():
        shape_df = pd.read_csv(shape_path)
        shape_df["sample_name"] = shape_df["sample_name"].astype(str)

        add_cols = []
        for c in shape_df.columns:
            if c == "sample_name" or c not in selected.columns:
                add_cols.append(c)

        selected = selected.merge(
            shape_df[add_cols],
            on="sample_name",
            how="left",
            suffixes=("", "_shape")
        )

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        splits = split_obj.get("splits", split_obj)

        split_map = {}
        for split_name, names in splits.items():
            for name in names:
                split_map[str(name)] = split_name

        selected["split"] = selected["sample_name"].map(split_map)

    if "source_id" in selected.columns:
        selected["FID"] = selected["source_id"].astype(str)
    elif "source_index" in selected.columns:
        selected["FID"] = selected["source_index"].astype(str)
    else:
        selected["FID"] = selected["sample_name"].astype(str)

    selected["_sample_order"] = selected["sample_name"].apply(extract_sample_order)
    selected = selected.sort_values("_sample_order").reset_index(drop=True)

    return selected


class FullFeatureDataset(Dataset):
    def __init__(self, dataset_root: Path, metadata: pd.DataFrame, enc_size: int, max_samples: int = 0):
        self.root = Path(dataset_root)
        self.metadata = metadata.copy()

        if max_samples and max_samples > 0:
            self.metadata = self.metadata.iloc[:max_samples].copy()

        self.enc_size = int(enc_size)

    def __len__(self):
        return len(self.metadata)

    def _resolve_feature_file(self, row) -> Path:
        sample_name = str(row["sample_name"])

        # Prefer current dataset root to avoid stale absolute path stored in CSV.
        fp = self.root / "features" / f"{sample_name}_features.npz"
        if fp.exists():
            return fp

        if "feature_file" in row and pd.notna(row["feature_file"]):
            old_fp = Path(str(row["feature_file"]))
            if old_fp.exists():
                return old_fp

        raise FileNotFoundError(f"Cannot find feature npz for {sample_name}")

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fp = self._resolve_feature_file(row)

        with np.load(fp, allow_pickle=True) as d:
            ft = d["feature_tensor"].astype(np.float32)
            sdf = d["sdf_normalized"].astype(np.float32)
            corner = d["corner_field"].astype(np.float32)
            mask = d["mask"].astype(np.float32)

            sample_name = str(d["sample_name"].tolist()) if "sample_name" in d else str(row["sample_name"])
            source_image = str(d["source_image"].tolist()) if "source_image" in d else str(row.get("source_id", sample_name))

        small = cv2.resize(
            np.transpose(ft, (1, 2, 0)),
            (self.enc_size, self.enc_size),
            interpolation=cv2.INTER_LINEAR
        )
        small = np.transpose(small, (2, 0, 1)).astype(np.float32)

        return {
            "sample_name": sample_name,
            "FID": str(row.get("FID", "")),
            "source_id": str(row.get("source_id", "")),
            "source_index": str(row.get("source_index", "")),
            "abcd_class": str(row.get("abcd_class", "")),
            "split": str(row.get("split", "")),
            "feature_file": str(fp),
            "source_image": source_image,
            "feature_small": small,
            "sdf": sdf,
            "corner": corner,
            "mask": mask,
        }


def collate_full(batch):
    return {
        "sample_name": [b["sample_name"] for b in batch],
        "FID": [b["FID"] for b in batch],
        "source_id": [b["source_id"] for b in batch],
        "source_index": [b["source_index"] for b in batch],
        "abcd_class": [b["abcd_class"] for b in batch],
        "split": [b["split"] for b in batch],
        "feature_file": [b["feature_file"] for b in batch],
        "source_image": [b["source_image"] for b in batch],
        "feature_small": torch.from_numpy(np.stack([b["feature_small"] for b in batch])),
        "sdf": torch.from_numpy(np.stack([b["sdf"] for b in batch])),
        "corner": torch.from_numpy(np.stack([b["corner"] for b in batch])),
        "mask": torch.from_numpy(np.stack([b["mask"] for b in batch])),
    }


# ============================================================
# Loss, sampling, reconstruction
# ============================================================

def grid(h, w, device):
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def sample_points(sdf, corner, mask, coords_flat, npts):
    B, _, H, W = sdf.shape
    HW = H * W
    idx = torch.randint(0, HW, (B, npts), device=sdf.device)

    def gather(x):
        return torch.gather(
            x.view(B, 1, HW).permute(0, 2, 1),
            1,
            idx.unsqueeze(-1)
        )

    return {
        "coords": coords_flat[idx],
        "sdf": gather(sdf),
        "corner": gather(corner),
        "mask": gather(mask),
    }


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(1)
    union = p.sum(1) + target.sum(1)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def loss_total(out, gsdf, gcorner, gmask, cfg, z=None, mu=None, lv=None):
    w = 1 + cfg.boundary_boost * torch.exp(-torch.abs(gsdf) / max(cfg.boundary_delta, 1e-6))
    lsdf = torch.mean(w * F.smooth_l1_loss(out["sdf"], gsdf, reduction="none"))
    lc = F.mse_loss(torch.sigmoid(out["corner_logits"]), gcorner)
    lm = F.binary_cross_entropy_with_logits(out["mask_logits"], gmask)
    ld = dice_loss(out["mask_logits"], gmask)
    lcon = F.mse_loss(
        torch.sigmoid(out["mask_logits"]),
        torch.sigmoid(-out["sdf"] / max(cfg.mask_temperature, 1e-6))
    )

    total = (
        cfg.weight_sdf * lsdf
        + cfg.weight_corner * lc
        + cfg.weight_mask * lm
        + cfg.weight_dice * ld
        + cfg.weight_consistency * lcon
    )

    if z is not None:
        lreg = torch.mean(z ** 2)
        total = total + cfg.weight_latent_reg * lreg

    return total


def opt_latent_batch(decoder, batch, cfg, device, posterior_steps: int, posterior_lr: float):
    sdf = batch["sdf"].to(device)
    corner = batch["corner"].to(device)
    mask = batch["mask"].to(device)

    B = sdf.shape[0]
    H = sdf.shape[-1]
    coords = grid(H, H, device).view(-1, 2)

    z = torch.zeros((B, cfg.latent_dim), device=device, requires_grad=True)
    optz = torch.optim.Adam([z], lr=posterior_lr)

    decoder.eval()
    params = list(decoder.parameters())
    old_requires_grad = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    try:
        for _ in range(int(posterior_steps)):
            smp = sample_points(sdf, corner, mask, coords, cfg.points_per_sample)
            optz.zero_grad(set_to_none=True)
            out = decoder(z, smp["coords"])
            loss = loss_total(out, smp["sdf"], smp["corner"], smp["mask"], cfg, z=z)
            loss.backward()
            optz.step()
    finally:
        for p, req in zip(params, old_requires_grad):
            p.requires_grad_(req)

    return z.detach()


@torch.no_grad()
def dense_batch(decoder, z, grid_size, device, chunk_size):
    decoder.eval()

    B = z.shape[0]
    coords = grid(grid_size, grid_size, device).view(-1, 2)

    sdf_list = []
    mask_list = []

    for s in range(0, coords.shape[0], int(chunk_size)):
        c = coords[s:s + int(chunk_size)][None].expand(B, -1, -1)
        out = decoder(z, c)
        sdf_list.append(out["sdf"].detach().cpu())
        mask_list.append(torch.sigmoid(out["mask_logits"]).detach().cpu())

    sdf = torch.cat(sdf_list, dim=1).view(B, grid_size, grid_size).numpy().astype(np.float32)
    mask_prob = torch.cat(mask_list, dim=1).view(B, grid_size, grid_size).numpy().astype(np.float32)

    return sdf, mask_prob


def post_process_mask(mask):
    m = (mask > 0).astype(np.uint8)
    n, lab, stat, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)

    if n > 1:
        keep = 1 + int(np.argmax(stat[1:, cv2.CC_STAT_AREA]))
        out = (lab == keep).astype(np.uint8)

    if out.any():
        flood = out.copy()
        h, w = out.shape
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff, (0, 0), 1)
        out = np.maximum(out, (flood == 0).astype(np.uint8))

    return out


def boundary(mask):
    cs, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    pts = [c.reshape(-1, 2).astype(np.float32) for c in cs if len(c) > 1]
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2), np.float32)


def chamfer(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    def mean_min(x, y):
        vals = []
        for s in range(0, len(x), 2048):
            d = x[s:s + 2048, None, :] - y[None, :, :]
            vals.append(np.sqrt((d * d).sum(2)).min(1))
        return float(np.concatenate(vals).mean())

    return 0.5 * (mean_min(a, b) + mean_min(b, a))


def hausdorff(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    def max_min(x, y):
        vals = []
        for s in range(0, len(x), 2048):
            d = x[s:s + 2048, None, :] - y[None, :, :]
            vals.append(np.sqrt((d * d).sum(2)).min(1).max())
        return float(np.max(vals))

    return max(max_min(a, b), max_min(b, a))


def topology(mask):
    m = (mask > 0).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m, 8)
    comp = max(0, n - 1)
    cs, h = cv2.findContours(m * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if h is not None:
        holes = sum(1 for x in h[0] if x[3] != -1)
    return comp, holes


def compute_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    pred_area = float(pred.sum())
    gt_area = float(gt.sum())

    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred_area + gt_area) if pred_area + gt_area else 1.0

    H, W = gt.shape
    diag = math.hypot(H, W)

    bp = boundary(pred)
    bg = boundary(gt)
    ch = chamfer(bp, bg)
    ha = hausdorff(bp, bg)

    comp, holes = topology(pred)

    def centroid(m):
        y, x = np.where(m > 0)
        if len(x) == 0:
            return None
        return np.array([x.mean(), y.mean()])

    cp = centroid(pred)
    cg = centroid(gt)
    cd = 1.0 if cp is None or cg is None else min(1.0, float(np.linalg.norm(cp - cg) / diag))

    return {
        "iou": iou,
        "dice": dice,
        "area_relative_error": abs(pred_area - gt_area) / gt_area if gt_area else 0.0,
        "centroid_distance_normalized": cd,
        "boundary_chamfer_normalized": 1.0 if not np.isfinite(ch) else min(1.0, ch / diag),
        "boundary_hausdorff_normalized": 1.0 if not np.isfinite(ha) else min(1.0, ha / diag),
        "component_count": float(comp),
        "hole_count": float(holes),
        "invalid_topology": float(comp != 1 or holes != 0),
    }


# ============================================================
# Model loading
# ============================================================

def load_checkpoint(ckpt_path: Path, device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def build_model_from_checkpoint(ckpt, cfg, n_train, device):
    model_type = ckpt.get("model_type", cfg.model_type)
    cfg.model_type = model_type

    if model_type == "vae":
        model = VAE(cfg).to(device)
    elif model_type == "autodecoder":
        model = AutoDecoder(n_train, cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ============================================================
# Export process
# ============================================================

def save_tables(rows, out_dir, iou_threshold, latent_dim):
    if not rows:
        return None, None

    df = pd.DataFrame(rows)

    meta_cols = [
        "sample_name", "FID", "source_id", "source_index",
        "abcd_class", "split", "feature_file"
    ]
    metric_cols = [
        "iou", "dice", "area_relative_error",
        "centroid_distance_normalized",
        "boundary_chamfer_normalized",
        "boundary_hausdorff_normalized",
        "component_count", "hole_count", "invalid_topology"
    ]
    z_cols = [f"z_{j:03d}" for j in range(latent_dim)]

    ordered_cols = [c for c in meta_cols if c in df.columns] + metric_cols + z_cols
    df = df[ordered_cols].copy()

    all_path = out_dir / "latent_features_all_samples.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")

    filtered = df[pd.to_numeric(df["iou"], errors="coerce") > float(iou_threshold)].copy()
    filtered_path = out_dir / f"latent_features_iou_gt_{str(iou_threshold).replace('.', 'p')}.csv"
    filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    metrics_path = out_dir / "reconstruction_metrics_all_samples.csv"
    metric_export_cols = [c for c in meta_cols if c in df.columns] + metric_cols
    df[metric_export_cols].to_csv(metrics_path, index=False, encoding="utf-8-sig")

    return all_path, filtered_path


def main():
    default_root = Path(DEFAULT_DATASET_ROOT)

    if default_root.exists():
        use_default = ask_string(
            "Dataset root",
            f"Use default dataset root?\n{default_root}\n\nInput y to use it, or n to choose another folder.",
            initialvalue="y"
        ).lower()

        if use_default in ["", "y", "yes"]:
            dataset_root = default_root
        else:
            dataset_root = ask_directory("Select dataset root folder")
    else:
        dataset_root = ask_directory("Select dataset root folder")

    if dataset_root is None:
        print("[Cancel] No dataset root selected.")
        return

    ckpt_path = ask_open_file(
        "Select best checkpoint",
        [("PyTorch checkpoint", "*.pt *.pth *.ckpt"), ("All files", "*.*")]
    )
    if ckpt_path is None:
        print("[Cancel] No checkpoint selected.")
        return

    out_dir = ask_directory("Select output folder for exported latent features")
    if out_dir is None:
        print("[Cancel] No output folder selected.")
        return
    out_dir = ensure_dir(out_dir)

    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = ask_string(
        "Device",
        "Input device:",
        initialvalue=device_default
    )
    if not device_str:
        device_str = device_default

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    print("\n[1] Load metadata")
    metadata = load_full_metadata(dataset_root)
    splits = load_splits(dataset_root)
    n_train = len(splits["train"])
    n_all = len(metadata)

    print(f"[Dataset] all samples = {n_all}")
    print(f"[Dataset] train samples = {n_train}")

    print("\n[2] Load checkpoint")
    ckpt = load_checkpoint(ckpt_path, device)

    cfg = config_from_checkpoint(ckpt, dataset_root, out_dir, str(device))

    posterior_steps_str = ask_string(
        "Posterior fitting steps",
        "Input posterior fitting steps for each sample:",
        initialvalue=str(cfg.posterior_steps)
    )
    posterior_steps = int(posterior_steps_str) if posterior_steps_str else int(cfg.posterior_steps)

    posterior_lr_str = ask_string(
        "Posterior fitting learning rate",
        "Input posterior fitting learning rate:",
        initialvalue=str(cfg.posterior_lr)
    )
    posterior_lr = float(posterior_lr_str) if posterior_lr_str else float(cfg.posterior_lr)

    batch_size_str = ask_string(
        "Batch size",
        "Input batch size for full-sample latent fitting.\nRecommended: 4 or 8 for GPU; 1 or 2 for CPU.",
        initialvalue="4"
    )
    infer_batch_size = int(batch_size_str) if batch_size_str else 4

    iou_threshold_str = ask_string(
        "IoU threshold",
        "Input IoU threshold for retaining samples:",
        initialvalue="0.85"
    )
    iou_threshold = float(iou_threshold_str) if iou_threshold_str else 0.85

    max_samples_str = ask_string(
        "Max samples",
        "Input max samples for test run. Use 0 for all samples.",
        initialvalue="0"
    )
    max_samples = int(max_samples_str) if max_samples_str else 0

    seed_all(cfg.seed)

    print("\n[3] Build model")
    model = build_model_from_checkpoint(ckpt, cfg, n_train, device)
    print(f"[Model] model_type = {cfg.model_type}")
    print(f"[Model] latent_dim = {cfg.latent_dim}")
    print(f"[Device] {device}")

    print("\n[4] Build full dataset")
    dataset = FullFeatureDataset(
        dataset_root=dataset_root,
        metadata=metadata,
        enc_size=cfg.encoder_input_size,
        max_samples=max_samples
    )

    loader = DataLoader(
        dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_full
    )

    rows = []
    z_cols = [f"z_{j:03d}" for j in range(cfg.latent_dim)]

    t0 = time.time()

    print("\n[5] Full-sample latent fitting and reconstruction")
    for batch_idx, batch in enumerate(tqdm(loader, desc="export latent features", dynamic_ncols=True)):
        if cfg.model_type == "vae":
            with torch.no_grad():
                mu, _ = model.encoder(batch["feature_small"].to(device))
                z = mu.detach()
        else:
            z = opt_latent_batch(
                decoder=model.decoder,
                batch=batch,
                cfg=cfg,
                device=device,
                posterior_steps=posterior_steps,
                posterior_lr=posterior_lr
            )

        sdf_pred_batch, mask_prob_batch = dense_batch(
            decoder=model.decoder,
            z=z,
            grid_size=cfg.eval_grid_size,
            device=device,
            chunk_size=cfg.eval_chunk_size
        )

        z_np = z.detach().cpu().numpy()

        B = z_np.shape[0]

        for j in range(B):
            gt = batch["mask"][j, 0].numpy().astype(np.uint8)

            pred = post_process_mask((sdf_pred_batch[j] <= 0).astype(np.uint8))

            if gt.shape != pred.shape:
                gt_eval = cv2.resize(
                    gt,
                    (pred.shape[1], pred.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                gt_eval = gt

            met = compute_metrics(pred, gt_eval)

            row = {
                "sample_name": batch["sample_name"][j],
                "FID": batch["FID"][j],
                "source_id": batch["source_id"][j],
                "source_index": batch["source_index"][j],
                "abcd_class": batch["abcd_class"][j],
                "split": batch["split"][j],
                "feature_file": batch["feature_file"][j],
                **met
            }

            for k, col in enumerate(z_cols):
                row[col] = float(z_np[j, k])

            rows.append(row)

        # Periodic save to avoid losing results in long runs.
        if (batch_idx + 1) % 20 == 0:
            save_tables(rows, out_dir, iou_threshold, cfg.latent_dim)

    print("\n[6] Save final tables")
    all_path, filtered_path = save_tables(rows, out_dir, iou_threshold, cfg.latent_dim)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    retained = int((pd.to_numeric(df["iou"], errors="coerce") > iou_threshold).sum())

    summary = {
        "dataset_root": str(dataset_root),
        "checkpoint": str(ckpt_path),
        "output_root": str(out_dir),
        "model_type": cfg.model_type,
        "latent_dim": int(cfg.latent_dim),
        "n_train_in_checkpoint": int(n_train),
        "n_samples_processed": int(len(df)),
        "n_samples_retained": retained,
        "iou_threshold": float(iou_threshold),
        "posterior_steps": int(posterior_steps),
        "posterior_lr": float(posterior_lr),
        "batch_size": int(infer_batch_size),
        "eval_grid_size": int(cfg.eval_grid_size),
        "points_per_sample": int(cfg.points_per_sample),
        "elapsed_seconds": float(elapsed),
        "output_all": str(all_path),
        "output_filtered": str(filtered_path),
        "note": "Latent vectors were obtained by posterior fitting for all samples using the fixed trained decoder."
    }

    summary_path = out_dir / "latent_feature_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Finished]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    show_info(
        "Full-sample latent feature export finished.\n\n"
        f"Processed samples: {len(df)}\n"
        f"Retained IoU > {iou_threshold}: {retained}\n\n"
        f"Filtered output:\n{filtered_path}"
    )


if __name__ == "__main__":
    main()