#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_full_eval_outlier_class_visuals.py

Purpose
-------
Evaluate the best single-layer field-first model on full train/val/test splits,
produce raw and outlier-filtered metric tables, and generate class-balanced
visualizations containing different label classes.

This script is designed to be placed in the same directory as
02_train_single_layer_field_model_GUI.py, but it also accepts --train-script.

Typical Windows usage
---------------------
python 03_full_eval_outlier_class_visuals.py ^
  --train-script "<Results第五章>/02_train_single_layer_field_model_GUI.py" ^
  --feature-root "D:\...\Results第五章\1" ^
  --checkpoint "D:\...\Results第五章\2\checkpoints\best.pt" ^
  --output-root "D:\...\Results第五章\2\full_eval_cleaned" ^
  --device cuda

Notes
-----
1. If existing full sample_metrics.csv files are found under
   <output-root>/../evaluation/<split>_best or <checkpoint-parent-parent>/evaluation/<split>_best,
   the script can reuse them by default to avoid recomputing all posterior optimization.
2. For new val/test evaluation of an auto-decoder model, each unseen sample requires
   posterior latent optimization, so full evaluation can be slow.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


HIGHER_BETTER = {"iou", "dice"}
LOWER_BETTER = {
    "area_relative_error",
    "centroid_distance_normalized",
    "boundary_chamfer_normalized",
    "boundary_hausdorff_normalized",
    "component_count",
    "hole_count",
    "invalid_topology",
}

DEFAULT_OUTLIER_METRICS = [
    "iou",
    "dice",
    "area_relative_error",
    "centroid_distance_normalized",
    "boundary_chamfer_normalized",
    "boundary_hausdorff_normalized",
]


def import_train_module(path: str):
    path = Path(path).resolve()
    module_name = "single_layer_train_module"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses 在处理动态导入类时会通过 sys.modules[cls.__module__]
    # 查找模块命名空间；必须先注册再 exec_module，否则在部分 Python 版本下会报 NoneType.__dict__。
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_existing_metrics_path(output_root: Path, split: str) -> Path:
    return output_root / "evaluation" / f"{split}_best" / "sample_metrics.csv"


def load_class_map(feature_root: Path, class_column: str = "abcd_class") -> Dict[str, str]:
    desc = feature_root / "metadata" / "shape_descriptors.csv"
    if not desc.exists():
        return {}
    df = pd.read_csv(desc)
    if "sample_name" not in df.columns or class_column not in df.columns:
        return {}
    return dict(zip(df["sample_name"].astype(str), df[class_column].astype(str)))


def expected_split_count(mod, feature_root: Path, split: str) -> int:
    return len(mod.load_splits(str(feature_root))[split])


def metrics_are_full(csv_path: Path, expected_n: int) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
        return len(df) == expected_n and "sample_name" in df.columns and "iou" in df.columns
    except Exception:
        return False


def build_model_from_checkpoint(mod, checkpoint: Path, feature_root: Path, output_root: Path,
                                device_name: str, posterior_steps: int, posterior_lr: float,
                                points_per_sample: int | None, eval_grid_size: int | None,
                                eval_chunk_size: int | None):
    raw = torch.load(str(checkpoint), map_location="cpu")
    cfg_dict = dict(raw["config"])
    cfg_dict["feature_root"] = str(feature_root)
    cfg_dict["output_root"] = str(output_root)
    cfg_dict["device"] = device_name
    cfg_dict["posterior_steps"] = int(posterior_steps)
    cfg_dict["posterior_lr"] = float(posterior_lr)
    if points_per_sample is not None:
        cfg_dict["points_per_sample"] = int(points_per_sample)
    if eval_grid_size is not None:
        cfg_dict["eval_grid_size"] = int(eval_grid_size)
    if eval_chunk_size is not None:
        cfg_dict["eval_chunk_size"] = int(eval_chunk_size)

    cfg = mod.Config(**cfg_dict)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    train_names = mod.load_splits(cfg.feature_root)["train"]
    model, _ = mod.build_model(cfg, len(train_names), device)
    mod.load_ckpt(str(checkpoint), model, None, device)
    model.eval()
    return cfg, model, device


def evaluate_split_full(mod, model, cfg, device, split: str, class_map: Dict[str, str],
                        output_root: Path, max_samples: int = 0) -> pd.DataFrame:
    train_names = mod.load_splits(cfg.feature_root)["train"]
    latent_index = {n: i for i, n in enumerate(train_names)}
    ds = mod.FeatureDataset(cfg.feature_root, split, cfg.encoder_input_size, latent_index,
                            cfg.cache_in_memory, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=mod.collate)

    rows = []
    start = time.time()
    for b in tqdm(loader, desc=f"full eval {split}", dynamic_ncols=True):
        name = str(b["sample_name"][0])
        cls = class_map.get(name, b["abcd_class"][0])
        gt = b["mask"][0, 0].numpy().astype(np.uint8)

        if cfg.model_type == "vae":
            with torch.no_grad():
                mu, _ = model.encoder(b["feature_small"].to(device))
                z = mu
        else:
            if split == "train" and name in latent_index:
                with torch.no_grad():
                    z = model.latent(torch.tensor([latent_index[name]], device=device))
            else:
                z = mod.opt_latent(model.decoder, {"sdf": b["sdf"], "corner": b["corner"], "mask": b["mask"]}, cfg, device)

        d = mod.dense(model.decoder, z, cfg.eval_grid_size, device, cfg.eval_chunk_size)
        pred = mod.post((d["sdf"] <= 0).astype(np.uint8))
        gt2 = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST) if gt.shape != pred.shape else gt

        met = mod.metrics(pred, gt2)
        rows.append({"sample_name": name, "split": split, "label_class": cls, "abcd_class": cls, **met})

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(output_root / "evaluation_raw" / f"{split}_best")
    df.to_csv(out_dir / "sample_metrics_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[Eval] {split}: {len(df)} samples, time={(time.time()-start)/60:.2f} min")
    return df


def add_outlier_flags(df: pd.DataFrame, metrics: List[str], group_mode: str = "split_class",
                      iqr_k: float = 1.5, lower_quantile: float = 0.25,
                      upper_quantile: float = 0.75) -> pd.DataFrame:
    df = df.copy()
    df["is_outlier"] = False
    df["outlier_reasons"] = ""

    if group_mode == "global":
        group_cols = []
    elif group_mode == "split":
        group_cols = ["split"]
    else:
        group_cols = ["split", "label_class"]

    if group_cols:
        groups = list(df.groupby(group_cols, dropna=False).groups.items())
    else:
        groups = [(("all",), df.index)]

    for _, idx in groups:
        sub = df.loc[idx]
        for m in metrics:
            if m not in sub.columns or sub[m].dropna().empty:
                continue
            s = pd.to_numeric(sub[m], errors="coerce")
            q1 = s.quantile(lower_quantile)
            q3 = s.quantile(upper_quantile)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue

            if m in HIGHER_BETTER:
                threshold = q1 - iqr_k * iqr
                mask = s < threshold
                reason = f"{m}<IQR_low({threshold:.6g})"
            else:
                threshold = q3 + iqr_k * iqr
                mask = s > threshold
                reason = f"{m}>IQR_high({threshold:.6g})"

            flagged = sub.index[mask.fillna(False)]
            if len(flagged) > 0:
                df.loc[flagged, "is_outlier"] = True
                old = df.loc[flagged, "outlier_reasons"].fillna("").astype(str)
                df.loc[flagged, "outlier_reasons"] = [
                    (o + ";" + reason).strip(";") for o in old
                ]
    return df


def summarize(df: pd.DataFrame, metrics: List[str], group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if group_cols:
        iterator = df.groupby(group_cols, dropna=False)
    else:
        iterator = [(("all",), df)]

    for key, g in iterator:
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: val for col, val in zip(group_cols, key)}
        base["samples"] = int(len(g))
        for m in metrics:
            if m not in g.columns:
                continue
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append({
                **base,
                "metric": m,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "minimum": float(s.min()),
                "maximum": float(s.max()),
                "p10": float(s.quantile(0.10)),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
            })
    return pd.DataFrame(rows)


def make_manuscript_table(summary_split: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        sub = summary_split[summary_split["metric"] == metric]
        for split in ["train", "val", "test"]:
            r = sub[sub["split"] == split]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "Metric": metric,
                "split": split,
                "Mean": r["mean"],
                "Minimum": r["minimum"],
                "Maximum": r["maximum"],
                "Median": r["median"],
                "P10": r["p10"],
                "P90": r["p90"],
                "Samples_after_filter": int(r["samples"]),
            })
    return pd.DataFrame(rows)


def choose_visual_samples(df_filtered: pd.DataFrame, per_class: int = 3,
                          metric: str = "iou") -> pd.DataFrame:
    choices = []
    for (split, cls), g in df_filtered.groupby(["split", "label_class"], dropna=False):
        if g.empty:
            continue
        gs = g.sort_values(metric, ascending=True).reset_index(drop=True)
        picks = []
        if per_class <= 1:
            picks = [("median", int(len(gs) // 2))]
        elif per_class == 2:
            picks = [("low", 0), ("high", len(gs) - 1)]
        else:
            picks = [("low", 0), ("median", int(len(gs) // 2)), ("high", len(gs) - 1)]
            if per_class >= 4 and len(gs) >= 4:
                picks.append(("q25", int(round((len(gs) - 1) * 0.25))))
            if per_class >= 5 and len(gs) >= 5:
                picks.append(("q75", int(round((len(gs) - 1) * 0.75))))

        used = set()
        for tag, pos in picks:
            pos = max(0, min(pos, len(gs)-1))
            sample = gs.iloc[pos].copy()
            name = sample["sample_name"]
            if name in used:
                continue
            used.add(name)
            sample["visual_rank_tag"] = tag
            choices.append(sample)
    if not choices:
        return pd.DataFrame()
    return pd.DataFrame(choices)


def predict_and_visualize_one(mod, model, cfg, device, split: str, sample_name: str,
                              class_label: str, tag: str, out_path: Path):
    train_names = mod.load_splits(cfg.feature_root)["train"]
    latent_index = {n: i for i, n in enumerate(train_names)}
    ds = mod.FeatureDataset(cfg.feature_root, split, cfg.encoder_input_size, latent_index,
                            cfg.cache_in_memory, max_samples=0)
    name_to_idx = {n: i for i, n in enumerate(ds.names)}
    if sample_name not in name_to_idx:
        print(f"[Warn] sample not found in {split}: {sample_name}")
        return None

    b = mod.collate([ds[name_to_idx[sample_name]]])
    gt = b["mask"][0, 0].numpy().astype(np.uint8)

    if cfg.model_type == "vae":
        with torch.no_grad():
            mu, _ = model.encoder(b["feature_small"].to(device))
            z = mu
    else:
        if split == "train" and sample_name in latent_index:
            with torch.no_grad():
                z = model.latent(torch.tensor([latent_index[sample_name]], device=device))
        else:
            z = mod.opt_latent(model.decoder, {"sdf": b["sdf"], "corner": b["corner"], "mask": b["mask"]}, cfg, device)

    d = mod.dense(model.decoder, z, cfg.eval_grid_size, device, cfg.eval_chunk_size)
    pred = mod.post((d["sdf"] <= 0).astype(np.uint8))
    gt2 = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST) if gt.shape != pred.shape else gt
    met = mod.metrics(pred, gt2)

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample_name))
    safe_cls = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(class_label))
    fname = f"{split}_{safe_cls}_{tag}_{safe_name}_iou{met['iou']:.4f}.png"
    save_path = out_path / split / safe_cls / fname
    title = f"{split} | {sample_name} | {class_label} | {tag} | IoU={met['iou']:.4f}"
    mod.visualize(save_path, gt2, pred, title)
    return save_path


def make_contact_sheet(image_paths: List[Path], output_path: Path, max_cols: int = 3):
    imgs = []
    labels = []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
        labels.append(p.parent.name + " | " + p.name[:60])
    if not imgs:
        return

    cols = min(max_cols, len(imgs))
    rows = int(math.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.0, rows * 2.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, im, lab in zip(axes, imgs, labels):
        ax.imshow(im)
        ax.set_title(lab, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-script", default="02_train_single_layer_field_model_GUI.py")
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--class-column", default="abcd_class", help="Column in metadata/shape_descriptors.csv used as label class.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--posterior-steps", type=int, default=200)
    ap.add_argument("--posterior-lr", type=float, default=2e-2)
    ap.add_argument("--points-per-sample", type=int, default=None)
    ap.add_argument("--eval-grid-size", type=int, default=None)
    ap.add_argument("--eval-chunk-size", type=int, default=None)
    ap.add_argument("--max-samples-per-split", type=int, default=0, help="0 means full split.")
    ap.add_argument("--reuse-metrics-if-full", action="store_true", default=True)
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--outlier-group-mode", choices=["global", "split", "split_class"], default="split_class")
    ap.add_argument("--outlier-iqr-k", type=float, default=1.5)
    ap.add_argument("--outlier-metrics", default=",".join(DEFAULT_OUTLIER_METRICS))
    ap.add_argument("--visuals-per-class", type=int, default=3)
    ap.add_argument("--skip-visuals", action="store_true")
    args = ap.parse_args()

    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = Path.cwd() / train_script
    feature_root = Path(args.feature_root)
    checkpoint = Path(args.checkpoint)
    output_root = ensure_dir(Path(args.output_root))

    mod = import_train_module(str(train_script))
    class_map = load_class_map(feature_root, args.class_column)

    cfg, model, device = build_model_from_checkpoint(
        mod, checkpoint, feature_root, output_root,
        args.device, args.posterior_steps, args.posterior_lr,
        args.points_per_sample, args.eval_grid_size, args.eval_chunk_size
    )

    metrics_to_summarize = [
        "iou",
        "dice",
        "area_relative_error",
        "centroid_distance_normalized",
        "boundary_chamfer_normalized",
        "boundary_hausdorff_normalized",
        "component_count",
        "hole_count",
        "invalid_topology",
    ]

    all_raw = []
    source_eval_root = checkpoint.parent.parent / "evaluation"

    for split in args.splits:
        expected_n = expected_split_count(mod, feature_root, split)
        existing_candidates = [
            output_root / "evaluation_raw" / f"{split}_best" / "sample_metrics_raw.csv",
            output_root / "evaluation" / f"{split}_best" / "sample_metrics.csv",
            source_eval_root / f"{split}_best" / "sample_metrics.csv",
        ]

        df = None
        if args.reuse_metrics_if_full and not args.force_recompute and args.max_samples_per_split == 0:
            for c in existing_candidates:
                if metrics_are_full(c, expected_n):
                    print(f"[Reuse] {split}: {c}")
                    df = pd.read_csv(c)
                    break

        if df is None:
            print(f"[Compute] {split}: full evaluation begins.")
            df = evaluate_split_full(mod, model, cfg, device, split, class_map, output_root, args.max_samples_per_split)

        if "label_class" not in df.columns:
            if "abcd_class" in df.columns:
                df["label_class"] = df["abcd_class"].astype(str)
            else:
                df["label_class"] = df["sample_name"].astype(str).map(class_map).fillna("unknown")
        df["split"] = split
        all_raw.append(df)

    raw = pd.concat(all_raw, ignore_index=True)
    ensure_dir(output_root / "tables")
    raw.to_csv(output_root / "tables" / "sample_metrics_raw_all_splits.csv", index=False, encoding="utf-8-sig")

    outlier_metrics = [m.strip() for m in args.outlier_metrics.split(",") if m.strip()]
    flagged = add_outlier_flags(raw, outlier_metrics, args.outlier_group_mode, args.outlier_iqr_k)
    filtered = flagged[~flagged["is_outlier"]].copy()

    flagged.to_csv(output_root / "tables" / "sample_metrics_with_outlier_flags.csv", index=False, encoding="utf-8-sig")
    filtered.to_csv(output_root / "tables" / "sample_metrics_filtered_all_splits.csv", index=False, encoding="utf-8-sig")

    summary_raw_split = summarize(flagged, metrics_to_summarize, ["split"])
    summary_filtered_split = summarize(filtered, metrics_to_summarize, ["split"])
    summary_raw_class = summarize(flagged, metrics_to_summarize, ["split", "label_class"])
    summary_filtered_class = summarize(filtered, metrics_to_summarize, ["split", "label_class"])

    summary_raw_split.to_csv(output_root / "tables" / "summary_raw_by_split.csv", index=False, encoding="utf-8-sig")
    summary_filtered_split.to_csv(output_root / "tables" / "summary_filtered_by_split.csv", index=False, encoding="utf-8-sig")
    summary_raw_class.to_csv(output_root / "tables" / "summary_raw_by_split_class.csv", index=False, encoding="utf-8-sig")
    summary_filtered_class.to_csv(output_root / "tables" / "summary_filtered_by_split_class.csv", index=False, encoding="utf-8-sig")

    manuscript = make_manuscript_table(summary_filtered_split, metrics_to_summarize)
    manuscript.to_csv(output_root / "tables" / "manuscript_metric_table_filtered_mean_min_max.csv", index=False, encoding="utf-8-sig")

    counts = flagged.groupby(["split", "label_class", "is_outlier"]).size().reset_index(name="count")
    counts.to_csv(output_root / "tables" / "outlier_counts_by_split_class.csv", index=False, encoding="utf-8-sig")

    meta = {
        "feature_root": str(feature_root),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "splits": args.splits,
        "class_column": args.class_column,
        "outlier_group_mode": args.outlier_group_mode,
        "outlier_iqr_k": args.outlier_iqr_k,
        "outlier_metrics": outlier_metrics,
        "raw_samples": int(len(flagged)),
        "filtered_samples": int(len(filtered)),
        "outlier_samples": int(flagged["is_outlier"].sum()),
    }
    write_json(output_root / "tables" / "evaluation_cleaning_metadata.json", meta)

    if not args.skip_visuals and args.visuals_per_class > 0:
        selections = choose_visual_samples(filtered, args.visuals_per_class, "iou")
        selections.to_csv(output_root / "tables" / "visual_selected_samples_by_class.csv", index=False, encoding="utf-8-sig")

        vroot = ensure_dir(output_root / "visuals_by_label_class")
        image_paths = []
        for _, r in tqdm(selections.iterrows(), total=len(selections), desc="visuals by class", dynamic_ncols=True):
            p = predict_and_visualize_one(
                mod, model, cfg, device,
                str(r["split"]),
                str(r["sample_name"]),
                str(r["label_class"]),
                str(r["visual_rank_tag"]),
                vroot
            )
            if p is not None:
                image_paths.append(Path(p))

        for split in args.splits:
            split_imgs = [p for p in image_paths if p.name.startswith(split + "_")]
            if split_imgs:
                make_contact_sheet(split_imgs, output_root / "visuals_by_label_class" / f"{split}_class_representatives_contact_sheet.png")

    print("\n[Done] Full evaluation and outlier-filtered summaries are saved to:")
    print(output_root)
    print("\nKey outputs:")
    print(output_root / "tables" / "sample_metrics_raw_all_splits.csv")
    print(output_root / "tables" / "sample_metrics_filtered_all_splits.csv")
    print(output_root / "tables" / "summary_filtered_by_split.csv")
    print(output_root / "tables" / "summary_filtered_by_split_class.csv")
    print(output_root / "tables" / "manuscript_metric_table_filtered_mean_min_max.csv")
    print(output_root / "visuals_by_label_class")


if __name__ == "__main__":
    main()
