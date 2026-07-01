#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A implementation for layered polygon JSON datasets.

Current output policy
---------------------
1. Numerical Stage A features, metadata, reconstructed polygons, and evaluation reports are retained.
2. Each sample exports one combined SDF-corner-mask figure in ``feature_visuals``.
3. Each sample exports only one self-check figure in ``selfcheck_visuals``:
       <sample>_selfcheck_overlay_advanced.png
4. The self-check figure contains light reference fills, dark dashed reference boundaries,
   layer-coloured reconstructed boundaries, and the mean IoU, Dice, Chamfer, and Hausdorff values.
5. It does not contain cyan/orange error regions, a local zoom inset, maximum-deviation
   markers, pixel-distance annotations, or legacy overlay images.
6. Publication figures are written at 600 dpi with Times New Roman typography.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10

SCRIPT_VERSION = "2026-06-07-advanced-only-selfcheck-v3"

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None


@dataclass
class Config:
    grid_width: int = 512
    grid_height: int = 512
    canvas_padding: int = 16
    sdf_clip_value: float = 64.0
    corner_sigma: float = 2.5
    corner_angle_threshold_deg: float = 140.0
    corner_polygon_simplify_ratio: float = 0.01
    min_polygon_area_pixels: float = 3.0
    contour_approx_epsilon_ratio: float = 0.002
    layer_order_preference: Tuple[str, ...] = ("top", "middle", "bottom")
    overlay_boundary_thickness: int = 1
    export_dpi: Tuple[int, int] = (600, 600)
    visual_crop_margin: int = 20


LAYER_COLOR_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "top": (0, 0, 255),
    "middle": (0, 255, 0),
    "bottom": (255, 0, 0),
}

LIGHT_BG_RGB = (0.94, 0.94, 0.94)


def select_folder(dialog_title: str) -> str:
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available. Please use --input_dir and --output_dir.")
    root = tk.Tk()
    root.withdraw()
    root.update()
    folder = filedialog.askdirectory(title=dialog_title)
    root.destroy()
    return folder


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_png_with_dpi(path: str, image_bgr: np.ndarray, dpi: Tuple[int, int]) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(path, dpi=dpi)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sanitize_sample_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.endswith("_layer_polygons"):
        base = base[: -len("_layer_polygons")]
    return base


def parse_layers(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_layers = data.get("layers", {})
    parsed: List[Dict[str, Any]] = []

    if isinstance(raw_layers, dict):
        for key, layer_obj in raw_layers.items():
            if not isinstance(layer_obj, dict):
                continue
            parsed.append(
                {
                    "layer_id": key,
                    "color_name": layer_obj.get("color_name", key),
                    "vertical_rank": layer_obj.get("vertical_rank", None),
                    "polygons": layer_obj.get("polygons", []),
                    "polygon_count": layer_obj.get("polygon_count", len(layer_obj.get("polygons", []))),
                }
            )
    elif isinstance(raw_layers, list):
        for idx, layer_obj in enumerate(raw_layers):
            if not isinstance(layer_obj, dict):
                continue
            layer_name = layer_obj.get("name", layer_obj.get("layer_id", f"layer_{idx+1}"))
            parsed.append(
                {
                    "layer_id": layer_name,
                    "color_name": layer_obj.get("color_name", layer_name),
                    "vertical_rank": layer_obj.get("vertical_rank", None),
                    "polygons": layer_obj.get("polygons", []),
                    "polygon_count": layer_obj.get("polygon_count", len(layer_obj.get("polygons", []))),
                }
            )

    rank_order = {"top": 0, "middle": 1, "bottom": 2}
    if any(layer.get("vertical_rank") in rank_order for layer in parsed):
        parsed.sort(key=lambda d: rank_order.get(str(d.get("vertical_rank", "")).lower(), 999))
    return parsed


def extract_canvas_size(data: Dict[str, Any]) -> Tuple[int, int]:
    canvas = data.get("canvas_size", {})
    width = int(canvas.get("width", 0))
    height = int(canvas.get("height", 0))
    if width > 0 and height > 0:
        return width, height

    x_vals, y_vals = [], []
    for layer in parse_layers(data):
        for poly in layer["polygons"]:
            for pt in poly.get("points", []):
                x_vals.append(float(pt[0]))
                y_vals.append(float(pt[1]))
    if not x_vals or not y_vals:
        return 1024, 512
    return int(math.ceil(max(x_vals))) + 1, int(math.ceil(max(y_vals))) + 1


def compute_normalization_transform(orig_width: int, orig_height: int, target_width: int, target_height: int, padding: int) -> Dict[str, float]:
    available_w = max(1, target_width - 2 * padding)
    available_h = max(1, target_height - 2 * padding)
    scale = min(available_w / max(orig_width - 1, 1), available_h / max(orig_height - 1, 1))
    used_w = (orig_width - 1) * scale
    used_h = (orig_height - 1) * scale
    offset_x = (target_width - used_w) / 2.0
    offset_y = (target_height - used_h) / 2.0
    return {
        "scale": float(scale),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
        "orig_width": int(orig_width),
        "orig_height": int(orig_height),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "padding": int(padding),
    }


def transform_points(points: List[List[float]], transform: Dict[str, float]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0] * transform["scale"] + transform["offset_x"]
    y = pts[:, 1] * transform["scale"] + transform["offset_y"]
    return np.stack([x, y], axis=1).astype(np.float32)


def inverse_transform_points(points: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    x = (points[:, 0] - transform["offset_x"]) / transform["scale"]
    y = (points[:, 1] - transform["offset_y"]) / transform["scale"]
    return np.stack([x, y], axis=1).astype(np.float32)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def rasterize_polygons(polygons: List[np.ndarray], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    int_polys = []
    for poly in polygons:
        if poly.shape[0] < 3:
            continue
        int_poly = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        int_polys.append(int_poly)
    if int_polys:
        cv2.fillPoly(mask, int_polys, 255)
    return mask


def compute_signed_distance(mask: np.ndarray, clip_value: float) -> np.ndarray:
    inside = (mask > 0).astype(np.uint8)
    outside = 1 - inside
    dist_inside = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    sdf = dist_outside.astype(np.float32)
    sdf[inside > 0] = -dist_inside[inside > 0]
    sdf = np.clip(sdf, -clip_value, clip_value)
    return sdf.astype(np.float32)


def normalize_sdf(sdf: np.ndarray, clip_value: float) -> np.ndarray:
    return np.clip(sdf / float(clip_value), -1.0, 1.0).astype(np.float32)


def simplify_polygon(points: np.ndarray, epsilon_ratio: float) -> np.ndarray:
    if len(points) < 3:
        return points
    contour = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx[:, 0, :].astype(np.float32)


def vertex_turn_angle(prev_pt: np.ndarray, pt: np.ndarray, next_pt: np.ndarray) -> float:
    v1 = prev_pt - pt
    v2 = next_pt - pt
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    angle = math.degrees(math.acos(cosang))
    return angle


def detect_corner_points(polygons: List[np.ndarray], cfg: Config) -> np.ndarray:
    corners: List[np.ndarray] = []
    angle_threshold = cfg.corner_angle_threshold_deg
    for poly in polygons:
        simp = simplify_polygon(poly, cfg.corner_polygon_simplify_ratio)
        if len(simp) < 3:
            continue
        n = len(simp)
        for i in range(n):
            prev_pt = simp[(i - 1) % n]
            pt = simp[i]
            next_pt = simp[(i + 1) % n]
            angle = vertex_turn_angle(prev_pt, pt, next_pt)
            if angle <= angle_threshold:
                corners.append(pt.copy())
    if not corners:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(corners, dtype=np.float32)


def compute_corner_field(corners: np.ndarray, width: int, height: int, sigma: float) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float32)
    if corners.size == 0:
        return field
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    inv_2sigma2 = 1.0 / max(2.0 * sigma * sigma, 1e-6)
    for x, y in corners:
        radius = int(max(3.0 * sigma, 3))
        x0 = max(0, int(round(x)) - radius)
        x1 = min(width - 1, int(round(x)) + radius)
        y0 = max(0, int(round(y)) - radius)
        y1 = min(height - 1, int(round(y)) + radius)
        if x1 < x0 or y1 < y0:
            continue
        sub_xx = xx[y0:y1 + 1, x0:x1 + 1]
        sub_yy = yy[y0:y1 + 1, x0:x1 + 1]
        gauss = np.exp(-((sub_xx - x) ** 2 + (sub_yy - y) ** 2) * inv_2sigma2)
        field[y0:y1 + 1, x0:x1 + 1] = np.maximum(field[y0:y1 + 1, x0:x1 + 1], gauss)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def boundary_mask_from_binary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask_u8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 255, 1)
    return (boundary > 0).astype(np.uint8)


def contour_points_from_mask(mask: np.ndarray, epsilon_ratio: float, min_area: float) -> List[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[np.ndarray] = []
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < min_area:
            continue
        perimeter = max(cv2.arcLength(cnt, True), 1.0)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx[:, 0, :].astype(np.float32)
        if len(pts) >= 3:
            polygons.append(pts)
    polygons.sort(key=lambda p: polygon_area(p), reverse=True)
    return polygons


def centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    denom = (mask_a > 0).sum() + (mask_b > 0).sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


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
    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return float(dist / max(diagonal, 1e-6))


def compute_boundary_distances(mask_a: np.ndarray, mask_b: np.ndarray, diagonal: float) -> Tuple[float, float]:
    b1 = boundary_mask_from_binary(mask_a)
    b2 = boundary_mask_from_binary(mask_b)
    n1 = int(b1.sum())
    n2 = int(b2.sum())
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
    return chamfer, hausdorff


def reconstruct_masks_from_sdf(sdf_stack: np.ndarray) -> np.ndarray:
    return (sdf_stack <= 0).astype(np.uint8)


def polygons_to_json_dict(polygons_by_layer: List[List[np.ndarray]], layer_names: List[str], vertical_ranks: List[str],
                          transform: Dict[str, float], source_image: str, grid_width: int, grid_height: int,
                          use_original_coordinates: bool) -> Dict[str, Any]:
    out_layers: Dict[str, Any] = {}
    for idx, polygons in enumerate(polygons_by_layer):
        layer_name = layer_names[idx]
        vertical_rank = vertical_ranks[idx]
        serialized_polys = []
        for poly in polygons:
            export_poly = inverse_transform_points(poly, transform) if use_original_coordinates else poly
            serialized_polys.append({"area_pixels": float(polygon_area(poly)),
                                     "points": [[float(p[0]), float(p[1])] for p in export_poly]})
        out_layers[layer_name] = {
            "vertical_rank": vertical_rank,
            "polygon_count": len(serialized_polys),
            "polygons": serialized_polys,
        }

    coord_desc = {
        "type": "original_canvas" if use_original_coordinates else "normalized_grid",
        "origin": "top-left",
        "x_direction": "right",
        "y_direction": "down",
    }

    return {
        "source_image": source_image,
        "canvas_size": (
            {"width": int(grid_width), "height": int(grid_height)}
            if not use_original_coordinates else
            {"width": int(transform["orig_width"]), "height": int(transform["orig_height"])}
        ),
        "coordinate_system": coord_desc,
        "normalization_transform": transform,
        "layers": out_layers,
    }


def color_from_layer_name(layer_name: str, vertical_rank: str) -> Tuple[int, int, int]:
    lname = str(layer_name).lower()
    vrank = str(vertical_rank).lower()
    if lname in LAYER_COLOR_BGR:
        return LAYER_COLOR_BGR[lname]
    if vrank in LAYER_COLOR_BGR:
        return LAYER_COLOR_BGR[vrank]
    return (150, 150, 150)


def bgr_to_rgb01(color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    b, g, r = color_bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def darken_rgb(rgb: Tuple[float, float, float], factor: float = 0.6) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


def display_label(layer_name: str, vertical_rank: str) -> str:
    vrank = str(vertical_rank).lower()
    if vrank == "top":
        return "Top layer"
    if vrank == "middle":
        return "Middle layer"
    if vrank == "bottom":
        return "Bottom layer"
    return str(layer_name)


def legend_panel_off(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_facecolor("white")


def add_note_text(ax: plt.Axes, lines: List[str], x: float = 0.02, y: float = 0.20, fontsize: int = 10) -> None:
    note = "\n".join(lines)
    ax.text(x, y, note, transform=ax.transAxes, va="top", ha="left", fontsize=fontsize)


def get_boundary_from_mask(mask_crop: np.ndarray) -> np.ndarray:
    boundary = boundary_mask_from_binary((mask_crop > 0.5).astype(np.uint8))
    return boundary


def crop_bounds_from_union(masks: List[np.ndarray], margin: int = 20) -> Tuple[int, int, int, int]:
    union = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        union = np.maximum(union, m.astype(np.uint8))
    ys, xs = np.where(union > 0)
    if len(xs) == 0:
        return 0, union.shape[0], 0, union.shape[1]
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(union.shape[0], int(ys.max()) + margin + 1)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(union.shape[1], int(xs.max()) + margin + 1)
    return y1, y2, x1, x2


def crop_array(arr: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    y1, y2, x1, x2 = bounds
    if arr.ndim == 2:
        return arr[y1:y2, x1:x2]
    return arr[y1:y2, x1:x2, ...]


def save_matplotlib_figure(fig: plt.Figure, path: str, dpi: Tuple[int, int]) -> None:
    """Save a publication figure and always release its memory."""
    try:
        fig.savefig(
            path,
            dpi=dpi[0],
            facecolor="white",
            bbox_inches=None,
            pad_inches=0.02,
        )
    finally:
        plt.close(fig)


def export_feature_grid_3x3(out_path: str, sdf_norm_stack: np.ndarray, corner_stack: np.ndarray, mask_stack: np.ndarray,
                            layer_names: List[str], vertical_ranks: List[str], dpi: Tuple[int, int]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor("white")

    for i in range(3):
        ax = axes[0, i]
        im = ax.imshow(sdf_norm_stack[i], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
        ax.set_title(f"SDF ({layer_names[i]})", fontsize=10)
        ax.axis("off")

    for i in range(3):
        ax = axes[1, i]
        ax.imshow(corner_stack[i], cmap="hot", vmin=0, vmax=1, origin="upper")
        ax.set_title(f"Corner ({layer_names[i]})", fontsize=10)
        ax.axis("off")

    for i in range(3):
        ax = axes[2, i]
        rgb = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        canvas = np.ones((*mask_stack[i].shape, 3), dtype=float) * 0.92
        canvas[mask_stack[i] > 0.5] = rgb
        ax.imshow(canvas, origin="upper")
        ax.set_title(f"Mask ({layer_names[i]})", fontsize=10)
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Normalized signed distance", fontsize=10)
    fig.tight_layout()
    save_matplotlib_figure(fig, out_path, dpi)


def export_sdf_triptych(out_path: str, sdf_norm_stack: np.ndarray, layer_names: List[str], vertical_ranks: List[str],
                        bounds: Tuple[int, int, int, int], dpi: Tuple[int, int]) -> None:
    fig = plt.figure(figsize=(15.6, 4.2), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.18], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    fig.patch.set_facecolor("white")

    im = None
    for i, ax in enumerate(axes):
        sdf_crop = crop_array(sdf_norm_stack[i], bounds)
        im = ax.imshow(sdf_crop, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
        ax.contour(sdf_crop, levels=[0.0], colors=["black"], linewidths=[0.9])
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Normalized signed distance")
    cbar.ax.set_title("SDF", pad=8)
    cbar.ax.tick_params(length=3)

    save_matplotlib_figure(fig, out_path, dpi)


def export_corner_triptych(out_path: str, corner_points_by_layer: List[np.ndarray], mask_stack: np.ndarray,
                           layer_names: List[str], vertical_ranks: List[str], bounds: Tuple[int, int, int, int],
                           dpi: Tuple[int, int]) -> None:
    fig = plt.figure(figsize=(16.4, 4.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(gs[0, 3])
    fig.patch.set_facecolor("white")
    y1, y2, x1, x2 = bounds

    corner_color = "crimson"
    boundary_color = "#404040"

    for i, ax in enumerate(axes):
        ax.set_facecolor(LIGHT_BG_RGB)
        mask_crop = crop_array(mask_stack[i], bounds)
        rgb = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        rgba = np.zeros((*mask_crop.shape, 4), dtype=float)
        rgba[..., :3] = rgb
        rgba[..., 3] = 0.35 * (mask_crop > 0.5).astype(float)
        ax.imshow(rgba, origin="upper")

        boundary = get_boundary_from_mask(mask_crop)
        boundary_rgba = np.zeros((*boundary.shape, 4), dtype=float)
        boundary_rgba[..., :3] = matplotlib.colors.to_rgb(boundary_color)
        boundary_rgba[..., 3] = (boundary > 0).astype(float)
        ax.imshow(boundary_rgba, origin="upper", zorder=2)

        corners = corner_points_by_layer[i]
        if corners.size > 0:
            x = corners[:, 0] - x1
            y = corners[:, 1] - y1
            ax.scatter(x, y, s=22, c=corner_color, edgecolors="white", linewidths=0.6, zorder=3)

        ax.set_xticks([])
        ax.set_yticks([])

    legend_panel_off(legend_ax)
    layer_patches = [
        mpatches.Patch(facecolor=bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i])),
                       alpha=0.35, edgecolor="none", label=display_label(layer_names[i], vertical_ranks[i]))
        for i in range(3)
    ]
    boundary_handle = mlines.Line2D([], [], color=boundary_color, linewidth=1.2, label="Layer boundary")
    point_handle = mlines.Line2D([], [], color=corner_color, marker="o", linestyle="None", markersize=6,
                                 markeredgecolor="white", markeredgewidth=0.6, label="Corner points")
    legend_ax.legend(handles=layer_patches + [boundary_handle, point_handle],
                     loc="upper left", frameon=False, borderaxespad=0.0, handlelength=1.8)
    add_note_text(legend_ax, [
        "Fill: occupancy mask",
        "Solid line: extracted boundary",
        "Points are projected on the",
        "corresponding layer boundary"
    ], y=0.43, fontsize=10)

    save_matplotlib_figure(fig, out_path, dpi)


def export_mask_triptych(out_path: str, mask_stack: np.ndarray, layer_names: List[str], vertical_ranks: List[str],
                         bounds: Tuple[int, int, int, int], dpi: Tuple[int, int]) -> None:
    fig = plt.figure(figsize=(16.4, 4.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(gs[0, 3])
    fig.patch.set_facecolor("white")
    boundary_color = "#404040"

    for i, ax in enumerate(axes):
        ax.set_facecolor(LIGHT_BG_RGB)
        mask_crop = crop_array(mask_stack[i], bounds)
        rgb = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        canvas = np.ones((*mask_crop.shape, 3), dtype=float) * 0.95
        canvas[mask_crop > 0.5] = rgb
        ax.imshow(canvas, origin="upper")
        boundary = get_boundary_from_mask(mask_crop)
        boundary_rgba = np.zeros((*boundary.shape, 4), dtype=float)
        boundary_rgba[..., :3] = matplotlib.colors.to_rgb(boundary_color)
        boundary_rgba[..., 3] = (boundary > 0).astype(float)
        ax.imshow(boundary_rgba, origin="upper", zorder=2)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_panel_off(legend_ax)
    layer_patches = [
        mpatches.Patch(facecolor=bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i])),
                       edgecolor="none", label=display_label(layer_names[i], vertical_ranks[i]))
        for i in range(3)
    ]
    boundary_handle = mlines.Line2D([], [], color=boundary_color, linewidth=1.2, label="Layer boundary")
    legend_ax.legend(handles=layer_patches + [boundary_handle],
                     loc="upper left", frameon=False, borderaxespad=0.0, handlelength=1.8)
    add_note_text(legend_ax, [
        "Mask = binary support region",
        "Colors indicate top, middle,",
        "and bottom layers, respectively"
    ], y=0.47, fontsize=10)

    save_matplotlib_figure(fig, out_path, dpi)


def export_layerwise_composite_triptych(out_path: str, sdf_norm_stack: np.ndarray, corner_points_by_layer: List[np.ndarray],
                                        mask_stack: np.ndarray, layer_names: List[str], vertical_ranks: List[str],
                                        bounds: Tuple[int, int, int, int], dpi: Tuple[int, int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    y1, y2, x1, x2 = bounds

    for i, ax in enumerate(axes):
        ax.set_facecolor(LIGHT_BG_RGB)
        mask_crop = crop_array(mask_stack[i], bounds)
        sdf_crop = crop_array(sdf_norm_stack[i], bounds)
        base = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        line_color = darken_rgb(base, 0.55)

        rgba = np.zeros((*mask_crop.shape, 4), dtype=float)
        rgba[..., :3] = base
        rgba[..., 3] = 0.55 * (mask_crop > 0.5).astype(float)
        ax.imshow(rgba, origin="upper")

        ax.contour(sdf_crop, levels=[-0.5, 0.0, 0.5],
                   colors=[line_color, "black", line_color],
                   linewidths=[0.7, 1.0, 0.7],
                   linestyles=["dashed", "solid", "dotted"])

        corners = corner_points_by_layer[i]
        if corners.size > 0:
            x = corners[:, 0] - x1
            y = corners[:, 1] - y1
            ax.scatter(x, y, s=18, c="black", edgecolors="white", linewidths=0.45, zorder=3)

        ax.set_title(display_label(layer_names[i], vertical_ranks[i]), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        mpatches.Patch(facecolor="gray", alpha=0.55, label="Occupancy fill"),
        mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.0, label="SDF zero-level boundary"),
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=5,
                      markeredgecolor="white", markeredgewidth=0.45, label="Detected corners"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    save_matplotlib_figure(fig, out_path, dpi)


def export_all_modalities_overlay(out_path: str, sdf_norm_stack: np.ndarray, corner_points_by_layer: List[np.ndarray],
                                  mask_stack: np.ndarray, layer_names: List[str], vertical_ranks: List[str],
                                  bounds: Tuple[int, int, int, int], dpi: Tuple[int, int]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT_BG_RGB)
    y1, y2, x1, x2 = bounds

    # draw fills from bottom to top for visual stability
    order = list(range(len(layer_names)))
    order = sorted(order, key=lambda i: {"bottom": 0, "middle": 1, "top": 2}.get(str(vertical_ranks[i]).lower(), i))

    for i in order:
        mask_crop = crop_array(mask_stack[i], bounds)
        base = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        rgba = np.zeros((*mask_crop.shape, 4), dtype=float)
        rgba[..., :3] = base
        rgba[..., 3] = 0.38 * (mask_crop > 0.5).astype(float)
        ax.imshow(rgba, origin="upper", zorder=1 + i)

    for i in order:
        sdf_crop = crop_array(sdf_norm_stack[i], bounds)
        base = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        line_color = darken_rgb(base, 0.55)
        ax.contour(sdf_crop, levels=[-0.5, 0.0, 0.5],
                   colors=[line_color, "black", line_color],
                   linewidths=[0.55, 0.95, 0.55],
                   linestyles=["dashed", "solid", "dotted"],
                   alpha=0.9, zorder=10 + i)

    for i in order:
        corners = corner_points_by_layer[i]
        if corners.size > 0:
            x = corners[:, 0] - x1
            y = corners[:, 1] - y1
            ax.scatter(x, y, s=14, c="black", edgecolors="white", linewidths=0.35, zorder=20 + i)

    ax.set_title("All-modalities overlay", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    layer_handles = [
        mpatches.Patch(facecolor=bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i])),
                       alpha=0.38, label=display_label(layer_names[i], vertical_ranks[i]))
        for i in range(len(layer_names))
    ]
    explain_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", linewidth=0.95, label="SDF zero-level boundary"),
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=5,
                      markeredgecolor="white", markeredgewidth=0.35, label="Detected corners"),
    ]
    ax.legend(handles=layer_handles + explain_handles, loc="lower left", fontsize=8, frameon=True)
    save_matplotlib_figure(fig, out_path, dpi)



def export_combined_triptych(out_path: str, sdf_norm_stack: np.ndarray, corner_points_by_layer: List[np.ndarray],
                             mask_stack: np.ndarray, layer_names: List[str], vertical_ranks: List[str],
                             bounds: Tuple[int, int, int, int], dpi: Tuple[int, int]) -> None:
    fig = plt.figure(figsize=(12.8, 3.6), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(gs[0, 3])
    fig.patch.set_facecolor("white")

    y1, y2, x1, x2 = bounds
    corner_color = "crimson"
    boundary_linewidth_px = 2

    for i, ax in enumerate(axes):
        sdf_crop = crop_array(sdf_norm_stack[i], bounds)
        mask_crop = crop_array(mask_stack[i], bounds)

        # Base layer: SDF field
        ax.imshow(sdf_crop, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")

        # Layer-specific mask boundary (thickened to twice the previous visual width)
        base_rgb = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        boundary_color = darken_rgb(base_rgb, 0.60)
        boundary = get_boundary_from_mask(mask_crop)
        kernel = np.ones((boundary_linewidth_px * 2 - 1, boundary_linewidth_px * 2 - 1), dtype=np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1)
        boundary_rgba = np.zeros((*boundary.shape, 4), dtype=float)
        boundary_rgba[..., :3] = boundary_color
        boundary_rgba[..., 3] = (boundary > 0).astype(float)
        ax.imshow(boundary_rgba, origin="upper", zorder=2)

        # Corner points
        corners = corner_points_by_layer[i]
        if corners.size > 0:
            x = corners[:, 0] - x1
            y = corners[:, 1] - y1
            ax.scatter(x, y, s=28, c=corner_color, edgecolors="white", linewidths=0.7, zorder=3)

        ax.set_xticks([])
        ax.set_yticks([])

    legend_panel_off(legend_ax)

    # Full-height colorbar aligned with the visual height of the left image panels.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(legend_ax, width="10%", height="92%", loc="upper left",
                     bbox_to_anchor=(0.02, 0.04, 0.96, 0.92),
                     bbox_transform=legend_ax.transAxes, borderpad=0.0)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalized SDF", fontsize=20, labelpad=10)
    cbar.ax.tick_params(labelsize=18, length=3)

    boundary_handles = []
    for i in range(3):
        base_rgb = bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        boundary_color = darken_rgb(base_rgb, 0.60)
        label = display_label(layer_names[i], vertical_ranks[i])
        boundary_handles.append(
            mlines.Line2D([], [], color=boundary_color, linewidth=3.2, label=f"{label} boundary")
        )

    corner_handle = mlines.Line2D([], [], color=corner_color, marker="o", linestyle="None", markersize=7,
                                  markeredgecolor="white", markeredgewidth=0.7, label="Corner point")

    legend_ax.legend(handles=boundary_handles + [corner_handle],
                     loc="upper left", bbox_to_anchor=(0.44, 0.98),
                     frameon=False, borderaxespad=0.0, handlelength=2.1,
                     labelspacing=0.9, handletextpad=0.8, fontsize=20)

    save_matplotlib_figure(fig, out_path, dpi)


def lighten_rgb(rgb: Tuple[float, float, float], white_ratio: float = 0.72) -> Tuple[float, float, float]:
    """Mix an RGB color in [0, 1] with white."""
    ratio = float(np.clip(white_ratio, 0.0, 1.0))
    return tuple((1.0 - ratio) * float(c) + ratio for c in rgb)


def alpha_blend_binary_region(base_rgb: np.ndarray,
                              mask: np.ndarray,
                              color_rgb: Tuple[float, float, float],
                              alpha: float) -> np.ndarray:
    """Alpha-blend one binary region onto an RGB float image in [0, 1]."""
    out = base_rgb.copy()
    selector = mask > 0
    if np.any(selector):
        color = np.asarray(color_rgb, dtype=np.float32)
        out[selector] = (1.0 - alpha) * out[selector] + alpha * color
    return out


def rasterize_reconstructed_layers(rec_polygons_by_layer: List[List[np.ndarray]],
                                   width: int,
                                   height: int) -> List[np.ndarray]:
    """Rasterize reconstructed polygon sets into layer-wise binary masks."""
    return [
        (rasterize_polygons(polys, width, height) > 0).astype(np.uint8)
        for polys in rec_polygons_by_layer
    ]


def bbox_from_binary_union(mask: np.ndarray,
                           padding: int = 18,
                           minimum_size: int = 120) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded (y1, y2, x1, x2) box for a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    height, width = mask.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(width, int(xs.max()) + padding + 1)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(height, int(ys.max()) + padding + 1)

    current_w = x2 - x1
    current_h = y2 - y1
    if current_w < minimum_size:
        extra = minimum_size - current_w
        left = extra // 2
        right = extra - left
        x1 = max(0, x1 - left)
        x2 = min(width, x2 + right)
        if x2 - x1 < minimum_size:
            x1 = max(0, x2 - minimum_size)
            x2 = min(width, x1 + minimum_size)
    if current_h < minimum_size:
        extra = minimum_size - current_h
        top = extra // 2
        bottom = extra - top
        y1 = max(0, y1 - top)
        y2 = min(height, y2 + bottom)
        if y2 - y1 < minimum_size:
            y1 = max(0, y2 - minimum_size)
            y2 = min(height, y1 + minimum_size)

    return y1, y2, x1, x2



def draw_reference_and_reconstructed_boundaries(ax: plt.Axes,
                                                crop_bbox: Tuple[int, int, int, int],
                                                gt_masks: List[np.ndarray],
                                                rec_polygons_by_layer: List[List[np.ndarray]],
                                                layer_colors_rgb: List[Tuple[float, float, float]],
                                                reference_linewidth: float = 0.9,
                                                reconstructed_linewidth: float = 1.8) -> None:
    """Draw reconstructed boundaries first and reference boundaries above them."""
    y1, y2, x1, x2 = crop_bbox

    # Reconstructed polygons: white halo followed by layer-coloured line.
    for layer_index, polygons in enumerate(rec_polygons_by_layer):
        line_color = darken_rgb(layer_colors_rgb[layer_index], 0.68)
        for polygon in polygons:
            if polygon is None or len(polygon) < 2:
                continue
            closed = np.vstack([polygon, polygon[0]])
            x = closed[:, 0] - x1
            y = closed[:, 1] - y1
            ax.plot(x, y, color='white', linewidth=reconstructed_linewidth + 1.3, zorder=4)
            ax.plot(x, y, color=line_color, linewidth=reconstructed_linewidth, zorder=5)

    # Reference contours: dark dashed lines drawn last so small offsets remain visible.
    for gt_mask in gt_masks:
        contours, _ = cv2.findContours(
            ((gt_mask > 0).astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        for contour in contours:
            pts = contour[:, 0, :].astype(np.float32)
            ax.plot(
                pts[:, 0] - x1,
                pts[:, 1] - y1,
                color='#303030',
                linewidth=reference_linewidth,
                linestyle=(0, (3.0, 2.0)),
                zorder=6,
            )



def export_advanced_selfcheck_figure(out_path: str,
                                     gt_masks: List[np.ndarray],
                                     rec_polygons_by_layer: List[List[np.ndarray]],
                                     layer_names: List[str],
                                     vertical_ranks: List[str],
                                     per_layer_rows: List[Dict[str, Any]],
                                     dpi: Tuple[int, int] = (600, 600),
                                     crop_margin: int = 18) -> None:
    """
    Export one publication-oriented self-check figure per sample.

    The figure contains only:
    - light layer-wise reference fills;
    - original reference boundaries shown as dark dashed lines;
    - reconstructed boundaries shown as layer-coloured solid lines;
    - mean IoU, Dice, normalized Chamfer, and normalized Hausdorff values.

    It intentionally excludes error-region shading, local zoom windows,
    maximum-deviation markers, and pixel-distance annotations.
    """
    height, width = gt_masks[0].shape[:2]
    rec_masks = rasterize_reconstructed_layers(rec_polygons_by_layer, width, height)
    layer_colors_rgb = [
        bgr_to_rgb01(color_from_layer_name(layer_names[i], vertical_ranks[i]))
        for i in range(len(layer_names))
    ]

    # Light reference fills only; no missed/extra error-region colouring.
    base_rgb = np.ones((height, width, 3), dtype=np.float32)
    for gt_mask, layer_rgb in zip(gt_masks, layer_colors_rgb):
        light_fill = lighten_rgb(layer_rgb, white_ratio=0.74)
        base_rgb = alpha_blend_binary_region(base_rgb, gt_mask, light_fill, alpha=0.72)
    base_rgb = np.clip(base_rgb, 0.0, 1.0)

    # Crop using the union of reference and reconstructed masks.
    full_union = np.zeros((height, width), dtype=np.uint8)
    for gt_mask, rec_mask in zip(gt_masks, rec_masks):
        full_union = np.maximum(full_union, (gt_mask > 0).astype(np.uint8))
        full_union = np.maximum(full_union, (rec_mask > 0).astype(np.uint8))

    main_bbox = bbox_from_binary_union(full_union, padding=crop_margin, minimum_size=180)
    if main_bbox is None:
        main_bbox = (0, height, 0, width)
    y1, y2, x1, x2 = main_bbox

    fig = plt.figure(figsize=(10.8, 5.0), facecolor='white', constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.9, 1.45], wspace=0.08)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    ax_main.imshow(base_rgb[y1:y2, x1:x2], origin='upper')
    draw_reference_and_reconstructed_boundaries(
        ax_main,
        main_bbox,
        gt_masks,
        rec_polygons_by_layer,
        layer_colors_rgb,
        reference_linewidth=1.05,
        reconstructed_linewidth=1.9,
    )
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#303030')

    legend_panel_off(ax_info)
    reference_handle = mlines.Line2D(
        [], [], color='#303030', linewidth=1.35, linestyle=(0, (3.0, 2.0)),
        label='Reference boundary'
    )
    reconstructed_handles = []
    for i in range(len(layer_names)):
        reconstructed_handles.append(
            mlines.Line2D(
                [], [],
                color=darken_rgb(layer_colors_rgb[i], 0.68),
                linewidth=2.4,
                label=f"{display_label(layer_names[i], vertical_ranks[i])} reconstruction",
            )
        )

    ax_info.legend(
        handles=[reference_handle] + reconstructed_handles,
        loc='upper left',
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.4,
        handletextpad=0.8,
        labelspacing=0.85,
        fontsize=10,
    )

    mean_iou = float(np.mean([row['mask_iou'] for row in per_layer_rows]))
    mean_dice = float(np.mean([row['mask_dice'] for row in per_layer_rows]))
    mean_chamfer = float(np.mean([row['boundary_chamfer_normalized'] for row in per_layer_rows]))
    mean_hausdorff = float(np.mean([row['boundary_hausdorff_normalized'] for row in per_layer_rows]))

    metrics_text = (
        f"Mean IoU: {mean_iou:.4f}\n"
        f"Mean Dice: {mean_dice:.4f}\n"
        f"Mean Chamfer: {mean_chamfer:.5f}\n"
        f"Mean Hausdorff: {mean_hausdorff:.5f}"
    )
    ax_info.text(
        0.0, 0.38, metrics_text,
        transform=ax_info.transAxes,
        ha='left', va='top',
        fontsize=10,
        linespacing=1.45,
    )

    fig.savefig(out_path, dpi=dpi[0], bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_numeric_rows(rows: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for metric in metric_names:
        values = []
        for row in rows:
            value = row.get(metric, None)
            if value is None:
                continue
            try:
                values.append(float(value))
            except Exception:
                continue
        if not values:
            summary[metric] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[metric] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }
    return summary


def process_single_json(json_path: str, out_dirs: Dict[str, str], cfg: Config) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = read_json(json_path)
    sample_name = sanitize_sample_name(json_path)
    source_image = data.get("source_image", f"{sample_name}.png")
    layers = parse_layers(data)
    if len(layers) == 0:
        raise ValueError(f"No valid layers were found in file: {json_path}")

    orig_width, orig_height = extract_canvas_size(data)
    transform = compute_normalization_transform(
        orig_width=orig_width,
        orig_height=orig_height,
        target_width=cfg.grid_width,
        target_height=cfg.grid_height,
        padding=cfg.canvas_padding,
    )

    layer_names: List[str] = []
    vertical_ranks: List[str] = []
    gt_polygons_by_layer: List[List[np.ndarray]] = []
    gt_masks: List[np.ndarray] = []
    sdf_list: List[np.ndarray] = []
    sdf_norm_list: List[np.ndarray] = []
    corner_list: List[np.ndarray] = []
    corner_points_by_layer: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    corner_counts: List[int] = []
    original_polygon_counts: List[int] = []

    default_ranks = list(cfg.layer_order_preference)

    for layer_idx, layer in enumerate(layers):
        layer_name = str(layer.get("color_name") or layer.get("layer_id") or f"layer_{layer_idx+1}")
        vertical_rank = str(layer.get("vertical_rank") or (default_ranks[layer_idx] if layer_idx < len(default_ranks) else f"layer_{layer_idx+1}"))
        raw_polygons = layer.get("polygons", [])

        transformed_polygons: List[np.ndarray] = []
        for poly in raw_polygons:
            pts = poly.get("points", []) if isinstance(poly, dict) else poly
            pts_t = transform_points(pts, transform)
            if len(pts_t) >= 3 and polygon_area(pts_t) >= cfg.min_polygon_area_pixels:
                transformed_polygons.append(pts_t)

        gt_mask = rasterize_polygons(transformed_polygons, cfg.grid_width, cfg.grid_height)
        sdf = compute_signed_distance(gt_mask, cfg.sdf_clip_value)
        sdf_norm = normalize_sdf(sdf, cfg.sdf_clip_value)
        corners = detect_corner_points(transformed_polygons, cfg)
        corner_field = compute_corner_field(corners, cfg.grid_width, cfg.grid_height, cfg.corner_sigma)

        layer_names.append(layer_name)
        vertical_ranks.append(vertical_rank)
        gt_polygons_by_layer.append(transformed_polygons)
        gt_masks.append((gt_mask > 0).astype(np.uint8))
        sdf_list.append(sdf)
        sdf_norm_list.append(sdf_norm)
        corner_list.append(corner_field)
        corner_points_by_layer.append(corners)
        mask_list.append((gt_mask > 0).astype(np.float32))
        corner_counts.append(int(corners.shape[0]))
        original_polygon_counts.append(int(len(transformed_polygons)))

    sdf_stack = np.stack(sdf_list, axis=0).astype(np.float32)
    sdf_norm_stack = np.stack(sdf_norm_list, axis=0).astype(np.float32)
    corner_stack = np.stack(corner_list, axis=0).astype(np.float32)
    mask_stack = np.stack(mask_list, axis=0).astype(np.float32)
    feature_tensor = np.concatenate([sdf_norm_stack, corner_stack, mask_stack], axis=0).astype(np.float32)

    feature_path = os.path.join(out_dirs["features"], f"{sample_name}_features.npz")
    np.savez_compressed(
        feature_path,
        feature_tensor=feature_tensor,
        sdf=sdf_stack,
        sdf_normalized=sdf_norm_stack,
        corner_field=corner_stack,
        mask=mask_stack,
        layer_names=np.asarray(layer_names),
        vertical_ranks=np.asarray(vertical_ranks),
        source_image=source_image,
        transform=json.dumps(transform),
    )

    metadata = {
        "sample_name": sample_name,
        "source_image": source_image,
        "input_json": os.path.basename(json_path),
        "feature_file": os.path.basename(feature_path),
        "grid_size": {"width": cfg.grid_width, "height": cfg.grid_height},
        "normalization_transform": transform,
        "channels": {
            "feature_tensor": [f"sdf_norm_{name}" for name in layer_names] +
                              [f"corner_field_{name}" for name in layer_names] +
                              [f"mask_{name}" for name in layer_names]
        },
        "layers": [
            {
                "layer_name": layer_names[i],
                "vertical_rank": vertical_ranks[i],
                "input_polygon_count": original_polygon_counts[i],
                "corner_count": corner_counts[i],
            }
            for i in range(len(layer_names))
        ],
        "config": asdict(cfg),
    }
    write_json(os.path.join(out_dirs["metadata"], f"{sample_name}_metadata.json"), metadata)

    # visualization outputs
    bounds = crop_bounds_from_union(gt_masks, margin=cfg.visual_crop_margin)

    feature_vis_dir = out_dirs["feature_visuals"]
    export_combined_triptych(
        out_path=os.path.join(feature_vis_dir, f"{sample_name}_combined_triptych.png"),
        sdf_norm_stack=sdf_norm_stack,
        corner_points_by_layer=corner_points_by_layer,
        mask_stack=mask_stack,
        layer_names=layer_names,
        vertical_ranks=vertical_ranks,
        bounds=bounds,
        dpi=cfg.export_dpi,
    )

    # no-learning self-check
    reconstructed_masks = reconstruct_masks_from_sdf(sdf_stack)
    rec_polygons_by_layer: List[List[np.ndarray]] = []
    per_layer_rows: List[Dict[str, Any]] = []
    diagonal = float(math.hypot(cfg.grid_width, cfg.grid_height))

    for i, layer_name in enumerate(layer_names):
        rec_mask = (reconstructed_masks[i] > 0).astype(np.uint8)
        rec_polygons = contour_points_from_mask(
            rec_mask,
            epsilon_ratio=cfg.contour_approx_epsilon_ratio,
            min_area=cfg.min_polygon_area_pixels,
        )
        rec_polygons_by_layer.append(rec_polygons)

        rec_mask_from_polys = rasterize_polygons(rec_polygons, cfg.grid_width, cfg.grid_height)
        gt_mask_u8 = (gt_masks[i] > 0).astype(np.uint8)
        rec_mask_u8 = (rec_mask_from_polys > 0).astype(np.uint8)

        iou = compute_iou(gt_mask_u8, rec_mask_u8)
        dice = compute_dice(gt_mask_u8, rec_mask_u8)
        area_rel_err = compute_area_relative_error(gt_mask_u8, rec_mask_u8)
        centroid_dist_norm = compute_centroid_distance(gt_mask_u8, rec_mask_u8, diagonal)
        chamfer_norm, hausdorff_norm = compute_boundary_distances(gt_mask_u8, rec_mask_u8, diagonal)

        per_layer_rows.append({
            "sample_name": sample_name,
            "source_image": source_image,
            "layer_name": layer_name,
            "vertical_rank": vertical_ranks[i],
            "input_polygon_count": int(original_polygon_counts[i]),
            "reconstructed_polygon_count": int(len(rec_polygons)),
            "polygon_count_abs_diff": int(abs(original_polygon_counts[i] - len(rec_polygons))),
            "corner_count": int(corner_counts[i]),
            "mask_iou": float(iou),
            "mask_dice": float(dice),
            "area_relative_error": float(area_rel_err),
            "centroid_distance_normalized": float(centroid_dist_norm),
            "boundary_chamfer_normalized": float(chamfer_norm),
            "boundary_hausdorff_normalized": float(hausdorff_norm),
        })

    reconstructed_json = polygons_to_json_dict(
        polygons_by_layer=rec_polygons_by_layer,
        layer_names=layer_names,
        vertical_ranks=vertical_ranks,
        transform=transform,
        source_image=source_image,
        grid_width=cfg.grid_width,
        grid_height=cfg.grid_height,
        use_original_coordinates=True,
    )
    write_json(os.path.join(out_dirs["selfcheck_polygons"], f"{sample_name}_reconstructed_from_sdf.json"), reconstructed_json)

    # ------------------------------------------------------------------
    # Self-check visualization: advanced-only mode (one PNG per sample).
    # Remove stale legacy images from earlier runs, then export only the
    # publication-oriented advanced comparison figure.
    # ------------------------------------------------------------------
    legacy_selfcheck_paths = [
        os.path.join(out_dirs["selfcheck_visuals"], f"{sample_name}_selfcheck_overlay.png"),
        os.path.join(out_dirs["selfcheck_visuals"], f"{sample_name}_selfcheck_overlay_cropped.png"),
    ]
    for legacy_path in legacy_selfcheck_paths:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    advanced_overlay_path = os.path.join(
        out_dirs["selfcheck_visuals"],
        f"{sample_name}_selfcheck_overlay_advanced.png",
    )
    export_advanced_selfcheck_figure(
        out_path=advanced_overlay_path,
        gt_masks=gt_masks,
        rec_polygons_by_layer=rec_polygons_by_layer,
        layer_names=layer_names,
        vertical_ranks=vertical_ranks,
        per_layer_rows=per_layer_rows,
        dpi=cfg.export_dpi,
        crop_margin=18,
    )

    sample_summary = {
        "sample_name": sample_name,
        "source_image": source_image,
        "feature_file": os.path.basename(feature_path),
        "layer_count": len(layer_names),
        "metrics_mean": {
            "mask_iou": float(np.mean([r["mask_iou"] for r in per_layer_rows])),
            "mask_dice": float(np.mean([r["mask_dice"] for r in per_layer_rows])),
            "area_relative_error": float(np.mean([r["area_relative_error"] for r in per_layer_rows])),
            "centroid_distance_normalized": float(np.mean([r["centroid_distance_normalized"] for r in per_layer_rows])),
            "boundary_chamfer_normalized": float(np.mean([r["boundary_chamfer_normalized"] for r in per_layer_rows])),
            "boundary_hausdorff_normalized": float(np.mean([r["boundary_hausdorff_normalized"] for r in per_layer_rows])),
            "polygon_count_abs_diff": float(np.mean([r["polygon_count_abs_diff"] for r in per_layer_rows])),
        },
    }

    return per_layer_rows, sample_summary


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    print("=" * 78)
    print(f"Running script: {os.path.abspath(__file__)}")
    print(f"Stage A script version: {SCRIPT_VERSION}")
    print("Self-check mode: advanced-only; one *_selfcheck_overlay_advanced.png per sample.")
    print("Legacy *_selfcheck_overlay.png and *_selfcheck_overlay_cropped.png are not exported.")
    print("=" * 78)

    parser = argparse.ArgumentParser(description="Build a Stage A feature database from layered polygon JSON files.")
    parser.add_argument("--input_dir", type=str, default=None, help="Folder containing *_layer_polygons.json files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder for saving Stage A outputs.")
    parser.add_argument("--grid_width", type=int, default=512, help="Normalized grid width.")
    parser.add_argument("--grid_height", type=int, default=512, help="Normalized grid height.")
    parser.add_argument("--padding", type=int, default=16, help="Padding on the normalized grid.")
    parser.add_argument("--sdf_clip", type=float, default=64.0, help="SDF clipping value.")
    parser.add_argument("--corner_sigma", type=float, default=2.5, help="Gaussian sigma for the corner field.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir:
        print("Please select the input folder that contains *_layer_polygons.json files.")
        input_dir = select_folder("Select the input folder containing *_layer_polygons.json files")
    if not output_dir:
        print("Please select the output folder for the Stage A feature database.")
        output_dir = select_folder("Select the output folder for Stage A results")

    if not input_dir or not os.path.isdir(input_dir):
        raise FileNotFoundError("A valid input folder was not selected.")
    if not output_dir:
        raise FileNotFoundError("A valid output folder was not selected.")

    cfg = Config(
        grid_width=int(args.grid_width),
        grid_height=int(args.grid_height),
        canvas_padding=int(args.padding),
        sdf_clip_value=float(args.sdf_clip),
        corner_sigma=float(args.corner_sigma),
    )

    out_dirs = {
        "features": os.path.join(output_dir, "features"),
        "metadata": os.path.join(output_dir, "metadata"),
        "feature_visuals": os.path.join(output_dir, "feature_visuals"),
        "selfcheck_polygons": os.path.join(output_dir, "selfcheck_polygons"),
        "selfcheck_visuals": os.path.join(output_dir, "selfcheck_visuals"),
        "reports": os.path.join(output_dir, "reports"),
    }
    for path in out_dirs.values():
        ensure_dir(path)

    json_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".json") and f.lower().endswith("_layer_polygons.json")
    ])

    if not json_files:
        raise FileNotFoundError("No *_layer_polygons.json files were found in the selected input folder.")

    print(f"Found {len(json_files)} JSON files. Stage A processing has started.")
    print("Confirmed output mode: advanced-only self-check visualization.")

    all_layer_rows: List[Dict[str, Any]] = []
    all_sample_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for idx, json_path in enumerate(json_files, start=1):
        print(f"[{idx}/{len(json_files)}] Processing: {os.path.basename(json_path)}")
        try:
            layer_rows, sample_summary = process_single_json(json_path, out_dirs, cfg)
            all_layer_rows.extend(layer_rows)
            all_sample_rows.append({
                "sample_name": sample_summary["sample_name"],
                "source_image": sample_summary["source_image"],
                "feature_file": sample_summary["feature_file"],
                "layer_count": sample_summary["layer_count"],
                "mean_mask_iou": sample_summary["metrics_mean"]["mask_iou"],
                "mean_mask_dice": sample_summary["metrics_mean"]["mask_dice"],
                "mean_area_relative_error": sample_summary["metrics_mean"]["area_relative_error"],
                "mean_centroid_distance_normalized": sample_summary["metrics_mean"]["centroid_distance_normalized"],
                "mean_boundary_chamfer_normalized": sample_summary["metrics_mean"]["boundary_chamfer_normalized"],
                "mean_boundary_hausdorff_normalized": sample_summary["metrics_mean"]["boundary_hausdorff_normalized"],
                "mean_polygon_count_abs_diff": sample_summary["metrics_mean"]["polygon_count_abs_diff"],
            })
            manifest_rows.append({
                "sample_name": sample_summary["sample_name"],
                "source_image": sample_summary["source_image"],
                "input_json": os.path.basename(json_path),
                "feature_file": f"features/{sample_summary['sample_name']}_features.npz",
                "metadata_file": f"metadata/{sample_summary['sample_name']}_metadata.json",
                "combined_triptych_file": f"feature_visuals/{sample_summary['sample_name']}_combined_triptych.png",
                "selfcheck_polygon_file": f"selfcheck_polygons/{sample_summary['sample_name']}_reconstructed_from_sdf.json",
                "selfcheck_visual_advanced_file": f"selfcheck_visuals/{sample_summary['sample_name']}_selfcheck_overlay_advanced.png",
            })
        except Exception as exc:
            print(f"Failed to process {os.path.basename(json_path)}: {exc}")
            errors.append({"file": os.path.basename(json_path), "error": str(exc)})

    layer_fieldnames = [
        "sample_name", "source_image", "layer_name", "vertical_rank",
        "input_polygon_count", "reconstructed_polygon_count", "polygon_count_abs_diff",
        "corner_count", "mask_iou", "mask_dice", "area_relative_error",
        "centroid_distance_normalized", "boundary_chamfer_normalized",
        "boundary_hausdorff_normalized",
    ]
    sample_fieldnames = [
        "sample_name", "source_image", "feature_file", "layer_count",
        "mean_mask_iou", "mean_mask_dice", "mean_area_relative_error",
        "mean_centroid_distance_normalized", "mean_boundary_chamfer_normalized",
        "mean_boundary_hausdorff_normalized", "mean_polygon_count_abs_diff",
    ]

    save_csv(os.path.join(out_dirs["reports"], "evaluation_per_layer.csv"), all_layer_rows, layer_fieldnames)
    save_csv(os.path.join(out_dirs["reports"], "evaluation_per_sample.csv"), all_sample_rows, sample_fieldnames)
    save_csv(os.path.join(out_dirs["reports"], "database_manifest.csv"), manifest_rows,
             list(manifest_rows[0].keys()) if manifest_rows else ["sample_name"])

    metric_names_layer = [
        "mask_iou", "mask_dice", "area_relative_error", "centroid_distance_normalized",
        "boundary_chamfer_normalized", "boundary_hausdorff_normalized",
        "polygon_count_abs_diff", "corner_count",
    ]
    metric_names_sample = [
        "mean_mask_iou", "mean_mask_dice", "mean_area_relative_error",
        "mean_centroid_distance_normalized", "mean_boundary_chamfer_normalized",
        "mean_boundary_hausdorff_normalized", "mean_polygon_count_abs_diff",
    ]

    summary = {
        "config": asdict(cfg),
        "input_folder": os.path.abspath(input_dir),
        "output_folder": os.path.abspath(output_dir),
        "processed_file_count": len(manifest_rows),
        "failed_file_count": len(errors),
        "errors": errors,
        "overall_layer_metrics": summarize_numeric_rows(all_layer_rows, metric_names_layer),
        "overall_sample_metrics": summarize_numeric_rows(all_sample_rows, metric_names_sample),
        "per_vertical_rank": {},
    }

    for rank in cfg.layer_order_preference:
        rank_rows = [row for row in all_layer_rows if str(row.get("vertical_rank", "")).lower() == rank]
        summary["per_vertical_rank"][rank] = summarize_numeric_rows(rank_rows, metric_names_layer)

    write_json(os.path.join(out_dirs["reports"], "evaluation_summary.json"), summary)
    write_json(os.path.join(out_dirs["reports"], "database_manifest.json"), {"samples": manifest_rows, "errors": errors})

    print("Stage A processing completed.")
    print(f"Processed files: {len(manifest_rows)}")
    print(f"Failed files: {len(errors)}")
    print(f"Results were saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
