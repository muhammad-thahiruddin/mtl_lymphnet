#!/usr/bin/env python3
"""Generate all Group-A manuscript figures from existing data.

Usage:
    python script/generate_figures.py

All outputs are saved to outputs_v2/figures/.
No GPU, Kaggle, or retraining required.
"""
from __future__ import annotations

import os
import pickle
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
PKL_PATH = PROJECT_ROOT / "outputs_v2" / "experiment_results.pkl"
FIG_DIR = PROJECT_ROOT / "outputs_v2" / "figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
}

# Color palette
C_BLUE   = "#2196F3"
C_ORANGE = "#FF9800"
C_GREEN  = "#4CAF50"
C_RED    = "#E91E63"
C_PURPLE = "#9C27B0"
C_TEAL   = "#009688"
C_BROWN  = "#795548"

C_BENIGN    = "#4CAF50"
C_MALIGNANT = "#E91E63"
C_CENTER1   = "#2196F3"
C_CENTER2   = "#FF5722"


def _save(fig, name: str) -> None:
    """Save figure to FIG_DIR."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out.relative_to(PROJECT_ROOT)}")


# ===================================================================
# HELPER: load dataset images/masks
# ===================================================================
def _collect_dataset_info(dataset_root: Path) -> list[dict]:
    """Scan all images and masks, return list of dicts with metadata."""
    records = []
    for center_name in ["Center 1", "Center 2"]:
        center_dir = dataset_root / center_name
        if not center_dir.exists():
            continue
        for class_name in ["Benign", "Malignant"]:
            class_dir = center_dir / class_name
            if not class_dir.exists():
                continue
            image_files = sorted(glob.glob(str(class_dir / "* Image.bmp")))
            for img_path in image_files:
                img_path = Path(img_path)
                basename = img_path.name
                # Extract number: "1 Benign Image.bmp" -> "1"
                num = basename.split(" ")[0]
                mask_path = class_dir / f"{num} {class_name} Mask.tif"
                lesion_path = class_dir / f"{num} {class_name} Lesion.bmp"
                records.append({
                    "center": center_name,
                    "class": class_name,
                    "num": int(num),
                    "image_path": img_path,
                    "mask_path": mask_path if mask_path.exists() else None,
                    "lesion_path": lesion_path if lesion_path.exists() else None,
                })
    return records


# ===================================================================
# FIG 1: Image Properties (EDA)
# ===================================================================
def fig1_image_properties(records: list[dict]) -> None:
    """2x2 grid: (a) resolution dist, (b) aspect ratio, (c) intensity hist, (d) size vs center."""
    print("  Generating fig1_image_properties...")
    plt.rcParams.update(PUB_RC)

    widths, heights, aspects, centers, classes = [], [], [], [], []
    intensities_c1, intensities_c2 = [], []

    for rec in records:
        img = Image.open(rec["image_path"])
        w, h = img.size
        widths.append(w)
        heights.append(h)
        aspects.append(w / h)
        centers.append(rec["center"])
        classes.append(rec["class"])
        arr = np.array(img.convert("L")).ravel()
        if rec["center"] == "Center 1":
            intensities_c1.append(arr)
        else:
            intensities_c2.append(arr)

    widths = np.array(widths)
    heights = np.array(heights)
    aspects = np.array(aspects)
    centers = np.array(centers)
    classes = np.array(classes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Resolution scatter by center
    ax = axes[0, 0]
    for cname, color in [("Center 1", C_CENTER1), ("Center 2", C_CENTER2)]:
        mask = centers == cname
        ax.scatter(widths[mask], heights[mask], c=color, alpha=0.5, s=30,
                   edgecolors="none", label=cname)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("(a) Image Resolution by Center", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Aspect ratio histogram by center
    ax = axes[0, 1]
    c1_asp = aspects[centers == "Center 1"]
    c2_asp = aspects[centers == "Center 2"]
    ax.hist(c1_asp, bins=15, alpha=0.6, color=C_CENTER1, label="Center 1", edgecolor="white")
    ax.hist(c2_asp, bins=15, alpha=0.6, color=C_CENTER2, label="Center 2", edgecolor="white")
    ax.set_xlabel("Aspect Ratio (W/H)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Aspect Ratio Distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Intensity histogram by center
    ax = axes[1, 0]
    all_c1 = np.concatenate(intensities_c1)
    all_c2 = np.concatenate(intensities_c2)
    ax.hist(all_c1, bins=64, range=(0, 255), alpha=0.5, color=C_CENTER1,
            label="Center 1", density=True, edgecolor="none")
    ax.hist(all_c2, bins=64, range=(0, 255), alpha=0.5, color=C_CENTER2,
            label="Center 2", density=True, edgecolor="none")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title("(c) Grayscale Intensity Distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) Resolution by class (box)
    ax = axes[1, 1]
    areas = widths * heights
    data_groups = [
        areas[(centers == "Center 1") & (classes == "Benign")],
        areas[(centers == "Center 1") & (classes == "Malignant")],
        areas[(centers == "Center 2") & (classes == "Benign")],
        areas[(centers == "Center 2") & (classes == "Malignant")],
    ]
    bp = ax.boxplot(data_groups, tick_labels=["C1-Ben", "C1-Mal", "C2-Ben", "C2-Mal"],
                    patch_artist=True, widths=0.6)
    box_colors = [C_BENIGN, C_MALIGNANT, C_BENIGN, C_MALIGNANT]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Image Area (px²)")
    ax.set_title("(d) Image Area by Center and Class", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Dataset Image Properties", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig1_image_properties.png")


# ===================================================================
# FIG 2: Mask Analysis (EDA)
# ===================================================================
def fig2_mask_analysis(records: list[dict]) -> None:
    """2x2: (a) mask area dist, (b) mask centroid, (c) compactness, (d) overlay samples."""
    print("  Generating fig2_mask_analysis...")
    plt.rcParams.update(PUB_RC)

    mask_areas, mask_ratios, compactness_vals = [], [], []
    cx_list, cy_list = [], []
    rec_centers, rec_classes = [], []

    # Collect 4 good overlays: one per center × class
    overlays = {"Center 1_Benign": None, "Center 1_Malignant": None,
                "Center 2_Benign": None, "Center 2_Malignant": None}

    for rec in records:
        if rec["mask_path"] is None:
            continue
        mask = np.array(Image.open(rec["mask_path"]))
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        binary = (mask > 0).astype(np.uint8)
        area = binary.sum()
        total = binary.size
        mask_areas.append(area)
        mask_ratios.append(area / total * 100)
        rec_centers.append(rec["center"])
        rec_classes.append(rec["class"])

        # Centroid
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            cx_list.append(xs.mean() / binary.shape[1])
            cy_list.append(ys.mean() / binary.shape[0])
        else:
            cx_list.append(0.5)
            cy_list.append(0.5)

        # Compactness: 4π * Area / Perimeter²
        # Approximate perimeter by counting boundary pixels
        from scipy import ndimage
        dilated = ndimage.binary_dilation(binary)
        perimeter = (dilated ^ binary).sum()
        if perimeter > 0:
            comp = 4 * np.pi * area / (perimeter ** 2)
        else:
            comp = 0
        compactness_vals.append(min(comp, 1.0))

        # Collect overlay
        key = f"{rec['center']}_{rec['class']}"
        if overlays[key] is None and area > 500:
            img_arr = np.array(Image.open(rec["image_path"]).convert("RGB"))
            overlays[key] = (img_arr, binary, rec["center"], rec["class"])

    mask_areas = np.array(mask_areas)
    mask_ratios = np.array(mask_ratios)
    compactness_vals = np.array(compactness_vals)
    cx_list = np.array(cx_list)
    cy_list = np.array(cy_list)
    rec_centers = np.array(rec_centers)
    rec_classes = np.array(rec_classes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Mask area ratio by center+class
    ax = axes[0, 0]
    groups = [
        mask_ratios[(rec_centers == "Center 1") & (rec_classes == "Benign")],
        mask_ratios[(rec_centers == "Center 1") & (rec_classes == "Malignant")],
        mask_ratios[(rec_centers == "Center 2") & (rec_classes == "Benign")],
        mask_ratios[(rec_centers == "Center 2") & (rec_classes == "Malignant")],
    ]
    bp = ax.boxplot(groups, tick_labels=["C1-Ben", "C1-Mal", "C2-Ben", "C2-Mal"],
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], [C_BENIGN, C_MALIGNANT, C_BENIGN, C_MALIGNANT]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mask Area (% of image)")
    ax.set_title("(a) Mask Area Ratio", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # (b) Centroid position scatter
    ax = axes[0, 1]
    for cname, color in [("Center 1", C_CENTER1), ("Center 2", C_CENTER2)]:
        m = rec_centers == cname
        ax.scatter(cx_list[m], cy_list[m], c=color, alpha=0.4, s=25,
                   edgecolors="none", label=cname)
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title("(b) Mask Centroid Position", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invert Y to match image coords
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # (c) Compactness by class
    ax = axes[1, 0]
    ben_comp = compactness_vals[rec_classes == "Benign"]
    mal_comp = compactness_vals[rec_classes == "Malignant"]
    ax.hist(ben_comp, bins=20, alpha=0.6, color=C_BENIGN, label="Benign", edgecolor="white")
    ax.hist(mal_comp, bins=20, alpha=0.6, color=C_MALIGNANT, label="Malignant", edgecolor="white")
    ax.set_xlabel("Compactness (4πA/P²)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Mask Compactness", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) 4 overlay samples
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("(d) Mask Overlay Samples", fontweight="bold")
    keys_order = ["Center 1_Benign", "Center 1_Malignant", "Center 2_Benign", "Center 2_Malignant"]
    sub_axes = ax.inset_axes([0, 0, 1, 1])
    sub_axes.axis("off")
    for i, key in enumerate(keys_order):
        inset = sub_axes.inset_axes([0.02 + (i % 2) * 0.5, 0.52 - (i // 2) * 0.5,
                                     0.46, 0.46])
        if overlays[key] is not None:
            img_arr, binary, cname, clname = overlays[key]
            overlay = img_arr.copy()
            color_mask = np.array([255, 0, 0]) if clname == "Malignant" else np.array([0, 255, 0])
            overlay[binary > 0] = (0.6 * overlay[binary > 0] + 0.4 * color_mask).astype(np.uint8)
            inset.imshow(overlay)
            inset.set_title(f"{cname[-1]}-{clname[:3]}", fontsize=8)
        inset.axis("off")

    fig.suptitle("Segmentation Mask Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig2_mask_analysis.png")


# ===================================================================
# FIG 3: Sample Grid (Updated 3x4: Original, GT, Overlay)
# ===================================================================
def fig3_sample_grid(records: list[dict]) -> None:
    """3x4 grid: rows=(Original, GT Mask, Overlay), cols=(C1 Benign, C1 Malignant, C2 Benign, C2 Malignant)."""
    print("  Generating fig3_sample_grid (3x4)...")
    plt.rcParams.update(PUB_RC)

    # 1. Group data by (center, class)
    groups = {}
    for rec in records:
        key = (rec["center"], rec["class"])
        if key not in groups:
            groups[key] = []
        groups[key].append(rec)

    # 2. Define column and row configuration
    col_configs = [
        ("Center 1", "Benign"),
        ("Center 1", "Malignant"),
        ("Center 2", "Benign"),
        ("Center 2", "Malignant")
    ]
    
    row_labels = ["Original", "Ground Truth", "Overlay"]
    
    # Create 3 rows and 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # 3. Iterate to fill each column
    for col_i, (center, cls) in enumerate(col_configs):
        # Take the first available sample for this combination
        recs = groups.get((center, cls), [])
        rec = recs[0] if recs else None
        
        if rec is not None:
            # --- A. Load Original Image ---
            img = np.array(Image.open(rec["image_path"]).convert("RGB"))
            
            # --- B. Load & Process Mask (Ground Truth) ---
            if rec["mask_path"] is not None:
                mask = np.array(Image.open(rec["mask_path"]))
                if mask.ndim > 2:
                    mask = mask[:, :, 0]
                binary = (mask > 0).astype(np.uint8)
                
                # Adjust mask resolution if different from image
                if binary.shape != img.shape[:2]:
                    from PIL import Image as PILImage
                    binary = np.array(PILImage.fromarray(binary).resize(
                        (img.shape[1], img.shape[0]), PILImage.NEAREST))
            else:
                # If no mask, create an empty array (black)
                binary = np.zeros(img.shape[:2], dtype=np.uint8)

            # --- C. Create Overlay ---
            color = np.array([0, 255, 0]) if cls == "Benign" else np.array([255, 0, 0])
            img_overlay = img.copy()
            img_overlay[binary > 0] = (0.65 * img[binary > 0] + 0.35 * color).astype(np.uint8)
            
            # --- Display to Grid ---
            axes[0, col_i].imshow(img)
            axes[1, col_i].imshow(binary, cmap="gray") # GT displayed in black/white
            axes[2, col_i].imshow(img_overlay)

        # Settings for all rows in this column
        for row_i in range(3):
            ax = axes[row_i, col_i]
            ax.axis("off") # Turn off standard lines/axes
            
            # Add Vertical Text (Y-Axis) only in the leftmost column
            if col_i == 0:
                # Use ax.text to avoid conflict with ax.axis("off")
                ax.text(-0.1, 0.5, row_labels[row_i], fontsize=12, fontweight="bold", 
                        rotation=90, ha="right", va="center", transform=ax.transAxes)
                
        # Give class title (Benign/Malignant) to the top row of each column
        axes[0, col_i].set_title(cls, fontsize=12, fontweight="bold", pad=10)

    # --- 4. Layout Settings and Center Text Placement ---
    fig.suptitle("Representative Samples: Image, Ground Truth, and Mask Overlay", 
                 fontweight="bold", fontsize=15, y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.93])
    
    fig.canvas.draw()
    
    # Center 1 Position (between column 0 & 1)
    pos_c1_left = axes[0, 0].get_position()
    pos_c1_right = axes[0, 1].get_position()
    mid_x1 = (pos_c1_left.x0 + pos_c1_right.x1) / 2.0
    
    # Center 2 Position (between column 2 & 3)
    pos_c2_left = axes[0, 2].get_position()
    pos_c2_right = axes[0, 3].get_position()
    mid_x2 = (pos_c2_left.x0 + pos_c2_right.x1) / 2.0
    
    # Place Center text exactly in the middle
    fig.text(mid_x1, 0.89, "Center 1", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(mid_x2, 0.89, "Center 2", ha="center", va="center", fontsize=14, fontweight="bold")

    _save(fig, "fig3_sample_grid.png")


# ===================================================================
# FIG 7: MTL Ablation Bars
# ===================================================================
def fig7_ablation_mtl(data: dict) -> None:
    """Grouped bar chart: MTL vs Seg-Only vs Cls-Only."""
    print("  Generating fig7_ablation_mtl...")
    plt.rcParams.update(PUB_RC)

    exp3 = data["exp3_ablation_mtl"]
    
    # Save config with its color for consistency when filtering
    configs = {
        "MTL (Proposed)": (exp3["mtl"], C_BLUE),
        "Seg-Only": (exp3["seg_only"], C_ORANGE),
        "Cls-Only": (exp3["cls_only"], C_GREEN),
    }
    
    seg_metrics = ["seg_dice", "seg_iou"]
    cls_metrics = ["cls_auc", "cls_accuracy", "cls_sensitivity", "cls_specificity", "cls_f1", "cls_mcc"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    width = 0.35 # Widened because now there are only 2 bars per group

    # --- Segmentation panel ---
    ax = axes[0]
    # Filter Cls-Only to avoid leaving empty space
    valid_seg = {k: v for k, v in configs.items() if k != "Cls-Only"}
    
    x = np.arange(len(seg_metrics))
    for i, (cfg_name, (folds, color)) in enumerate(valid_seg.items()):
        means = [np.nanmean([f.get(m, np.nan) for f in folds]) for m in seg_metrics]
        stds = [np.nanstd([f.get(m, np.nan) for f in folds]) for m in seg_metrics]
        
        bars = ax.bar(x + i * width, means, width, yerr=stds, label=cfg_name,
                      color=color, alpha=0.85, edgecolor="black", capsize=4)
        
        # Value labels
        for bar, mean in zip(bars, means):
            if mean > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=7)
                
    # Middle position between 2 bars is at x + width / 2
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Dice", "IoU"])
    ax.set_title("Segmentation Metrics", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Classification panel ---
    ax = axes[1]
    # Filter Seg-Only to avoid leaving empty space
    valid_cls = {k: v for k, v in configs.items() if k != "Seg-Only"}
    
    x = np.arange(len(cls_metrics))
    for i, (cfg_name, (folds, color)) in enumerate(valid_cls.items()):
        means = [np.nanmean([f.get(m, np.nan) for f in folds]) for m in cls_metrics]
        stds = [np.nanstd([f.get(m, np.nan) for f in folds]) for m in cls_metrics]
        
        bars = ax.bar(x + i * width, means, width, yerr=stds, label=cfg_name,
                      color=color, alpha=0.85, edgecolor="black", capsize=4)
                      
        for bar, mean in zip(bars, means):
            if mean > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=7)
                        
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["AUC", "Acc", "Sens", "Spec", "F1", "MCC"])
    ax.set_title("Classification Metrics", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-Task vs Single-Task Ablation (Internal CV)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig7_ablation_mtl.png")


# ===================================================================
# FIG 11: Dice Histogram
# ===================================================================
def fig11_dice_histogram(data: dict) -> None:
    """(a) Histogram colored by class, (b) Violin MTL vs Seg-Only."""
    print("  Generating fig11_dice_histogram...")
    plt.rcParams.update(PUB_RC)

    exp1_folds = data["exp1_internal_cv"]["fold_results"]
    exp3 = data["exp3_ablation_mtl"]

    # Pool per-image dice from proposed model
    all_dice, all_targets = [], []
    for f in exp1_folds:
        all_dice.extend(f["_per_image_dice"])
        all_targets.extend(f["_cls_targets"])
    all_dice = np.array(all_dice)
    all_targets = np.array(all_targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Histogram by class
    ax = axes[0]
    ben_dice = all_dice[all_targets == 0]
    mal_dice = all_dice[all_targets == 1]
    ax.hist(ben_dice, bins=30, alpha=0.6, color=C_BENIGN, label=f"Benign (n={len(ben_dice)})",
            edgecolor="white", range=(0, 1))
    ax.hist(mal_dice, bins=30, alpha=0.6, color=C_MALIGNANT, label=f"Malignant (n={len(mal_dice)})",
            edgecolor="white", range=(0, 1))
    ax.axvline(np.mean(all_dice), color="black", linestyle="--", linewidth=1.5,
               label=f"Overall mean = {np.mean(all_dice):.3f}")
    ax.set_xlabel("Dice Score")
    ax.set_ylabel("Count")
    # Add "Pooled" note
    ax.set_title("(a) Per-Image Dice by Class (Pooled, 3 Seeds)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) Violin: MTL vs Seg-Only
    ax = axes[1]
    mtl_dice = []
    # Take first 5 folds only (1 seed) for fair comparison
    for f in exp3["mtl"][:5]:
        mtl_dice.extend(f["_per_image_dice"])
        
    seg_dice = []
    for f in exp3["seg_only"][:5]:
        seg_dice.extend(f["_per_image_dice"])

    vp = ax.violinplot([mtl_dice, seg_dice], positions=[1, 2], showmeans=True,
                       showmedians=True, widths=0.7)
    colors_vp = [C_BLUE, C_ORANGE]
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors_vp[i])
        body.set_alpha(0.6)
    vp["cmeans"].set_color("black")
    vp["cmedians"].set_color("red")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MTL (Proposed)", "Seg-Only"])
    ax.set_ylabel("Dice Score")
    ax.set_title("(b) MTL vs Seg-Only Distribution", fontweight="bold")

    # Add stats annotation
    ax.text(1, np.mean(mtl_dice) + 0.01, f"mu={np.mean(mtl_dice):.3f}",
            ha="center", fontsize=8, color=C_BLUE)
    ax.text(2, np.mean(seg_dice) + 0.01, f"mu={np.mean(seg_dice):.3f}",
            ha="center", fontsize=8, color=C_ORANGE)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Image Dice Score Analysis (Internal CV)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig11_dice_histogram.png")


# ===================================================================
# FIG 14: Dice vs Classification Confidence
# ===================================================================
def fig14_dice_vs_confidence(data: dict) -> None:
    """Scatter: Dice (x) vs cls probability (y), colored by true label."""
    print("  Generating fig14_dice_vs_confidence...")
    plt.rcParams.update(PUB_RC)

    exp1_folds = data["exp1_internal_cv"]["fold_results"]
    all_dice, all_probs, all_targets = [], [], []
    for f in exp1_folds:
        all_dice.extend(f["_per_image_dice"])
        all_probs.extend(f["_cls_probs"])
        all_targets.extend(f["_cls_targets"])

    all_dice = np.array(all_dice)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Classification correctness (threshold=0.5)
    preds = (all_probs >= 0.5).astype(float)
    correct = (preds == all_targets)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 4 groups: benign-correct, benign-wrong, malignant-correct, malignant-wrong
    groups = [
        (all_targets == 0) & correct,   # TN
        (all_targets == 0) & ~correct,  # FP
        (all_targets == 1) & correct,   # TP
        (all_targets == 1) & ~correct,  # FN
    ]
    labels = ["Benign (correct)", "Benign (misclassified)",
              "Malignant (correct)", "Malignant (misclassified)"]
    colors = [C_BENIGN, "#81C784", C_MALIGNANT, "#F48FB1"]
    markers = ["o", "X", "o", "X"]
    sizes = [25, 60, 25, 60]
    alphas = [0.4, 0.9, 0.4, 0.9]

    for mask, label, color, marker, s, alpha in zip(groups, labels, colors, markers, sizes, alphas):
        ax.scatter(all_dice[mask], all_probs[mask], c=color, marker=marker, s=s,
                   alpha=alpha, label=f"{label} (n={mask.sum()})", edgecolors="none")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Cls threshold (0.5)")
    ax.set_xlabel("Segmentation Dice Score")
    ax.set_ylabel("Classification Probability (malignancy)")
    # Add "Pooled" note
    ax.set_title("Dice vs Classification Confidence (Pooled, 3 Seeds)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig14_dice_vs_confidence.png")


# ===================================================================
# FIG 6: Extended External Validation
# ===================================================================
def fig6_external_extended(data: dict) -> None:
    """Grouped bar chart: all 7 external configs × 2 directions × key metrics."""
    print("  Generating fig6_external_extended...")
    plt.rcParams.update(PUB_RC)

    # Build config list: default + 5 variants
    configs = {}

    # Default (TinyViT + Uncertainty)
    configs["TinyViT\nUncertainty"] = {
        "C1->C2": data["exp2_c1_to_c2"]["results"],
        "C2->C1": data["exp2_c2_to_c1"]["results"],
    }

    # Variant mapping
    variant_labels = {
        "MobileNetV3-Large_uncertainty_w0.5": "MobileNet\nUncertainty",
        "EfficientNet-B0_uncertainty_w0.5": "EffNet-B0\nUncertainty",
        "TinyViT-5M_fixed_w0.5": "TinyViT\nFixed 0.5",
        "TinyViT-5M_fixed_w0.7": "TinyViT\nFixed 0.7",
        "TinyViT-5M_dwa_w0.5": "TinyViT\nDWA",
    }

    for vk, label in variant_labels.items():
        vdata = data["exp2_variants"].get(vk, {})
        # FIX: Hapus redundant lookup vdata.get(..., vdata.get(...))
        c1c2 = vdata.get("C1->C2", {})
        c2c1 = vdata.get("C2->C1", {})
        
        configs[label] = {
            "C1->C2": c1c2.get("results", c1c2) if isinstance(c1c2, dict) else {},
            "C2->C1": c2c1.get("results", c2c1) if isinstance(c2c1, dict) else {},
        }

    metrics = ["seg_dice", "cls_auc", "cls_accuracy", "cls_sensitivity", "cls_specificity"]
    metric_labels = ["Dice", "AUC", "Accuracy", "Sensitivity", "Specificity"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    directions = ["C1->C2", "C2->C1"]
    dir_titles = ["Center 1 -> Center 2", "Center 2 -> Center 1"]

    n_configs = len(configs)
    colors_list = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_TEAL, C_BROWN]

    for ax, direction, dir_title in zip(axes, directions, dir_titles):
        x = np.arange(len(metrics))
        width = 0.8 / n_configs

        for i, (cfg_name, cfg_data) in enumerate(configs.items()):
            results = cfg_data.get(direction, {})
            vals = [results.get(m, 0) for m in metrics]
            ax.bar(x + i * width, vals, width, label=cfg_name,
                   color=colors_list[i % len(colors_list)], alpha=0.85,
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x + width * (n_configs - 1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.set_title(dir_title, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=7, ncol=2, loc="upper right")

    # Add internal CV reference line
    exp1_sum = data["exp1_internal_cv"]["summary"]
    for ax in axes:
        for m_i, m in enumerate(metrics):
            if m in exp1_sum:
                mean_val = exp1_sum[m]["mean"]
                # FIX: Remove unused x_pos variable, use m_i directly
                ax.plot([m_i - 0.05, m_i + width * n_configs + 0.05], [mean_val, mean_val],
                        color="black", linestyle=":", linewidth=1, alpha=0.5)

    fig.suptitle("Bidirectional External Validation — All Configurations", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig6_external_extended.png")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("Generate Group-A Figures")
    print("=" * 60)

    # Load pkl
    print(f"\nLoading {PKL_PATH.name}...")
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  Keys: {list(data.keys())}")

    # Collect dataset records
    print(f"\nScanning dataset at {DATASET_ROOT}...")
    records = _collect_dataset_info(DATASET_ROOT)
    print(f"  Found {len(records)} images")

    print(f"\nGenerating figures -> {FIG_DIR.relative_to(PROJECT_ROOT)}/\n")

    # EDA figures (from raw dataset)
    fig1_image_properties(records)
    fig2_mask_analysis(records)
    fig3_sample_grid(records)

    # Results figures (from pkl)
    fig7_ablation_mtl(data)
    fig11_dice_histogram(data)
    fig14_dice_vs_confidence(data)
    fig6_external_extended(data)

    print(f"\n{'=' * 60}")
    
    active_figs = 7
    print(f"Done! {active_figs} figures saved to {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()