"""Exploratory Data Analysis (EDA) & Publication Figure Generation Module for Phase 1."""

import os
import random
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend safe for Colab/headless environments
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from src.phase1.label_analysis import PRIMARY_CLASS_NAMES

def select_representative_samples(
    df_primary: pd.DataFrame,
    df_sethy: pd.DataFrame,
    df_bd5: pd.DataFrame,
    seed: int = 42,
    samples_per_class: int = 2
) -> pd.DataFrame:
    """
    Deterministically sample representative images across all datasets for reproducible EDA.
    """
    rng = random.Random(seed)
    sampled_records = []

    # 1. Primary Dataset Samples
    for cls in PRIMARY_CLASS_NAMES:
        cls_df = df_primary[df_primary["class_name"] == cls].sort_values("sample_id")
        if not cls_df.empty:
            indices = list(range(len(cls_df)))
            rng.shuffle(indices)
            chosen = cls_df.iloc[indices[:samples_per_class]]
            for _, r in chosen.iterrows():
                sampled_records.append({
                    "sample_id": r["sample_id"],
                    "dataset": "RiceLeafDiseaseBD",
                    "class_name": r["class_name"],
                    "relative_path": r["relative_path"],
                    "figure_role": "primary_representative"
                })

    # 2. Sethy Samples
    for cls in sorted(df_sethy["class_name"].unique()):
        cls_df = df_sethy[df_sethy["class_name"] == cls].sort_values("sample_id")
        if not cls_df.empty:
            indices = list(range(len(cls_df)))
            rng.shuffle(indices)
            chosen = cls_df.iloc[indices[:samples_per_class]]
            for _, r in chosen.iterrows():
                sampled_records.append({
                    "sample_id": r["sample_id"],
                    "dataset": "Sethy_5932",
                    "class_name": r["class_name"],
                    "relative_path": r["relative_path"],
                    "figure_role": "sethy_riceseg_pair"
                })

    # 3. BD5 Samples
    for cls in sorted(df_bd5["class_name"].unique()):
        cls_df = df_bd5[df_bd5["class_name"] == cls].sort_values("sample_id")
        if not cls_df.empty:
            indices = list(range(len(cls_df)))
            rng.shuffle(indices)
            chosen = cls_df.iloc[indices[:samples_per_class]]
            for _, r in chosen.iterrows():
                sampled_records.append({
                    "sample_id": r["sample_id"],
                    "dataset": "RiceLeafDisease_BD5",
                    "class_name": r["class_name"],
                    "relative_path": r["relative_path"],
                    "figure_role": "bd5_field_sample"
                })

    return pd.DataFrame(sampled_records)

def generate_all_phase1_figures(
    df_primary_splits: pd.DataFrame,
    split_dfs: Dict[str, pd.DataFrame],
    df_sethy: pd.DataFrame,
    df_bd5: pd.DataFrame,
    dataset_roots: Dict[str, str],
    output_dir: str,
    seed: int = 42
) -> Dict[str, str]:
    """
    Generate all 8 publication-grade EDA figures and export sample registry.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    generated_figures = {}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 13,
        "figure.dpi": 300
    })

    # --- Figure 1: Primary Class Distribution ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    class_counts = df_primary_splits["class_name"].value_counts()[PRIMARY_CLASS_NAMES]
    total_p = len(df_primary_splits)
    bars = ax.bar(class_counts.index, class_counts.values, color="#2b5c8f", width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_p) * 100
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 80, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("RiceLeafDiseaseBD (Primary Development Dataset): Class Distribution")
    ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(class_counts.values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    f1_path = out_p / "class_distribution.png"
    plt.savefig(f1_path)
    plt.close()
    generated_figures["class_distribution"] = str(f1_path.resolve())

    # --- Figure 2: Split Distribution by Class ---
    fig, ax = plt.subplots(figsize=(10, 5))
    split_colors = {"train": "#1f77b4", "validation": "#ff7f0e", "calibration": "#2ca02c", "internal_test": "#d62728"}
    x = np.arange(len(PRIMARY_CLASS_NAMES))
    width = 0.2
    for idx, (s_name, s_df) in enumerate(split_dfs.items()):
        s_counts = [int((s_df["class_name"] == c).sum()) for c in PRIMARY_CLASS_NAMES]
        ax.bar(x + (idx - 1.5) * width, s_counts, width, label=s_name.replace("_", " ").title(), color=split_colors[s_name], edgecolor="black", linewidth=0.6)
    ax.set_title("Group-Aware Stratified Split Distribution Across Disease Classes")
    ax.set_xticks(x)
    ax.set_xticklabels(PRIMARY_CLASS_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Number of Samples")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    f2_path = out_p / "split_distribution.png"
    plt.savefig(f2_path)
    plt.close()
    generated_figures["split_distribution"] = str(f2_path.resolve())

    # --- Figure 3: Image Dimensions Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    datasets_data = [
        ("Primary", df_primary_splits["width"], df_primary_splits["height"], "#2b5c8f"),
        ("Sethy", df_sethy["width"], df_sethy["height"], "#e66101"),
        ("BD5", df_bd5["width"], df_bd5["height"], "#5e3c99")
    ]
    for label, w, h, col in datasets_data:
        ax1.scatter(w, h, alpha=0.4, label=label, color=col, edgecolors="none", s=20)
    ax1.set_title("Image Resolution Distribution (Width vs. Height)")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Height (pixels)")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    df_primary_splits["file_size_kb"] = df_primary_splits["file_size_bytes"] / 1024.0
    ax2.hist(df_primary_splits["file_size_kb"], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.7)
    ax2.set_title("Primary Dataset File Size Distribution")
    ax2.set_xlabel("File Size (KB)")
    ax2.set_ylabel("Frequency")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    f3_path = out_p / "image_dimensions.png"
    plt.savefig(f3_path)
    plt.close()
    generated_figures["image_dimensions"] = str(f3_path.resolve())

    # --- Figure 4: Aspect Ratio Distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_primary_splits["aspect_ratio"].dropna(), bins=25, color="#3182bd", edgecolor="black", alpha=0.8)
    ax.set_title("Primary Dataset Aspect Ratio Distribution (Width / Height)")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    f4_path = out_p / "aspect_ratio_distribution.png"
    plt.savefig(f4_path)
    plt.close()
    generated_figures["aspect_ratio_distribution"] = str(f4_path.resolve())

    # Deterministic Sample Selection
    df_samples = select_representative_samples(df_primary_splits, df_sethy, df_bd5, seed=seed)

    # --- Figure 5: Representative Primary Samples ---
    p_root = Path(dataset_roots.get("primary_dataset", "."))
    p_samples = df_samples[df_samples["figure_role"] == "primary_representative"]
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    axes = axes.flatten()
    for idx, cls in enumerate(PRIMARY_CLASS_NAMES):
        ax = axes[idx]
        cls_sample = p_samples[p_samples["class_name"] == cls].iloc[0]
        img_path = p_root / cls_sample["relative_path"]
        try:
            with Image.open(str(img_path)) as img:
                ax.imshow(img)
                ax.set_title(f"{cls}", fontsize=11, fontweight="bold")
                ax.axis("off")
        except Exception:
            ax.text(0.5, 0.5, f"{cls}\n(Image unreadable)", ha="center", va="center")
            ax.axis("off")
    plt.suptitle("Representative Samples: RiceLeafDiseaseBD Classes", y=0.98)
    plt.tight_layout()
    f5_path = out_p / "representative_samples.png"
    plt.savefig(f5_path)
    plt.close()
    generated_figures["representative_samples"] = str(f5_path.resolve())

    # --- Figure 6: Annotation Overlay Visualization ---
    annot_candidates = df_primary_splits[df_primary_splits["annotation_path"].notna() & df_primary_splits["class_name"].isin(["Blast", "Brown spot"])].sort_values("sample_id")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    classes_to_plot = ["Blast", "Brown spot"]
    
    for idx, cls in enumerate(classes_to_plot):
        ax = axes[idx]
        cls_rows = annot_candidates[annot_candidates["class_name"] == cls]
        if not cls_rows.empty:
            row = cls_rows.iloc[0]
            img_path = p_root / row["relative_path"]
            lbl_path = Path(row["annotation_path"])
            
            try:
                with Image.open(str(img_path)) as img:
                    img_draw = img.copy()
                    draw = ImageDraw.Draw(img_draw)
                    W, H = img.size
                    
                    if lbl_path.exists():
                        with open(lbl_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                    x1 = (xc - w / 2.0) * W
                                    y1 = (yc - h / 2.0) * H
                                    x2 = (xc + w / 2.0) * W
                                    y2 = (yc + h / 2.0) * H
                                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    ax.imshow(img_draw)
                    ax.set_title(f"{cls} — Verified Bounding Boxes", fontsize=10)
                    ax.axis("off")
            except Exception:
                ax.axis("off")
        else:
            ax.axis("off")
            
    plt.suptitle("Annotation Syntax Verification Visualization\n[Note: Visual inspection only; semantic lesion-grounded usage deferred to evaluation phases]", fontsize=9, y=0.98)
    plt.tight_layout()
    f6_path = out_p / "primary_annotation_samples.png"
    plt.savefig(f6_path)
    plt.close()
    generated_figures["primary_annotation_samples"] = str(f6_path.resolve())

    # --- Figure 7: Sethy <-> RiceSeg Paired Visualization ---
    s_root = Path(dataset_roots.get("sethy_external", "."))
    r_root = Path(dataset_roots.get("riceseg_ground_truth", "."))
    
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    sethy_classes = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
    
    for c_idx, cls in enumerate(sethy_classes):
        s_cls_rows = df_sethy[df_sethy["class_name"] == cls].sort_values("sample_id")
        if not s_cls_rows.empty:
            s_row = s_cls_rows.iloc[0]
            s_img_path = s_root / cls / s_row["filename"]
            r_msk_path = r_root / cls / f"{Path(s_row['filename']).stem}.png"
            
            ax_img = axes[0, c_idx]
            ax_msk = axes[1, c_idx]
            
            try:
                with Image.open(str(s_img_path)) as img:
                    ax_img.imshow(img)
                    ax_img.set_title(f"Sethy: {cls}", fontsize=9)
                    ax_img.axis("off")
            except Exception:
                ax_img.axis("off")
                
            try:
                with Image.open(str(r_msk_path)) as msk:
                    ax_msk.imshow(msk, cmap="gray")
                    ax_msk.set_title(f"RiceSeg Mask", fontsize=9)
                    ax_msk.axis("off")
            except Exception:
                ax_msk.axis("off")
                
    plt.suptitle("Sethy 5932 Images Paired with RiceSeg Ground Truth Masks", fontsize=11, y=0.98)
    plt.tight_layout()
    f7_path = out_p / "sethy_riceseg_pairs.png"
    plt.savefig(f7_path)
    plt.close()
    generated_figures["sethy_riceseg_pairs"] = str(f7_path.resolve())

    # --- Figure 8: BD5 Field Dataset Samples ---
    b_root = Path(dataset_roots.get("bd5_external", "."))
    bd5_classes = sorted(df_bd5["class_name"].unique())
    fig, axes = plt.subplots(1, len(bd5_classes), figsize=(12, 3))
    
    for idx, cls in enumerate(bd5_classes):
        ax = axes[idx]
        b_rows = df_bd5[df_bd5["class_name"] == cls].sort_values("sample_id")
        if not b_rows.empty:
            b_row = b_rows.iloc[0]
            b_img_path = b_root / cls / b_row["filename"]
            try:
                with Image.open(str(b_img_path)) as img:
                    ax.imshow(img)
                    ax.set_title(f"{cls}", fontsize=9)
                    ax.axis("off")
            except Exception:
                ax.axis("off")
        else:
            ax.axis("off")
            
    plt.suptitle("RiceLeafDisease-BD5: Representative Field Domain Images\n[Narrow Brown Spot is preserved as a distinct class]", fontsize=10, y=1.05)
    plt.tight_layout()
    f8_path = out_p / "bd5_samples.png"
    plt.savefig(f8_path)
    plt.close()
    generated_figures["bd5_samples"] = str(f8_path.resolve())

    return generated_figures, df_samples
