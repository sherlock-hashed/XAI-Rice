"""Deterministic, Group-Aware Stratified 4-Way Splitting Module for Phase 1.
Implements constrained optimization of class distribution alignment while strictly isolating duplicate groups.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from src.phase1.label_analysis import CLASS_NAME_TO_ID, PRIMARY_CLASS_NAMES

TARGET_PROPORTIONS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "internal_test": 0.10
}

def generate_group_aware_stratified_splits(
    df_primary_with_groups: pd.DataFrame,
    seed: int = 42,
    target_ratios: Dict[str, float] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Partition primary dataset into Train, Validation, Calibration, and Internal Test.
    
    Mathematical Formulation:
      Class-by-Class Group-Aware Proportional Allocation:
      For each class c, allocate groups to split s* = argmin_s (allocated_{s, c} / target_{s, c})
      Subject to: G_i ∩ G_j = ∅ for all i != j (strict atomic group isolation)
    """
    if target_ratios is None:
        target_ratios = TARGET_PROPORTIONS

    df = df_primary_with_groups.copy()
    num_classes = len(PRIMARY_CLASS_NAMES)
    total_samples = len(df)
    split_names = list(target_ratios.keys())

    # 1. Aggregate group-level statistics
    group_data = []
    for grp_id, grp_df in df.groupby("duplicate_group_id"):
        class_counts_vec = np.zeros(num_classes, dtype=int)
        for _, row in grp_df.iterrows():
            c_name = row["class_name"]
            if c_name in CLASS_NAME_TO_ID:
                class_counts_vec[CLASS_NAME_TO_ID[c_name]] += 1
        
        maj_class = int(np.argmax(class_counts_vec)) if np.sum(class_counts_vec) > 0 else 0
        group_data.append({
            "group_id": grp_id,
            "sample_count": len(grp_df),
            "class_counts": class_counts_vec,
            "majority_class": maj_class,
            "sample_ids": grp_df["sample_id"].tolist()
        })

    # Overall dataset class distribution target
    overall_class_counts = np.zeros(num_classes, dtype=float)
    for g in group_data:
        overall_class_counts += g["class_counts"]

    # Target class counts per split
    split_class_targets = {
        s: np.maximum(target_ratios[s] * overall_class_counts, 1.0) for s in split_names
    }
    split_targets = {s: target_ratios[s] * total_samples for s in split_names}

    # Deterministic RNG for seed reproducibility
    rng = np.random.RandomState(seed)

    # State tracking
    split_allocated_counts = {s: np.zeros(num_classes, dtype=float) for s in split_names}
    split_group_assignments = {s: [] for s in split_names}

    # 2. Stratified Allocation Class-by-Class
    # Group the group_data by majority class
    groups_by_class = {c_idx: [] for c_idx in range(num_classes)}
    for g in group_data:
        groups_by_class[g["majority_class"]].append(g)

    for c_idx in range(num_classes):
        cls_groups = groups_by_class[c_idx]
        if not cls_groups:
            continue

        # Shuffle deterministically using seed + class_idx
        cls_rng = np.random.RandomState(seed + c_idx * 100)
        indices = np.arange(len(cls_groups))
        cls_rng.shuffle(indices)
        cls_groups = [cls_groups[i] for i in indices]
        
        # Sort by group size descending to pack larger duplicate clusters first
        cls_groups.sort(key=lambda x: -x["sample_count"])

        for g in cls_groups:
            g_counts = g["class_counts"]
            
            # Select split with lowest fulfillment ratio for this class
            # (with tie-breaker on overall split capacity)
            best_split = None
            best_ratio = float("inf")

            for s in split_names:
                curr_c = split_allocated_counts[s][c_idx]
                target_c = split_class_targets[s][c_idx]
                fulfillment_ratio = curr_c / target_c
                
                # Secondary metric: overall split fulfillment
                curr_total = np.sum(split_allocated_counts[s])
                target_total = split_targets[s]
                overall_ratio = curr_total / target_total

                combined_ratio = fulfillment_ratio + 0.1 * overall_ratio

                if combined_ratio < best_ratio:
                    best_ratio = combined_ratio
                    best_split = s

            split_group_assignments[best_split].append(g["group_id"])
            split_allocated_counts[best_split] += g_counts

    # 3. Map assignments back to sample rows
    group_to_split = {}
    for s, grps in split_group_assignments.items():
        for grp in grps:
            group_to_split[grp] = s

    df["split"] = df["duplicate_group_id"].map(group_to_split)

    # 4. Construct split DataFrames
    split_dfs = {
        "train": df[df["split"] == "train"].copy().reset_index(drop=True),
        "validation": df[df["split"] == "validation"].copy().reset_index(drop=True),
        "calibration": df[df["split"] == "calibration"].copy().reset_index(drop=True),
        "internal_test": df[df["split"] == "internal_test"].copy().reset_index(drop=True)
    }

    # 5. Compute split distribution metrics
    split_summary = {
        "seed": seed,
        "total_primary_samples": total_samples,
        "splits": {}
    }

    for s in split_names:
        s_df = split_dfs[s]
        s_cnt = len(s_df)
        s_pct = round((s_cnt / total_samples) * 100, 2)
        
        cls_counts = s_df["class_name"].value_counts().to_dict()
        cls_pcts = {c: round((cls_counts.get(c, 0) / s_cnt) * 100, 2) if s_cnt > 0 else 0.0 for c in PRIMARY_CLASS_NAMES}
        
        split_summary["splits"][s] = {
            "sample_count": s_cnt,
            "target_count": round(split_targets[s]),
            "actual_percentage": s_pct,
            "target_percentage": round(target_ratios[s] * 100, 2),
            "class_counts": {c: cls_counts.get(c, 0) for c in PRIMARY_CLASS_NAMES},
            "class_percentages": cls_pcts
        }

    return df, split_dfs, split_summary

def freeze_manifests(
    df_all_splits: pd.DataFrame,
    split_dfs: Dict[str, pd.DataFrame],
    df_duplicate_groups: pd.DataFrame,
    df_sethy: pd.DataFrame,
    df_bd5: pd.DataFrame,
    output_dir: str
) -> Dict[str, Any]:
    """
    Export and cryptographically freeze all Phase 1 manifests.
    Calculates SHA-256 checksum for each generated manifest file.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    manifest_files = {
        "primary_all_splits": ("primary_all_splits.csv", df_all_splits),
        "primary_train": ("primary_train.csv", split_dfs["train"]),
        "primary_validation": ("primary_validation.csv", split_dfs["validation"]),
        "primary_calibration": ("primary_calibration.csv", split_dfs["calibration"]),
        "primary_internal_test": ("primary_internal_test.csv", split_dfs["internal_test"]),
        "duplicate_groups": ("duplicate_groups.csv", df_duplicate_groups),
        "sethy_external": ("sethy_external.csv", df_sethy),
        "bd5_external": ("bd5_external.csv", df_bd5)
    }

    # Extract RiceSeg ground truth manifest from Sethy pairing
    riceseg_records = []
    for _, row in df_sethy.iterrows():
        if pd.notna(row.get("mask_path")):
            riceseg_records.append({
                "sample_id": f"RICESEG_{len(riceseg_records):06d}",
                "paired_sethy_id": row["sample_id"],
                "paired_sethy_filename": row["filename"],
                "class_name": row["class_name"],
                "mask_path": row["mask_path"],
                "dataset_role": "XAI_GROUND_TRUTH"
            })
    df_riceseg = pd.DataFrame(riceseg_records)
    manifest_files["riceseg_ground_truth"] = ("riceseg_ground_truth.csv", df_riceseg)

    checksums = {}
    paths_saved = {}

    for key, (fname, dframe) in manifest_files.items():
        fpath = out_p / fname
        dframe.to_csv(fpath, index=False)
        
        # Calculate SHA-256
        hasher = hashlib.sha256()
        with open(fpath, "rb") as f:
            hasher.update(f.read())
        sha = hasher.hexdigest()
        
        checksums[fname] = sha
        paths_saved[key] = str(fpath.resolve())

    # Write checksum registry
    checksum_file = out_p / "manifest_checksums.json"
    with open(checksum_file, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    return {
        "manifest_paths": paths_saved,
        "checksums": checksums,
        "checksum_file": str(checksum_file.resolve())
    }
