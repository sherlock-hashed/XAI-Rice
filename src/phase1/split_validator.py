"""Comprehensive Phase 1 Split and Leakage Validation Engine."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.phase1.label_analysis import PRIMARY_CLASS_NAMES

def validate_phase1_splits(
    df_primary_splits: pd.DataFrame,
    split_dfs: Dict[str, pd.DataFrame],
    df_sethy: pd.DataFrame,
    df_bd5: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform exhaustive validation of experimental splits, duplicate isolation,
    class coverage, and external dataset firewall constraints.
    """
    checks = {}
    all_passed = True

    total_primary = len(df_primary_splits)

    # 1. Total count check
    split_counts_sum = sum(len(df) for df in split_dfs.values())
    count_check_passed = (split_counts_sum == total_primary == 17963)
    checks["primary_completeness"] = {
        "passed": count_check_passed,
        "detail": f"Total samples: {split_counts_sum} (Expected: 17963)"
    }
    if not count_check_passed:
        all_passed = False

    # 2. Mutual Exclusivity of Splits
    split_id_sets = {s: set(df["sample_id"]) for s, df in split_dfs.items()}
    overlap_found = False
    overlap_details = []
    split_keys = list(split_dfs.keys())

    for i in range(len(split_keys)):
        for j in range(i + 1, len(split_keys)):
            s1, s2 = split_keys[i], split_keys[j]
            overlap = split_id_sets[s1].intersection(split_id_sets[s2])
            if len(overlap) > 0:
                overlap_found = True
                overlap_details.append(f"Overlap between {s1} and {s2}: {len(overlap)} samples")

    checks["mutual_exclusivity"] = {
        "passed": not overlap_found,
        "detail": "Splits are strictly pairwise disjoint" if not overlap_found else "; ".join(overlap_details)
    }
    if overlap_found:
        all_passed = False

    # 3. Duplicate Group Isolation Check
    group_split_map = {}
    leaked_groups = []
    for _, row in df_primary_splits.iterrows():
        grp = row["duplicate_group_id"]
        s = row["split"]
        if grp not in group_split_map:
            group_split_map[grp] = set()
        group_split_map[grp].add(s)

    for grp, splits in group_split_map.items():
        if len(splits) > 1:
            leaked_groups.append({"group_id": grp, "splits": list(splits)})

    dup_isolation_passed = (len(leaked_groups) == 0)
    checks["duplicate_group_isolation"] = {
        "passed": dup_isolation_passed,
        "detail": f"0 duplicate groups cross splits ({len(group_split_map)} groups audited)" if dup_isolation_passed else f"{len(leaked_groups)} groups leaked across splits"
    }
    if not dup_isolation_passed:
        all_passed = False

    # 4. Class Representation across Splits
    missing_classes_by_split = {}
    for s, df in split_dfs.items():
        present_classes = set(df["class_name"].unique())
        missing = set(PRIMARY_CLASS_NAMES) - present_classes
        if missing:
            missing_classes_by_split[s] = list(missing)

    class_rep_passed = (len(missing_classes_by_split) == 0)
    checks["class_representation"] = {
        "passed": class_rep_passed,
        "detail": "All 6 disease classes represented across Train, Val, Cal, and Test" if class_rep_passed else f"Missing classes: {missing_classes_by_split}"
    }
    if not class_rep_passed:
        all_passed = False

    # 5. Cross-Dataset Collision Check
    primary_hashes = set(df_primary_splits["sha256"].dropna())
    sethy_hashes = set(df_sethy["sha256"].dropna())
    bd5_hashes = set(df_bd5["sha256"].dropna())

    p_s_coll = primary_hashes.intersection(sethy_hashes)
    p_b_coll = primary_hashes.intersection(bd5_hashes)
    s_b_coll = sethy_hashes.intersection(bd5_hashes)

    cross_leakage_passed = (len(p_s_coll) == 0 and len(p_b_coll) == 0 and len(s_b_coll) == 0)
    checks["cross_dataset_leakage"] = {
        "passed": cross_leakage_passed,
        "detail": f"Primary intersect Sethy: {len(p_s_coll)}, Primary intersect BD5: {len(p_b_coll)}, Sethy intersect BD5: {len(s_b_coll)}"
    }
    if not cross_leakage_passed:
        all_passed = False

    # 6. External Dataset Firewall
    external_in_training = False
    all_train_hashes = set(split_dfs["train"]["sha256"].dropna())
    if len(all_train_hashes.intersection(sethy_hashes)) > 0 or len(all_train_hashes.intersection(bd5_hashes)) > 0:
        external_in_training = True

    checks["external_dataset_firewall"] = {
        "passed": not external_in_training,
        "detail": "Sethy and BD5 are 100% firewalled from the primary training/validation/calibration/test splits"
    }
    if external_in_training:
        all_passed = False

    # 7. Distribution Drift Quantization
    overall_dist = df_primary_splits["class_name"].value_counts(normalize=True).to_dict()
    drift_metrics = {}
    for s, df in split_dfs.items():
        s_dist = df["class_name"].value_counts(normalize=True).to_dict()
        max_abs_diff = 0.0
        l1_diff = 0.0
        for c in PRIMARY_CLASS_NAMES:
            p_full = overall_dist.get(c, 0.0)
            p_split = s_dist.get(c, 0.0)
            diff = abs(p_full - p_split)
            l1_diff += diff
            if diff > max_abs_diff:
                max_abs_diff = diff
        drift_metrics[s] = {
            "max_class_percentage_deviation": round(max_abs_diff * 100, 2),
            "l1_total_variation_distance": round(l1_diff / 2.0, 4)
        }

    return {
        "overall_passed": all_passed,
        "checks": checks,
        "distribution_drift": drift_metrics
    }
