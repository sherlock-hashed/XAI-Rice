"""Duplicate Group Construction and Split Isolation Verification Module."""

from typing import Dict, Any, Tuple, List
import pandas as pd
from collections import defaultdict

def build_duplicate_groups(df_primary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Construct deterministic duplicate groups strictly using verified SHA-256 hashes.
    
    Assigns:
      - DUPLICATE_GROUP_0001 ... to hashes appearing > 1 times (duplicate clusters)
      - UNIQUE_GROUP_000001 ... to hashes appearing exactly 1 time
    """
    df = df_primary.copy()
    
    # Map hash frequencies
    hash_counts = df["sha256"].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1].index.tolist()
    unique_hashes = hash_counts[hash_counts == 1].index.tolist()

    # Sort deterministically for 100% reproducibility
    duplicate_hashes.sort()
    unique_hashes.sort()

    hash_to_group = {}
    
    # Assign duplicate group IDs
    for idx, h in enumerate(duplicate_hashes, start=1):
        hash_to_group[h] = f"DUPLICATE_GROUP_{idx:04d}"

    # Assign unique group IDs
    for idx, h in enumerate(unique_hashes, start=1):
        hash_to_group[h] = f"UNIQUE_GROUP_{idx:06d}"

    df["duplicate_group_id"] = df["sha256"].map(hash_to_group)
    df["is_duplicate"] = df["sha256"].isin(duplicate_hashes)

    # Build duplicate groups table
    group_records = []
    for grp_id, grp_df in df.groupby("duplicate_group_id"):
        group_records.append({
            "duplicate_group_id": grp_id,
            "sha256": grp_df["sha256"].iloc[0],
            "sample_count": len(grp_df),
            "is_duplicate_group": (len(grp_df) > 1),
            "classes": sorted(grp_df["class_name"].unique().tolist()),
            "sample_ids": sorted(grp_df["sample_id"].tolist())
        })

    df_groups = pd.DataFrame(group_records).sort_values("duplicate_group_id").reset_index(drop=True)

    summary = {
        "total_images": len(df),
        "total_unique_hashes": len(hash_counts),
        "duplicate_hash_count": len(duplicate_hashes),
        "unique_hash_count": len(unique_hashes),
        "total_duplicate_groups": len(duplicate_hashes),
        "total_unique_groups": len(unique_hashes),
        "total_images_in_duplicate_groups": int(df["is_duplicate"].sum()),
        "total_images_in_unique_groups": int((~df["is_duplicate"]).sum()),
        "max_images_in_single_group": int(hash_counts.max()) if not hash_counts.empty else 0
    }

    return df, df_groups, summary

def validate_duplicate_group_isolation(df_splits: pd.DataFrame) -> Dict[str, Any]:
    """
    Formally verify that NO duplicate_group_id appears in more than one split partition.
    Constraint: For every group g, |unique(split_assignments(g))| == 1.
    """
    group_split_map = defaultdict(set)
    for _, row in df_splits.iterrows():
        grp = row["duplicate_group_id"]
        split = row["split"]
        group_split_map[grp].add(split)

    leaked_groups = {
        grp: list(splits) for grp, splits in group_split_map.items() if len(splits) > 1
    }

    passed = (len(leaked_groups) == 0)

    return {
        "passed": passed,
        "total_groups_audited": len(group_split_map),
        "leaked_groups_count": len(leaked_groups),
        "leaked_groups_sample": list(leaked_groups.items())[:10]
    }
