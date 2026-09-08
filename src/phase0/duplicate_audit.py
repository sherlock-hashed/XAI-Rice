"""Duplicate and Hash Collision Audit Module for Phase 0."""

from typing import Dict, Any, List
from collections import defaultdict

def audit_duplicates(image_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Audit intra-dataset filename duplicates and exact SHA-256 hash duplicates.
    """
    hash_to_records = defaultdict(list)
    filename_to_records = defaultdict(list)

    for rec in image_records:
        fname = rec.get("filename")
        sha = rec.get("sha256")
        if fname:
            filename_to_records[fname].append(rec)
        if sha:
            hash_to_records[sha].append(rec)

    exact_hash_duplicates = {
        h: recs for h, recs in hash_to_records.items() if len(recs) > 1
    }

    filename_collisions = {
        f: recs for f, recs in filename_to_records.items() if len(recs) > 1
    }

    # Format duplicate clusters for reporting
    duplicate_groups = []
    for sha, recs in exact_hash_duplicates.items():
        duplicate_groups.append({
            "sha256": sha,
            "count": len(recs),
            "files": [r.get("relative_path") for r in recs],
            "classes": list({r.get("class_name") for r in recs})
        })

    return {
        "total_unique_hashes": len(hash_to_records),
        "total_duplicate_hashes": len(exact_hash_duplicates),
        "total_files_in_duplicate_groups": sum(len(recs) for recs in exact_hash_duplicates.values()),
        "total_filename_collisions": len(filename_collisions),
        "duplicate_groups_sample": duplicate_groups[:25],
        "filename_collisions_sample": [
            {"filename": k, "count": len(v), "paths": [r.get("relative_path") for r in v]}
            for k, v in list(filename_collisions.items())[:25]
        ]
    }

def audit_cross_dataset_leakage_risk(
    dataset_records_dict: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Check for identical SHA-256 hashes across different datasets
    (e.g., Primary vs Sethy External vs BD5 External).
    Detects potential cross-contamination / data leakage.
    """
    dataset_hashes = {}
    for ds_name, records in dataset_records_dict.items():
        dataset_hashes[ds_name] = {
            r["sha256"]: r["relative_path"] for r in records if r.get("sha256")
        }

    cross_collisions = []
    dataset_names = list(dataset_hashes.keys())

    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            ds1 = dataset_names[i]
            ds2 = dataset_names[j]
            common_hashes = set(dataset_hashes[ds1].keys()).intersection(set(dataset_hashes[ds2].keys()))
            
            if common_hashes:
                cross_collisions.append({
                    "dataset_pair": (ds1, ds2),
                    "collision_count": len(common_hashes),
                    "sample_collisions": [
                        {
                            "sha256": h,
                            f"{ds1}_path": dataset_hashes[ds1][h],
                            f"{ds2}_path": dataset_hashes[ds2][h]
                        }
                        for h in list(common_hashes)[:10]
                    ]
                })

    return {
        "cross_dataset_collisions_found": len(cross_collisions) > 0,
        "collision_pairs_count": len(cross_collisions),
        "collision_details": cross_collisions
    }
