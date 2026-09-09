"""Primary Dataset Label Analysis and Class Distribution Module."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Canonical class ordering for RiceLeafDiseaseBD
PRIMARY_CLASS_NAMES = [
    "Healthy",
    "Blast",
    "Brown spot",
    "Leaf smut",
    "Rice Tungro",
    "Sheath blight"
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(PRIMARY_CLASS_NAMES)}

def analyze_primary_labels(df_primary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform rigorous label analysis and calculate distribution metrics on RiceLeafDiseaseBD.
    """
    df = df_primary.copy()
    
    # Map class IDs deterministically
    df["class_id"] = df["class_name"].map(CLASS_NAME_TO_ID)
    
    total_images = len(df)
    class_counts = df["class_name"].value_counts().to_dict()
    
    distribution_rows = []
    for cls_name in PRIMARY_CLASS_NAMES:
        cnt = class_counts.get(cls_name, 0)
        pct = round((cnt / total_images) * 100, 4) if total_images > 0 else 0.0
        distribution_rows.append({
            "class_id": CLASS_NAME_TO_ID[cls_name],
            "class_name": cls_name,
            "sample_count": cnt,
            "percentage": pct
        })

    df_dist = pd.DataFrame(distribution_rows)
    
    counts_list = [r["sample_count"] for r in distribution_rows if r["sample_count"] > 0]
    max_c = max(counts_list) if counts_list else 0
    min_c = min(counts_list) if counts_list else 1
    imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else 0.0

    summary = {
        "total_samples": total_images,
        "num_classes": len(PRIMARY_CLASS_NAMES),
        "class_counts": {r["class_name"]: r["sample_count"] for r in distribution_rows},
        "class_percentages": {r["class_name"]: r["percentage"] for r in distribution_rows},
        "largest_class": max(distribution_rows, key=lambda x: x["sample_count"])["class_name"],
        "smallest_class": min(distribution_rows, key=lambda x: x["sample_count"])["class_name"],
        "imbalance_ratio": imbalance_ratio,
        "all_expected_classes_present": all(r["sample_count"] > 0 for r in distribution_rows)
    }

    return df_dist, summary

def export_class_distributions(
    df_dist: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """Export class distribution to CSV and JSON."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    csv_path = out_p / "primary_class_distribution.csv"
    json_path = out_p / "primary_class_distribution.json"

    df_dist.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve())
    }
