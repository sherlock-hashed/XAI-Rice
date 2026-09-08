"""Manifest Generation Module for Phase 0."""

import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def generate_dataset_manifest(
    dataset_name: str,
    dataset_role: str,
    image_records: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, str]:
    """
    Generate machine-readable CSV and JSON manifests for audited datasets.
    Calculates and returns manifest checksum for reproducible experiment tracking.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_manifest.csv"
    json_filename = f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_manifest.json"

    csv_path = out_path / csv_filename
    json_path = out_path / json_filename

    # Enrich records with dataset role
    enriched_records = []
    for r in image_records:
        rec_copy = dict(r)
        rec_copy["dataset_role"] = dataset_role
        if isinstance(rec_copy.get("anomalies"), list):
            rec_copy["anomalies"] = "; ".join(rec_copy["anomalies"])
        enriched_records.append(rec_copy)

    # Save to CSV
    df = pd.DataFrame(enriched_records)
    df.to_csv(str(csv_path), index=False, encoding="utf-8")

    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": dataset_name,
            "dataset_role": dataset_role,
            "total_records": len(enriched_records),
            "records": enriched_records
        }, f, indent=2)

    # Compute manifest SHA-256 for tracking
    hasher = hashlib.sha256()
    with open(csv_path, "rb") as f:
        hasher.update(f.read())
    csv_sha256 = hasher.hexdigest()

    return {
        "dataset_name": dataset_name,
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "total_records": len(enriched_records),
        "csv_sha256": csv_sha256
    }
