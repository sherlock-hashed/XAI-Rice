"""Phase 1 Manifest Loader and Integrity Validation Module."""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from src.phase0.config import resolve_paths

EXPECTED_COUNTS = {
    "primary": 17963,
    "sethy": 5932,
    "bd5": 3150,
    "riceseg": 5932
}

def load_phase0_manifests(
    manifests_dir: str = None,
    force_env: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Load authoritative Phase 0 manifests and normalize column schemas.
    """
    if manifests_dir is None:
        paths = resolve_paths(force_env=force_env)
        manifests_dir = paths["artifacts"]["manifests"]

    m_dir = Path(manifests_dir)
    primary_file = m_dir / "riceleafdiseasebd_manifest.csv"
    sethy_file = m_dir / "sethy_rice_leaf_disease_manifest.csv"
    bd5_file = m_dir / "riceleafdisease_bd5_manifest.csv"

    if not primary_file.exists():
        raise FileNotFoundError(f"Primary manifest not found: {primary_file}")
    if not sethy_file.exists():
        raise FileNotFoundError(f"Sethy manifest not found: {sethy_file}")
    if not bd5_file.exists():
        raise FileNotFoundError(f"BD5 manifest not found: {bd5_file}")

    df_primary = pd.read_csv(primary_file)
    df_sethy = pd.read_csv(sethy_file)
    df_bd5 = pd.read_csv(bd5_file)

    # Standardize sample_id across manifests
    df_primary["sample_id"] = [f"PRIMARY_{i:06d}" for i in range(len(df_primary))]
    df_sethy["sample_id"] = [f"SETHY_{i:06d}" for i in range(len(df_sethy))]
    df_bd5["sample_id"] = [f"BD5_{i:06d}" for i in range(len(df_bd5))]

    # Link annotation paths for primary dataset
    paths = resolve_paths(force_env=force_env)
    p_root = Path(paths["dataset_roots"]["primary_dataset"])
    
    annotation_paths = []
    for _, row in df_primary.iterrows():
        rel_p = str(row["relative_path"])
        stem = Path(rel_p).stem
        cls = str(row["class_name"])
        # Check standard YOLO label path
        candidate_lbl = p_root / "Annotated images ( visual with labels)" / cls / "labels" / f"{stem}.txt"
        if candidate_lbl.exists():
            annotation_paths.append(str(candidate_lbl.resolve()))
        else:
            annotation_paths.append(None)
            
    df_primary["annotation_path"] = annotation_paths

    # Link RiceSeg ground truth masks to Sethy
    r_root = Path(paths["dataset_roots"]["riceseg_ground_truth"])
    mask_paths = []
    for _, row in df_sethy.iterrows():
        cls = str(row["class_name"])
        fname = str(row["filename"])
        candidate_mask = r_root / cls / fname
        if candidate_mask.exists():
            mask_paths.append(str(candidate_mask.resolve()))
        else:
            # Check with .png extension
            candidate_mask_png = r_root / cls / f"{Path(fname).stem}.png"
            if candidate_mask_png.exists():
                mask_paths.append(str(candidate_mask_png.resolve()))
            else:
                mask_paths.append(None)
    df_sethy["mask_path"] = mask_paths

    return {
        "primary": df_primary,
        "sethy": df_sethy,
        "bd5": df_bd5
    }

def validate_manifest_completeness(manifests: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Verify that loaded manifest row counts exactly match the audited Phase 0 baseline counts.
    """
    results = {}
    all_passed = True

    for key, expected_count in EXPECTED_COUNTS.items():
        if key == "riceseg":
            # Verified via sethy mask_path matches
            matched_masks = int(manifests["sethy"]["mask_path"].notna().sum())
            passed = (matched_masks == expected_count)
            results[key] = {
                "expected": expected_count,
                "actual": matched_masks,
                "passed": passed
            }
            if not passed:
                all_passed = False
        elif key in manifests:
            actual_count = len(manifests[key])
            passed = (actual_count == expected_count)
            results[key] = {
                "expected": expected_count,
                "actual": actual_count,
                "passed": passed
            }
            if not passed:
                all_passed = False

    return {
        "all_passed": all_passed,
        "details": results
    }
