"""YOLO Annotation & Metadata Audit Module for RiceLeafDiseaseBD (Phase 0)."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

def audit_single_yolo_file(file_path: str) -> Dict[str, Any]:
    """
    Audit a single YOLO format .txt annotation file.
    Validates coordinate ranges [0, 1], class IDs, and syntax.
    """
    path = Path(file_path)
    bboxes = []
    syntax_errors = []
    out_of_bounds = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line_idx, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                syntax_errors.append(f"Line {line_idx}: Expected 5 fields, got {len(parts)} ('{line}')")
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Validate normalized bounding box coordinates
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                    out_of_bounds.append(f"Line {line_idx}: Coordinates out of [0, 1] range: ({x_center}, {y_center}, {width}, {height})")

                bboxes.append({
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                })
            except ValueError as ve:
                syntax_errors.append(f"Line {line_idx}: Value parse error ({ve}) in line '{line}'")

    except Exception as e:
        return {
            "filename": path.name,
            "readable": False,
            "error": str(e),
            "bbox_count": 0,
            "bboxes": [],
            "syntax_errors": [str(e)],
            "out_of_bounds": []
        }

    return {
        "filename": path.name,
        "readable": True,
        "error": None,
        "bbox_count": len(bboxes),
        "bboxes": bboxes,
        "syntax_errors": syntax_errors,
        "out_of_bounds": out_of_bounds
    }

def audit_yolo_annotations(
    primary_dataset_root: str
) -> Dict[str, Any]:
    """
    Audit all YOLO annotation files and visual rendering folders in RiceLeafDiseaseBD.
    Verifies correspondence between:
      - Original images
      - Visual annotated images
      - YOLO label text files
    """
    root = Path(primary_dataset_root)
    annotated_dir = root / "Annotated images ( visual with labels)"
    original_dir = root / "Original images"

    if not annotated_dir.exists():
        return {
            "has_annotations": False,
            "error": f"Annotated images directory not found at: {annotated_dir}"
        }

    results = {
        "has_annotations": True,
        "annotated_directory": str(annotated_dir.resolve()),
        "original_directory": str(original_dir.resolve()) if original_dir.exists() else "Not found",
        "classes_audited": {},
        "total_label_files": 0,
        "total_visual_files": 0,
        "total_bboxes": 0,
        "total_syntax_errors": 0,
        "total_out_of_bounds": 0,
        "pairing_with_originals": {
            "matched": 0,
            "unmatched_labels": 0,
            "unmatched_visuals": 0,
            "unannotated_originals": 0
        }
    }

    # Discover classes inside annotated directory
    classes = [d.name for d in annotated_dir.iterdir() if d.is_dir()]

    for cls_name in classes:
        cls_dir = annotated_dir / cls_name
        labels_dir = cls_dir / "labels"
        visuals_dir = cls_dir / "visuals"
        orig_cls_dir = original_dir / cls_name if original_dir.exists() else None

        cls_stat = {
            "labels_dir_exists": labels_dir.exists(),
            "visuals_dir_exists": visuals_dir.exists(),
            "label_file_count": 0,
            "visual_file_count": 0,
            "original_image_count": 0,
            "total_bboxes": 0,
            "class_ids_found": set(),
            "syntax_errors_count": 0,
            "out_of_bounds_count": 0,
            "sample_bboxes": []
        }

        orig_stems = set()
        if orig_cls_dir and orig_cls_dir.exists():
            orig_images = [f for f in orig_cls_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            cls_stat["original_image_count"] = len(orig_images)
            orig_stems = {f.stem for f in orig_images}

        label_stems = set()
        if labels_dir.exists():
            label_files = [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
            cls_stat["label_file_count"] = len(label_files)
            results["total_label_files"] += len(label_files)

            for lf in label_files:
                audit_res = audit_single_yolo_file(str(lf))
                cls_stat["total_bboxes"] += audit_res["bbox_count"]
                results["total_bboxes"] += audit_res["bbox_count"]
                
                for b in audit_res["bboxes"]:
                    cls_stat["class_ids_found"].add(b["class_id"])

                if audit_res["syntax_errors"]:
                    cls_stat["syntax_errors_count"] += len(audit_res["syntax_errors"])
                    results["total_syntax_errors"] += len(audit_res["syntax_errors"])
                
                if audit_res["out_of_bounds"]:
                    cls_stat["out_of_bounds_count"] += len(audit_res["out_of_bounds"])
                    results["total_out_of_bounds"] += len(audit_res["out_of_bounds"])

                label_stems.add(lf.stem)

        if visuals_dir.exists():
            visual_files = [f for f in visuals_dir.iterdir() if f.is_file()]
            cls_stat["visual_file_count"] = len(visual_files)
            results["total_visual_files"] += len(visual_files)

        # Pairing calculations
        if orig_stems:
            matched = len(orig_stems.intersection(label_stems))
            unmatched_lbl = len(label_stems - orig_stems)
            unannotated_orig = len(orig_stems - label_stems)
            results["pairing_with_originals"]["matched"] += matched
            results["pairing_with_originals"]["unmatched_labels"] += unmatched_lbl
            results["pairing_with_originals"]["unannotated_originals"] += unannotated_orig

        cls_stat["class_ids_found"] = sorted(list(cls_stat["class_ids_found"]))
        results["classes_audited"][cls_name] = cls_stat

    return results

def audit_dataset_metadata(primary_dataset_root: str) -> Dict[str, Any]:
    """
    Check for presence of documentation and metadata files:
    - Dataset metadata.xlsx
    - Annotation Protocol.pdf
    - README.md
    - Data Collection Pipeline.png
    - Dataset folder Structure.png
    Searches current and ancestor directory levels.
    """
    start_dir = Path(primary_dataset_root).resolve()
    candidates_dirs = [start_dir, start_dir.parent, start_dir.parent.parent, start_dir.parent.parent.parent]
    found_docs = {}

    expected_files = [
        "Dataset metadata.xlsx",
        "Annotation Protocol.pdf",
        "README.md",
        "Data Collection Pipeline.png",
        "Dataset folder Structure.png"
    ]

    for fname in expected_files:
        found = False
        for cdir in candidates_dirs:
            fpath = cdir / fname
            if fpath.exists():
                found_docs[fname] = {
                    "exists": True,
                    "size_bytes": fpath.stat().st_size,
                    "path": str(fpath.resolve())
                }
                found = True
                break
        if not found:
            found_docs[fname] = {
                "exists": False,
                "size_bytes": 0,
                "path": None
            }

    # Inspect Excel metadata if available
    xlsx_info = {}
    xlsx_path = found_docs.get("Dataset metadata.xlsx", {}).get("path")
    if xlsx_path and Path(xlsx_path).exists():
        try:
            excel_file = pd.ExcelFile(str(xlsx_path))
            xlsx_info["sheet_names"] = excel_file.sheet_names
            df_sample = pd.read_excel(str(xlsx_path), sheet_name=excel_file.sheet_names[0], nrows=5)
            xlsx_info["columns"] = list(df_sample.columns)
            xlsx_info["sample_row_count"] = len(df_sample)
            xlsx_info["readable"] = True
        except Exception as e:
            xlsx_info["readable"] = False
            xlsx_info["error"] = str(e)

    return {
        "documentation_files": found_docs,
        "excel_metadata_summary": xlsx_info
    }
