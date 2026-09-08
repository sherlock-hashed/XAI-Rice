"""RiceSeg-5932 Mask & Sethy Image-Mask Pairing Audit Module for Phase 0."""

import os
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import numpy as np

def inspect_mask_file(mask_full_path: str) -> Dict[str, Any]:
    """Inspect mask dimensions, mode, and unique pixel values non-destructively."""
    path = Path(mask_full_path)
    res = {
        "filename": path.name,
        "readable": False,
        "width": None,
        "height": None,
        "mode": None,
        "unique_values": [],
        "is_binary": False,
        "error": None
    }
    try:
        with Image.open(str(path)) as img:
            res["width"], res["height"] = img.size
            res["mode"] = img.mode
            res["readable"] = True
            
            # Sample unique values using numpy
            arr = np.array(img)
            uniques = np.unique(arr).tolist()
            res["unique_values"] = uniques
            res["is_binary"] = (len(uniques) <= 2)
    except Exception as e:
        res["error"] = str(e)
    return res

def audit_sethy_riceseg_pairing(
    sethy_root: str,
    riceseg_root: str
) -> Dict[str, Any]:
    """
    Perform a complete image-to-mask pairing audit between:
      1. Sethy External Dataset Images
      2. RiceSeg-5932 Ground Truth Masks
    """
    sethy_path = Path(sethy_root)
    riceseg_path = Path(riceseg_root)

    if not sethy_path.exists():
        return {"error": f"Sethy dataset directory not found: {sethy_root}"}
    if not riceseg_path.exists():
        return {"error": f"RiceSeg dataset directory not found: {riceseg_root}"}

    # Discover classes in Sethy
    sethy_classes = {d.name for d in sethy_path.iterdir() if d.is_dir()}
    # Discover classes in RiceSeg
    riceseg_classes = {d.name for d in riceseg_path.iterdir() if d.is_dir()}

    class_alignment = {
        "sethy_classes": sorted(list(sethy_classes)),
        "riceseg_classes": sorted(list(riceseg_classes)),
        "common_classes": sorted(list(sethy_classes.intersection(riceseg_classes))),
        "sethy_only": sorted(list(sethy_classes - riceseg_classes)),
        "riceseg_only": sorted(list(riceseg_classes - sethy_classes)),
    }

    pairing_summary = {
        "total_sethy_images": 0,
        "total_riceseg_masks": 0,
        "exact_matches": 0,
        "unmatched_images": 0,
        "unmatched_masks": 0,
        "dimension_mismatches": 0,
        "dimension_mismatch_samples": [],
        "unmatched_image_samples": [],
        "unmatched_mask_samples": [],
        "mask_properties_summary": {
            "modes": {},
            "binary_mask_count": 0,
            "non_binary_mask_count": 0,
            "unique_values_sample": {}
        },
        "per_class_pairing": {}
    }

    for cls in sorted(list(sethy_classes.union(riceseg_classes))):
        s_cls_dir = sethy_path / cls
        r_cls_dir = riceseg_path / cls

        s_files = {}
        if s_cls_dir.exists():
            for f in s_cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    s_files[f.stem] = f

        r_files = {}
        if r_cls_dir.exists():
            for f in r_cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    r_files[f.stem] = f

        pairing_summary["total_sethy_images"] += len(s_files)
        pairing_summary["total_riceseg_masks"] += len(r_files)

        s_stems = set(s_files.keys())
        r_stems = set(r_files.keys())

        matched_stems = s_stems.intersection(r_stems)
        unmatched_img = s_stems - r_stems
        unmatched_msk = r_stems - s_stems

        pairing_summary["exact_matches"] += len(matched_stems)
        pairing_summary["unmatched_images"] += len(unmatched_img)
        pairing_summary["unmatched_masks"] += len(unmatched_msk)

        if unmatched_img:
            pairing_summary["unmatched_image_samples"].extend(
                [f"{cls}/{s_files[st].name}" for st in list(unmatched_img)[:5]]
            )
        if unmatched_msk:
            pairing_summary["unmatched_mask_samples"].extend(
                [f"{cls}/{r_files[st].name}" for st in list(unmatched_msk)[:5]]
            )

        # Audit dimensions on a subset or all matched pairs
        dim_mismatches_in_class = 0
        # Check sample of matched files for dimensions & mask integrity
        for stem in list(matched_stems)[:50]:
            img_p = s_files[stem]
            msk_p = r_files[stem]
            
            try:
                with Image.open(str(img_p)) as img, Image.open(str(msk_p)) as msk:
                    if img.size != msk.size:
                        dim_mismatches_in_class += 1
                        pairing_summary["dimension_mismatches"] += 1
                        pairing_summary["dimension_mismatch_samples"].append({
                            "class": cls,
                            "image": img_p.name,
                            "img_dim": img.size,
                            "mask_dim": msk.size
                        })
            except Exception:
                pass

        # Inspect sample mask properties
        for stem in list(r_stems)[:20]:
            msk_p = r_files[stem]
            minfo = inspect_mask_file(str(msk_p))
            if minfo["readable"]:
                mode = minfo["mode"]
                pairing_summary["mask_properties_summary"]["modes"][mode] = (
                    pairing_summary["mask_properties_summary"]["modes"].get(mode, 0) + 1
                )
                if minfo["is_binary"]:
                    pairing_summary["mask_properties_summary"]["binary_mask_count"] += 1
                else:
                    pairing_summary["mask_properties_summary"]["non_binary_mask_count"] += 1
                
                val_key = str(minfo["unique_values"])
                pairing_summary["mask_properties_summary"]["unique_values_sample"][val_key] = (
                    pairing_summary["mask_properties_summary"]["unique_values_sample"].get(val_key, 0) + 1
                )

        pairing_summary["per_class_pairing"][cls] = {
            "sethy_images": len(s_files),
            "riceseg_masks": len(r_files),
            "matched_pairs": len(matched_stems),
            "unmatched_images": len(unmatched_img),
            "unmatched_masks": len(unmatched_msk),
            "dimension_mismatches": dim_mismatches_in_class
        }

    return {
        "class_alignment": class_alignment,
        "pairing_summary": pairing_summary
    }
