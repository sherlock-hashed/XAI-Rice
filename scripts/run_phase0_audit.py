#!/usr/bin/env python3
"""
Phase 0 Audit Runner: XAI-RiceGuard
Executes the comprehensive Phase 0 environment snapshot, dataset discovery,
integrity audit, YOLO annotation audit, RiceSeg pairing audit, manifest generation,
and IEEE report generation.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.phase0.environment import detect_environment, generate_environment_report
from src.phase0.config import resolve_paths, load_project_config
from src.phase0.dataset_discovery import discover_dataset_structure
from src.phase0.image_audit import audit_images
from src.phase0.annotation_audit import audit_yolo_annotations, audit_dataset_metadata
from src.phase0.duplicate_audit import audit_duplicates, audit_cross_dataset_leakage_risk
from src.phase0.riceseg_audit import audit_sethy_riceseg_pairing
from src.phase0.manifest_generator import generate_dataset_manifest
from src.phase0.report_generator import generate_phase0_audit_report, generate_phase0_completion_report

def run_phase0_audit(force_env: str = None):
    print("=" * 75)
    print("  XAI-RiceGuard: Phase 0 Project Foundation & Dataset Audit")
    print("=" * 75)

    # 1. Environment Detection
    print("\n[1/7] Detecting Computational Environment & Libraries...")
    env_info = detect_environment()
    env_report_text = generate_environment_report(env_info)
    print(env_report_text)

    # 2. Path Resolution & Directory Verification
    print("\n[2/7] Resolving Configured Paths & Checking Project Directories...")
    paths = resolve_paths(force_env=force_env)
    proj_cfg = load_project_config()
    print(f"  Active Environment Mode: {paths['environment'].upper()}")
    print(f"  Project Root: {paths['project_root']}")

    # Save Environment Report
    reports_dir = Path(paths["artifacts"]["reports"])
    env_report_file = reports_dir / "phase0_environment_report.txt"
    with open(env_report_file, "w", encoding="utf-8") as f:
        f.write(env_report_text)
    print(f"  -> Environment report saved to: {env_report_file}")

    # 3. Dataset Discovery
    print("\n[3/7] Discovering Dataset File Hierarchy on Disk/Drive...")
    discovery = {}
    for ds_key, root_path in paths["dataset_roots"].items():
        disc = discover_dataset_structure(root_path)
        discovery[ds_key] = disc
        print(f"  - [{ds_key}] Exists: {disc['exists']} | Total Files: {disc.get('total_files', 0)} | Images: {disc.get('total_images', 0)} ({root_path})")

    # 4. Deep Image Integrity Audits
    print("\n[4/7] Performing Non-Destructive Image Integrity Audits (Dimensions, Modes, Hashes)...")
    audit_results = {}
    manifest_results = {}
    manifests_dir = paths["artifacts"]["manifests"]

    # 4.1 Primary Dataset (RiceLeafDiseaseBD)
    p_disc = discovery.get("primary_dataset", {})
    if p_disc.get("exists") and p_disc.get("total_images", 0) > 0:
        p_root = paths["dataset_roots"]["primary_dataset"]
        print("  -> Auditing Primary Dataset (RiceLeafDiseaseBD)...")
        p_images = [img for img in p_disc.get("sample_image_paths", [])] # We will discover all images
        # Discover all relative image paths
        all_p_images = []
        for dp, _, fnames in os.walk(p_root):
            for fn in fnames:
                if Path(fn).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    all_p_images.append(os.path.relpath(os.path.join(dp, fn), p_root))
        
        def extract_p_class(rel_p):
            parts = Path(rel_p).parts
            if len(parts) >= 2:
                return parts[1]
            return parts[0]

        p_audit = audit_images("RiceLeafDiseaseBD", p_root, all_p_images, compute_hashes=True, class_extractor=extract_p_class)
        audit_results["primary_dataset"] = p_audit
        m_info = generate_dataset_manifest("RiceLeafDiseaseBD", "PRIMARY_DEVELOPMENT_DATASET", p_audit["image_records"], manifests_dir)
        manifest_results["RiceLeafDiseaseBD"] = m_info
        print(f"     Audited {p_audit['total_images']} images. Readable: {p_audit['readable_count']}. Manifest generated.")

    # 4.2 Sethy External Dataset
    s_disc = discovery.get("sethy_external", {})
    if s_disc.get("exists") and s_disc.get("total_images", 0) > 0:
        s_root = paths["dataset_roots"]["sethy_external"]
        print("  -> Auditing Sethy External Dataset...")
        all_s_images = []
        for dp, _, fnames in os.walk(s_root):
            for fn in fnames:
                if Path(fn).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    all_s_images.append(os.path.relpath(os.path.join(dp, fn), s_root))
        
        s_audit = audit_images("Sethy_Rice_Leaf_Disease", s_root, all_s_images, compute_hashes=True)
        audit_results["sethy_external"] = s_audit
        m_info = generate_dataset_manifest("Sethy_Rice_Leaf_Disease", "EXTERNAL_DATASET", s_audit["image_records"], manifests_dir)
        manifest_results["Sethy_Rice_Leaf_Disease"] = m_info
        print(f"     Audited {s_audit['total_images']} images. Readable: {s_audit['readable_count']}. Manifest generated.")

    # 4.3 BD5 External Dataset
    b_disc = discovery.get("bd5_external", {})
    if b_disc.get("exists") and b_disc.get("total_images", 0) > 0:
        b_root = paths["dataset_roots"]["bd5_external"]
        print("  -> Auditing BD5 External Dataset...")
        all_b_images = []
        for dp, _, fnames in os.walk(b_root):
            for fn in fnames:
                if Path(fn).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    all_b_images.append(os.path.relpath(os.path.join(dp, fn), b_root))
        
        b_audit = audit_images("RiceLeafDisease_BD5", b_root, all_b_images, compute_hashes=True)
        audit_results["bd5_external"] = b_audit
        m_info = generate_dataset_manifest("RiceLeafDisease_BD5", "EXTERNAL_FIELD_DATASET", b_audit["image_records"], manifests_dir)
        manifest_results["RiceLeafDisease_BD5"] = m_info
        print(f"     Audited {b_audit['total_images']} images. Readable: {b_audit['readable_count']}. Manifest generated.")

    # 5. Annotation & Pairing Audits
    print("\n[5/7] Auditing Annotations, Metadata, and Image-Mask Pairings...")
    # 5.1 RiceLeafDiseaseBD YOLO Annotations
    p_parent = str(Path(paths["dataset_roots"]["primary_dataset"]).parent)
    yolo_audit = audit_yolo_annotations(paths["dataset_roots"]["primary_dataset"])
    meta_audit = audit_dataset_metadata(p_parent)
    print(f"  -> YOLO Bounding Box Audit: {yolo_audit.get('total_bboxes', 0)} bboxes in {yolo_audit.get('total_label_files', 0)} label files.")
    print(f"     Syntax Errors: {yolo_audit.get('total_syntax_errors', 0)}, Out-of-bounds: {yolo_audit.get('total_out_of_bounds', 0)}")

    # 5.2 Sethy <-> RiceSeg-5932 Mask Pairing
    riceseg_audit = audit_sethy_riceseg_pairing(
        paths["dataset_roots"]["sethy_external"],
        paths["dataset_roots"]["riceseg_ground_truth"]
    )
    p_pair = riceseg_audit.get("pairing_summary", {})
    print(f"  -> Sethy <-> RiceSeg Pairing: {p_pair.get('exact_matches', 0)} matched pairs, {p_pair.get('unmatched_images', 0)} unmatched images, {p_pair.get('dimension_mismatches', 0)} dim mismatches.")

    # 6. Duplicates & Cross-Dataset Leakage Risk
    print("\n[6/7] Auditing Duplicates & Cross-Dataset Leakage...")
    all_dataset_records = {
        ds_k: res["image_records"] for ds_k, res in audit_results.items()
    }
    dup_intra_count = sum(audit_duplicates(recs)["total_duplicate_hashes"] for recs in all_dataset_records.values())
    cross_dup = audit_cross_dataset_leakage_risk(all_dataset_records)
    dup_summary = {
        "intra_dataset_duplicates_total": dup_intra_count,
        "cross_dataset_collisions_found": cross_dup.get("cross_dataset_collisions_found", False),
        "cross_dup_details": cross_dup
    }
    print(f"  -> Intra-dataset duplicate hashes found: {dup_intra_count}")
    print(f"  -> Cross-dataset collision risk: {'FOUND (Review Leakage Policy)' if cross_dup.get('cross_dataset_collisions_found') else 'NONE DETECTED'}")

    # 7. Report Generation & Hard Gate Verification
    print("\n[7/7] Generating IEEE-Ready Markdown Audit & Completion Reports...")
    audit_report_file = reports_dir / "phase0_dataset_audit.md"
    completion_report_file = reports_dir / "phase0_completion_report.md"

    generate_phase0_audit_report(
        env_info=env_info,
        discovery_results=discovery,
        audit_results=audit_results,
        yolo_results=yolo_audit,
        metadata_results=meta_audit,
        riceseg_results=riceseg_audit,
        duplicate_results=dup_summary,
        manifest_results=manifest_results,
        output_filepath=str(audit_report_file)
    )
    print(f"  -> Detailed Dataset Audit Report saved to: {audit_report_file}")

    audit_summary = {
        "primary_unreadable": audit_results.get("primary_dataset", {}).get("unreadable_count", 0),
        "sethy_unreadable": audit_results.get("sethy_external", {}).get("unreadable_count", 0),
        "riceseg_unmatched": p_pair.get("unmatched_images", 0),
        "bd5_unreadable": audit_results.get("bd5_external", {}).get("unreadable_count", 0),
    }

    generate_phase0_completion_report(
        env_info=env_info,
        audit_summary=audit_summary,
        output_filepath=str(completion_report_file)
    )
    print(f"  -> Phase 0 Completion Report saved to: {completion_report_file}")

    print("\n" + "=" * 75)
    print("  PHASE 0 AUDIT COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    run_phase0_audit()
