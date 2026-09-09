"""Phase 1 Comprehensive Markdown Report and Run Metadata Generator."""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.phase1.label_analysis import PRIMARY_CLASS_NAMES

def generate_all_phase1_reports(
    env_info: Dict[str, Any],
    label_summary: Dict[str, Any],
    duplicate_summary: Dict[str, Any],
    split_summary: Dict[str, Any],
    validation_results: Dict[str, Any],
    dim_stats_primary: Dict[str, Any],
    manifest_checksums: Dict[str, str],
    output_dir: str
) -> Dict[str, str]:
    """
    Generate all Phase 1 documentation reports and run metadata JSON.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    report_paths = {}

    # 1. Dataset Statistics Report
    stat_lines = [
        "# Phase 1: Comprehensive Dataset Statistics & Profiling Report",
        "",
        "**Project:** XAI-RiceGuard  ",
        f"**Date (UTC):** {env_info.get('timestamp')}  ",
        f"**Runtime:** {'Google Colab Free Tier' if env_info.get('is_colab') else 'Local Environment'}  ",
        f"**Git Commit:** `{env_info.get('git_commit')}`  ",
        "",
        "---",
        "",
        "## 1. Primary Dataset Overview (RiceLeafDiseaseBD)",
        f"- **Total Primary Images:** `{label_summary['total_samples']:,}`",
        f"- **Number of Disease Classes:** `{label_summary['num_classes']}`",
        f"- **Largest Class:** `{label_summary['largest_class']}` | **Smallest Class:** `{label_summary['smallest_class']}`",
        f"- **Imbalance Ratio:** `{label_summary['imbalance_ratio']}:1`",
        "",
        "### Class Distribution Table",
        "| Class ID | Class Name | Sample Count | Percentage (%) |",
        "| :---: | :--- | :---: | :---: |"
    ]
    for idx, c_name in enumerate(PRIMARY_CLASS_NAMES):
        cnt = label_summary["class_counts"].get(c_name, 0)
        pct = label_summary["class_percentages"].get(c_name, 0.0)
        stat_lines.append(f"| {idx} | `{c_name}` | {cnt:,} | {pct:.2f}% |")

    stat_lines.extend([
        "",
        "---",
        "",
        "## 2. Duplicate Group Analysis",
        f"- **Total Unique SHA-256 Hashes:** `{duplicate_summary['total_unique_hashes']:,}`",
        f"- **Duplicate Hash Groups:** `{duplicate_summary['total_duplicate_groups']:,}`",
        f"- **Unique (Single-Sample) Groups:** `{duplicate_summary['total_unique_groups']:,}`",
        f"- **Samples in Duplicate Clusters:** `{duplicate_summary['total_images_in_duplicate_groups']:,}`",
        f"- **Samples in Unique Groups:** `{duplicate_summary['total_images_in_unique_groups']:,}`",
        f"- **Max Group Cluster Size:** `{duplicate_summary['max_images_in_single_group']}`",
        "",
        "---",
        "",
        "## 3. Image Dimensions and Resolution Profiling",
        "| Metric | Width (px) | Height (px) | Aspect Ratio | File Size (KB) |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])
    d_info = dim_stats_primary["dimensions"]
    stat_lines.append(f"| **Mean** | {d_info['width']['mean']} | {d_info['height']['mean']} | {d_info['aspect_ratio']['mean']} | {d_info['file_size_kb']['mean']} |")
    stat_lines.append(f"| **Median** | {d_info['width']['median']} | {d_info['height']['median']} | {d_info['aspect_ratio']['median']} | {d_info['file_size_kb']['median']} |")
    stat_lines.append(f"| **Min** | {d_info['width']['min']} | {d_info['height']['min']} | {d_info['aspect_ratio']['min']} | {d_info['file_size_kb']['min']} |")
    stat_lines.append(f"| **Max** | {d_info['width']['max']} | {d_info['height']['max']} | {d_info['aspect_ratio']['max']} | {d_info['file_size_kb']['max']} |")

    stat_file = out_p / "phase1_dataset_statistics.md"
    with open(stat_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stat_lines))
    report_paths["dataset_statistics"] = str(stat_file.resolve())

    # 2. Split Report
    split_lines = [
        "# Phase 1: Group-Aware Stratified Split & Leakage Report",
        "",
        "**Project:** XAI-RiceGuard  ",
        f"**Random Seed:** `{split_summary['seed']}`  ",
        "**Splitting Strategy:** Constrained optimization of class distribution alignment subject to atomic duplicate group isolation ($G_i \\cap G_j = \\emptyset$).",
        "",
        "---",
        "",
        "## 1. Experimental Split Summary",
        "",
        "| Partition | Target Count | Actual Count | Target % | Actual % |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for s_name, s_info in split_summary["splits"].items():
        split_lines.append(f"| **{s_name.replace('_', ' ').title()}** | {s_info['target_count']:,} | {s_info['sample_count']:,} | {s_info['target_percentage']}% | {s_info['actual_percentage']}% |")

    split_lines.extend([
        "",
        "---",
        "",
        "## 2. Class Distribution by Partition",
        "",
        "| Disease Class | Train | Validation | Calibration | Internal Test | Total |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ])
    for cls in PRIMARY_CLASS_NAMES:
        t_c = split_summary["splits"]["train"]["class_counts"].get(cls, 0)
        v_c = split_summary["splits"]["validation"]["class_counts"].get(cls, 0)
        c_c = split_summary["splits"]["calibration"]["class_counts"].get(cls, 0)
        e_c = split_summary["splits"]["internal_test"]["class_counts"].get(cls, 0)
        tot = t_c + v_c + c_c + e_c
        split_lines.append(f"| `{cls}` | {t_c:,} | {v_c:,} | {c_c:,} | {e_c:,} | {tot:,} |")

    split_lines.extend([
        "",
        "---",
        "",
        "## 3. Cryptographic Manifest Checksums (Frozen Split Baseline)",
        "",
        "| Manifest File | SHA-256 Checksum | Lifecycle Role |",
        "| :--- | :--- | :--- |"
    ])
    for m_name, sha in manifest_checksums.items():
        role = "Frozen Internal Test (LOCKED)" if "internal_test" in m_name else "Experimental Partition"
        split_lines.append(f"| `{m_name}` | `{sha}` | {role} |")

    split_file = out_p / "phase1_split_report.md"
    with open(split_file, "w", encoding="utf-8") as f:
        f.write("\n".join(split_lines))
    report_paths["split_report"] = str(split_file.resolve())

    # 3. Validation Report
    val_lines = [
        "# Phase 1: Experimental Split & Leakage Validation Report",
        "",
        "**Project:** XAI-RiceGuard  ",
        f"**Validation Status:** `{'PASS' if validation_results['overall_passed'] else 'FAIL'}`  ",
        "",
        "---",
        "",
        "## 1. Hard-Gate Integrity Checklist",
        "",
        "| Integrity Check | Status | Verification Detail |",
        "| :--- | :---: | :--- |"
    ]
    for chk_name, chk_info in validation_results["checks"].items():
        val_lines.append(f"| **{chk_name.replace('_', ' ').title()}** | `{'PASS' if chk_info['passed'] else 'FAIL'}` | {chk_info['detail']} |")

    val_lines.extend([
        "",
        "---",
        "",
        "## 2. Partition Distribution Drift Metrics",
        "",
        "| Partition | Max Class % Deviation | $L_1$ Total Variation Distance |",
        "| :--- | :---: | :---: |"
    ])
    for s_name, drift in validation_results.get("distribution_drift", {}).items():
        val_lines.append(f"| **{s_name.replace('_', ' ').title()}** | {drift['max_class_percentage_deviation']}% | {drift['l1_total_variation_distance']} |")

    val_file = out_p / "phase1_validation_report.md"
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines))
    report_paths["validation_report"] = str(val_file.resolve())

    # 4. Completion Report
    comp_lines = [
        "# Phase 1 Completion & Readiness Report",
        "",
        "**Project:** XAI-RiceGuard  ",
        f"**Phase 1 Status:** `{'PASS' if validation_results['overall_passed'] else 'FAIL'}`  ",
        f"**Readiness:** `{'PHASE 01 COMPLETE — READY FOR PHASE 02' if validation_results['overall_passed'] else 'PHASE 01 BLOCKED — CORRECTION REQUIRED'}`  ",
        "",
        "---",
        "",
        "### Key Phase 1 Invariants Established:",
        "1. **Zero Data Leakage:** Primary dataset partitioned at duplicate-group level ($G_i \\cap G_j = \\emptyset$). No duplicate hashes cross partitions.",
        "2. **Four-Way Stratification:** Primary dataset cleanly divided into Train (~70%), Validation (~10%), Calibration (~10%), and Internal Test (~10%).",
        "3. **External Dataset Firewall:** Sethy 5932, RiceSeg-5932, and BD5 are 100% held out from model development splits.",
        "4. **Frozen Internal Test:** `primary_internal_test.csv` is cryptographically hashed and permanently locked for final IEEE benchmarking."
    ]
    comp_file = out_p / "phase1_completion_report.md"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(comp_lines))
    report_paths["completion_report"] = str(comp_file.resolve())

    # 5. Run Metadata JSON
    metadata = {
        "project": "XAI-RiceGuard",
        "phase": 1,
        "timestamp_utc": env_info.get("timestamp"),
        "git_commit": env_info.get("git_commit"),
        "random_seed": split_summary["seed"],
        "environment": {
            "is_colab": env_info.get("is_colab"),
            "os": env_info.get("system"),
            "python": env_info.get("python_version"),
            "pytorch": env_info.get("packages", {}).get("torch")
        },
        "split_summary": split_summary,
        "validation_passed": validation_results["overall_passed"],
        "manifest_checksums": manifest_checksums
    }
    meta_file = out_p / "phase1_run_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    report_paths["run_metadata"] = str(meta_file.resolve())

    return report_paths
