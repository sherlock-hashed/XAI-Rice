"""Phase 1: Leakage-Safe Dataset Preparation, Duplicate-Aware Stratified Splitting, EDA, and Frozen Manifest Protocol."""

from src.phase1.manifest_loader import load_phase0_manifests, validate_manifest_completeness
from src.phase1.duplicate_groups import build_duplicate_groups, validate_duplicate_group_isolation
from src.phase1.label_analysis import analyze_primary_labels, export_class_distributions
from src.phase1.split_generator import generate_group_aware_stratified_splits, freeze_manifests
from src.phase1.split_validator import validate_phase1_splits
from src.phase1.dataset_statistics import compute_image_dimension_statistics
from src.phase1.eda import generate_all_phase1_figures, select_representative_samples
from src.phase1.phase1_report import generate_all_phase1_reports

__all__ = [
    "load_phase0_manifests",
    "validate_manifest_completeness",
    "build_duplicate_groups",
    "validate_duplicate_group_isolation",
    "analyze_primary_labels",
    "export_class_distributions",
    "generate_group_aware_stratified_splits",
    "freeze_manifests",
    "validate_phase1_splits",
    "compute_image_dimension_statistics",
    "generate_all_phase1_figures",
    "select_representative_samples",
    "generate_all_phase1_reports",
]
