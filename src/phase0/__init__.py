"""Phase 0: Project Foundation, Environment Verification, Dataset Audit, and Research Protocol."""

from src.phase0.environment import detect_environment, generate_environment_report
from src.phase0.config import load_paths_config, load_project_config, resolve_paths
from src.phase0.dataset_discovery import discover_dataset_structure
from src.phase0.image_audit import audit_images
from src.phase0.annotation_audit import audit_yolo_annotations, audit_dataset_metadata
from src.phase0.duplicate_audit import audit_duplicates, audit_cross_dataset_leakage_risk
from src.phase0.riceseg_audit import audit_sethy_riceseg_pairing
from src.phase0.manifest_generator import generate_dataset_manifest
from src.phase0.report_generator import generate_phase0_audit_report, generate_phase0_completion_report

__all__ = [
    "detect_environment",
    "generate_environment_report",
    "load_paths_config",
    "load_project_config",
    "resolve_paths",
    "discover_dataset_structure",
    "audit_images",
    "audit_yolo_annotations",
    "audit_dataset_metadata",
    "audit_duplicates",
    "audit_cross_dataset_leakage_risk",
    "audit_sethy_riceseg_pairing",
    "generate_dataset_manifest",
    "generate_phase0_audit_report",
    "generate_phase0_completion_report",
]
