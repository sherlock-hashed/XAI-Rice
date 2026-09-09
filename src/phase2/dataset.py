"""
PyTorch Dataset and Strict Data Firewall Enforcement for Phase 02.
"""

import os
import pandas as pd
from PIL import Image
from typing import Optional, Callable, Dict, Any, List, Tuple

# Canonical 6-class mapping strictly consistent with Phase 0 and Phase 01 manifests
CANONICAL_CLASSES: List[str] = [
    "Healthy",
    "Blast",
    "Brown spot",
    "Leaf smut",
    "Rice Tungro",
    "Sheath blight"
]

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(CANONICAL_CLASSES)}

FORBIDDEN_PHASE2_MANIFESTS = {
    "primary_calibration.csv": "FROZEN CALIBRATION SET (Reserved for Phase 08)",
    "primary_internal_test.csv": "LOCKED INTERNAL TEST SET (Reserved for Final Evaluation)",
    "sethy_external.csv": "EXTERNAL BENCHMARK DATASET",
    "bd5_external.csv": "EXTERNAL FIELD BENCHMARK DATASET",
    "riceseg_ground_truth.csv": "XAI GROUND TRUTH MASKS"
}


class DataFirewallViolationError(Exception):
    """Raised when a forbidden dataset or split is accessed during Phase 02."""
    pass


def verify_manifest_firewall(manifest_path: str, expected_role: str) -> None:
    """
    Validates that a given manifest path obeys strict Phase 02 data firewall rules.
    """
    base_name = os.path.basename(manifest_path)
    
    if base_name in FORBIDDEN_PHASE2_MANIFESTS:
        raise DataFirewallViolationError(
            f"PHASE 02 DATA FIREWALL VIOLATION!\n"
            f"Manifest: {base_name}\n"
            f"Role: {FORBIDDEN_PHASE2_MANIFESTS[base_name]}\n"
            f"Access Status: STRICTLY FORBIDDEN IN PHASE 02."
        )
        
    expected_role = expected_role.lower()
    if expected_role == "train" and base_name != "primary_train.csv":
        raise DataFirewallViolationError(
            f"Expected 'primary_train.csv' for training, but received: {base_name}"
        )
    elif expected_role == "validation" and base_name != "primary_validation.csv":
        raise DataFirewallViolationError(
            f"Expected 'primary_validation.csv' for validation, but received: {base_name}"
        )


try:
    from torch.utils.data import Dataset
    _BaseDataset = Dataset
except ImportError:
    class _BaseDataset:
        pass


class RiceLeafDataset(_BaseDataset):
    """
    Standardized PyTorch Dataset for Rice Leaf Disease Classification.
    """
    def __init__(
        self,
        manifest_path: str,
        role: str,
        transform: Optional[Callable] = None,
        base_dir: Optional[str] = None
    ):
        verify_manifest_firewall(manifest_path, role)
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            
        self.manifest_path = manifest_path
        self.role = role
        self.transform = transform
        self.base_dir = base_dir
        
        self.df = pd.read_csv(manifest_path)
        
        # Verify required columns (accept class_name or disease_class, relative_path or image_path)
        if "disease_class" in self.df.columns:
            self.class_col = "disease_class"
        elif "class_name" in self.df.columns:
            self.class_col = "class_name"
        else:
            raise ValueError(f"Manifest {manifest_path} missing class column ('class_name' or 'disease_class')")
            
        if "image_path" in self.df.columns:
            self.path_col = "image_path"
        elif "relative_path" in self.df.columns:
            self.path_col = "relative_path"
        else:
            raise ValueError(f"Manifest {manifest_path} missing image path column ('image_path' or 'relative_path')")
            
        # Validate all classes are recognized
        for cls_name in self.df[self.class_col].unique():
            if cls_name not in CLASS_TO_IDX:
                raise ValueError(f"Unknown class '{cls_name}' found in manifest {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
            
        candidate_roots = []
        if self.base_dir:
            candidate_roots.append(self.base_dir)
            
        # Try resolving from centralized config
        try:
            from src.phase0.config import resolve_paths
            paths = resolve_paths()
            if "dataset_roots" in paths and "primary_dataset" in paths["dataset_roots"]:
                candidate_roots.append(paths["dataset_roots"]["primary_dataset"])
        except Exception:
            pass
            
        # Common Google Drive and local project dataset roots
        candidate_roots.extend([
            "/content/drive/MyDrive/BTech Final Year Project/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/RiceLeafDiseaseBD",
            "/content/drive/MyDrive/BTech Final Year Project/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/RiceLeafDiseaseBD",
            "/content/drive/MyDrive/BTech Final Year Project/XAI-RiceGuard/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/RiceLeafDiseaseBD",
            "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/RiceLeafDiseaseBD",
            "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/RiceLeafDiseaseBD",
            "RiceLeafDiseaseBD",
            os.path.join(os.getcwd(), "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/RiceLeafDiseaseBD")
        ])
        
        for root in candidate_roots:
            if root and os.path.exists(root):
                cand = os.path.join(root, path)
                if os.path.exists(cand):
                    return cand
                    
        return path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row[self.path_col])
        label_str = row[self.class_col]
        label_idx = CLASS_TO_IDX[label_str]
        sample_id = row.get("sample_id", f"{self.role}_{idx}")


        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"ERROR loading sample ID '{sample_id}' at path '{img_path}': {type(e).__name__}: {e}"
            ) from e
            
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": label_idx,
            "label_name": label_str,
            "sample_id": sample_id,
            "image_path": img_path
        }
