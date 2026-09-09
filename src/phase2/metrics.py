"""
Comprehensive Multi-Class Metric Evaluation Suite for Phase 02 Validation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from src.phase2.dataset import CANONICAL_CLASSES


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes all standard multi-class classification metrics for validation evaluation.
    Primary model selection metric: macro_f1.
    """
    if class_names is None:
        class_names = CANONICAL_CLASSES
        
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(range(len(class_names))), zero_division=0
    )
    
    per_class_dict = {}
    for i, name in enumerate(class_names):
        per_class_dict[name] = {
            "class_id": i,
            "precision": float(p_per_class[i]),
            "recall": float(r_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(support[i])
        }
        
    cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names)))).tolist()
    
    # Normalized confusion matrix (row-normalized)
    cm_arr = np.array(cm_raw, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0).tolist()
    
    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class_dict,
        "confusion_matrix": cm_raw,
        "confusion_matrix_normalized": cm_norm,
        "total_samples": len(y_true_arr)
    }
