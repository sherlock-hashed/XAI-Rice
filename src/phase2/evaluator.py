"""
Validation Evaluator and Diagnostic Error Analysis for Phase 02.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from src.phase2.metrics import compute_metrics
from src.phase2.dataset import CANONICAL_CLASSES, IDX_TO_CLASS


def evaluate_model(
    model: Any,
    val_loader: Any,
    criterion: Any,
    device: str = "cpu"
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Evaluates model on validation loader, returning average loss, metrics dict, and predictions DataFrame.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required for validation evaluation.")
        
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    pred_rows = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            sample_ids = batch["sample_id"]
            img_paths = batch["image_path"]
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            targets_np = targets.cpu().numpy()
            
            for i in range(bs):
                pred_cls = preds[i]
                true_cls = targets_np[i]
                conf = float(probs[i, pred_cls])
                
                row = {
                    "sample_id": sample_ids[i],
                    "true_class": IDX_TO_CLASS[true_cls],
                    "true_class_id": int(true_cls),
                    "predicted_class": IDX_TO_CLASS[pred_cls],
                    "predicted_class_id": int(pred_cls),
                    "confidence": conf,
                    "correct": bool(pred_cls == true_cls),
                    "image_path": img_paths[i]
                }
                # Add per-class probabilities
                for c_idx, c_name in enumerate(CANONICAL_CLASSES):
                    row[f"prob_{c_name}"] = float(probs[i, c_idx])
                    
                pred_rows.append(row)
                all_preds.append(int(pred_cls))
                all_targets.append(int(true_cls))
                
    avg_loss = total_loss / max(1, total_samples)
    metrics = compute_metrics(all_targets, all_preds, class_names=CANONICAL_CLASSES)
    metrics["val_loss"] = float(avg_loss)
    
    preds_df = pd.DataFrame(pred_rows)
    return avg_loss, metrics, preds_df


def save_validation_diagnostics(
    output_dir: str,
    metrics: Dict[str, Any],
    preds_df: pd.DataFrame,
    model_name: str
) -> None:
    """
    Saves validation predictions, error logs, confusion matrix CSV, and figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Validation predictions
    preds_path = os.path.join(output_dir, "validation_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    
    # 2. Validation errors
    errors_df = preds_df[~preds_df["correct"]].copy()
    errors_path = os.path.join(output_dir, "validation_errors.csv")
    errors_df.to_csv(errors_path, index=False)
    
    # 3. Metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # 4. Confusion Matrix CSV
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"True_{c}" for c in CANONICAL_CLASSES],
        columns=[f"Pred_{c}" for c in CANONICAL_CLASSES]
    )
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    
    # 5. Plot confusion matrix
    plt.figure(figsize=(8, 6), dpi=300)
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    
    im = plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(f"Validation Confusion Matrix — {model_name}\n(Row-Normalized Recall)", fontsize=13, pad=15)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(CANONICAL_CLASSES))
    plt.xticks(tick_marks, CANONICAL_CLASSES, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, CANONICAL_CLASSES, fontsize=9)
    
    thresh = cm_norm.max() / 2.
    for i in range(len(CANONICAL_CLASSES)):
        for j in range(len(CANONICAL_CLASSES)):
            raw_val = metrics["confusion_matrix"][i][j]
            norm_val = cm_norm[i, j]
            txt = f"{norm_val:.1%}\n({raw_val})"
            plt.text(
                j, i, txt,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if norm_val > thresh else "black",
                fontsize=8
            )
            
    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"))
    plt.close()
