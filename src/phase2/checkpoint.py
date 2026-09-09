"""
Atomic Checkpoint Management and Resume Facility for Phase 02 Training.
"""

import os
import random
import shutil
import tempfile
import numpy as np
from typing import Dict, Any, Optional, Tuple


def get_rng_states() -> Dict[str, Any]:
    """Captures random state across Python, NumPy, and PyTorch/CUDA."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state()
    }
    try:
        import torch
        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restores random state across Python, NumPy, and PyTorch/CUDA."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    try:
        import torch
        if "torch" in states and states["torch"] is not None:
            torch.set_rng_state(states["torch"])
        if "cuda" in states and states["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])
    except ImportError:
        pass


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    best_val_macro_f1: float,
    best_val_accuracy: float,
    best_epoch: int,
    config: Dict[str, Any],
    git_commit: str,
    training_history: list,
    is_best: bool = False
) -> str:
    """
    Atomically saves training checkpoint and optionally updates best_model.pt.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for checkpointing.")
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "config": config,
        "git_commit": git_commit,
        "rng_states": get_rng_states(),
        "training_history": training_history
    }
    
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    
    # Atomic write to temporary file first
    with tempfile.NamedTemporaryFile(delete=False, dir=checkpoint_dir, suffix=".tmp") as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        torch.save(state, tmp_path)
        shutil.move(tmp_path, last_ckpt_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
        
    if is_best:
        best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        shutil.copy2(last_ckpt_path, best_ckpt_path)
        
    return last_ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: Any,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads checkpoint and restores model, optimizer, scheduler, scaler, and PRNG states.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for loading checkpoints.")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    if "rng_states" in checkpoint and checkpoint["rng_states"]:
        set_rng_states(checkpoint["rng_states"])
        
    return checkpoint
