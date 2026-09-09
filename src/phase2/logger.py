"""
Training Logger and Live Status Formatter for Phase 02 Execution.
"""

import os
import time
import logging
from typing import Optional, Dict, Any


def format_duration(seconds: float) -> str:
    """Formats duration in seconds into HH:MM:SS string."""
    if seconds < 0 or seconds != seconds:  # NaN check
        return "00:00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ExperimentLogger:
    """
    Manages dual console and file logging with publication-grade formatting.
    """
    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "train.log")
        
        self.logger = logging.getLogger(f"phase2_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear previous handlers
        
        # File handler
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def info(self, msg: str):
        self.logger.info(msg)
        print(msg)
        
    def log_only(self, msg: str):
        self.logger.info(msg)
        
    def print_banner(self, title: str):
        border = "=" * 70
        self.info(f"\n{border}\n  {title}\n{border}")
        
    def print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        model_name: str,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_macro_f1: float,
        val_weighted_f1: float,
        lr: float,
        epoch_time_sec: float,
        elapsed_time_sec: float,
        eta_sec: float,
        best_macro_f1: float,
        best_epoch: int,
        is_best: bool,
        gpu_info: Dict[str, Any]
    ):
        border = "-" * 70
        progress_pct = (epoch / total_epochs) * 100
        
        vram_str = "CPU"
        if gpu_info.get("available"):
            vram_str = f"{gpu_info['allocated_gb']} GB / {gpu_info['total_gb']} GB ({gpu_info['device_name']})"
            
        best_marker = " [NEW BEST]" if is_best else ""
        
        summary = (
            f"\n{border}\n"
            f"Epoch [{epoch:02d}/{total_epochs:02d}] Progress: {progress_pct:.1f}%\n"
            f"Model: {model_name} | LR: {lr:.6f} | Device: {vram_str}\n"
            f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.2f}%\n"
            f"Val Loss   : {val_loss:.4f} | Val Acc   : {val_acc:.2f}%\n"
            f"Val MacroF1: {val_macro_f1:.4f}{best_marker} | Val WeightedF1: {val_weighted_f1:.4f}\n"
            f"Epoch Time : {format_duration(epoch_time_sec)} | Elapsed: {format_duration(elapsed_time_sec)} | Estimated Remaining ETA: {format_duration(eta_sec)}\n"
            f"Best MacroF1: {best_macro_f1:.4f} (Epoch {best_epoch:02d})\n"
            f"Checkpoints: last_checkpoint.pt{' & best_model.pt' if is_best else ''}\n"
            f"{border}"
        )
        self.info(summary)
