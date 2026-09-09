"""
End-to-End Fault-Tolerant PyTorch Training Engine with Mixed Precision and Live Progress.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

from src.phase2.evaluator import evaluate_model, save_validation_diagnostics
from src.phase2.checkpoint import save_checkpoint, load_checkpoint
from src.phase2.logger import ExperimentLogger, format_duration
from src.phase2.profiler import get_gpu_memory_info


class Trainer:
    """
    Manages the training loop, validation passes, checkpointing, AMP, and early stopping.
    """
    def __init__(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        criterion: Any,
        optimizer: Any,
        scheduler: Optional[Any],
        config: Dict[str, Any],
        output_dir: str,
        logger: ExperimentLogger,
        device: str = "cpu",
        git_commit: str = "unknown"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.device = device
        self.git_commit = git_commit
        
        self.checkpoint_dir = os.path.join(output_dir, "checkpoint")
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Mixed Precision (AMP)
        try:
            import torch
            self.use_amp = (
                device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
            ) and config.get("training", {}).get("amp", True)
            self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        except Exception:
            self.use_amp = False
            self.scaler = None
            
        # Training state
        self.epochs = config.get("training", {}).get("epochs", 30)
        self.start_epoch = 1
        self.best_val_macro_f1 = -1.0
        self.best_val_accuracy = -1.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopping_cfg = config.get("early_stopping", {"enabled": True, "patience": 5})
        
        self.history = []
        self.epoch_durations = []
        self.total_elapsed_time = 0.0

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Restores training state from an existing checkpoint."""
        self.logger.info(f"Resuming training state from checkpoint: {checkpoint_path}")
        ckpt = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        self.best_val_accuracy = ckpt.get("best_val_accuracy", -1.0)
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.history = ckpt.get("training_history", [])
        self.logger.info(
            f"Resumed at Epoch {self.start_epoch} | Current Best Val Macro-F1: {self.best_val_macro_f1:.4f} (Epoch {self.best_epoch})"
        )

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs a single training epoch across all training batches."""
        try:
            import torch
            from tqdm import tqdm
        except ImportError:
            raise ImportError("PyTorch and tqdm are required for training.")
            
        self.model.train()
        self.model.to(self.device)
        
        total_loss = 0.0
        correct_samples = 0
        total_samples = 0
        
        grad_clip = self.config.get("training", {}).get("gradient_clip_norm", None)
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Train]",
            leave=False,
            dynamic_ncols=True
        )
        
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            bs = images.size(0)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
            loss_val = loss.item()
            total_loss += loss_val * bs
            total_samples += bs
            
            preds = outputs.argmax(dim=1)
            correct_samples += (preds == targets).sum().item()
            
            curr_acc = (correct_samples / max(1, total_samples)) * 100.0
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "acc": f"{curr_acc:.1f}%"
            })
            
        avg_loss = total_loss / max(1, total_samples)
        avg_acc = (correct_samples / max(1, total_samples)) * 100.0
        return avg_loss, avg_acc

    def fit(self) -> Dict[str, Any]:
        """
        Executes the full training loop with checkpointing, validation evaluation, and history logging.
        """
        model_name = self.config.get("model", {}).get("name", "model")
        self.logger.print_banner(f"Phase 02 Training: {model_name} (Target Epochs: {self.epochs})")
        
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                epoch_start_time = time.time()
                
                # 1. Train epoch
                train_loss, train_acc = self.train_one_epoch(epoch)
                
                # 2. Validation epoch
                val_loss, val_metrics, val_preds_df = evaluate_model(
                    model=self.model,
                    val_loader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device
                )
                
                # 3. Step scheduler
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                # 4. Timing & ETA calculation
                epoch_time = time.time() - epoch_start_time
                self.epoch_durations.append(epoch_time)
                self.total_elapsed_time += epoch_time
                
                avg_epoch_time = float(np.mean(self.epoch_durations))
                remaining_epochs = max(0, self.epochs - epoch)
                eta_seconds = avg_epoch_time * remaining_epochs
                
                # 5. Check if best model (Validation Macro-F1 primary, Validation Accuracy secondary)
                val_macro_f1 = val_metrics["macro_f1"]
                val_acc = val_metrics["accuracy"] * 100.0
                
                is_best = False
                if val_macro_f1 > self.best_val_macro_f1:
                    is_best = True
                elif abs(val_macro_f1 - self.best_val_macro_f1) < 1e-6 and val_acc > self.best_val_accuracy:
                    is_best = True
                    
                if is_best:
                    self.best_val_macro_f1 = val_macro_f1
                    self.best_val_accuracy = val_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    # Save diagnostics for the best epoch
                    save_validation_diagnostics(
                        output_dir=self.output_dir,
                        metrics=val_metrics,
                        preds_df=val_preds_df,
                        model_name=model_name
                    )
                else:
                    self.patience_counter += 1
                    
                # 6. Log history entry
                history_entry = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "train_accuracy": round(train_acc, 2),
                    "val_loss": round(val_loss, 5),
                    "val_accuracy": round(val_acc, 2),
                    "val_macro_precision": round(val_metrics["macro_precision"], 4),
                    "val_macro_recall": round(val_metrics["macro_recall"], 4),
                    "val_macro_f1": round(val_macro_f1, 4),
                    "val_weighted_f1": round(val_metrics["weighted_f1"], 4),
                    "learning_rate": float(curr_lr),
                    "epoch_time_seconds": round(epoch_time, 1),
                    "elapsed_time_seconds": round(self.total_elapsed_time, 1),
                    "eta_seconds": round(eta_seconds, 1),
                    "is_best": is_best
                }
                self.history.append(history_entry)
                
                # Write history.csv after every epoch
                pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "history.csv"), index=False)
                
                # 7. Atomic checkpointing
                gpu_info = get_gpu_memory_info()
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    best_val_macro_f1=self.best_val_macro_f1,
                    best_val_accuracy=self.best_val_accuracy,
                    best_epoch=self.best_epoch,
                    config=self.config,
                    git_commit=self.git_commit,
                    training_history=self.history,
                    is_best=is_best
                )
                
                # 8. Print live formatted summary
                self.logger.print_epoch_summary(
                    epoch=epoch,
                    total_epochs=self.epochs,
                    model_name=model_name,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    val_macro_f1=val_macro_f1,
                    val_weighted_f1=val_metrics["weighted_f1"],
                    lr=curr_lr,
                    epoch_time_sec=epoch_time,
                    elapsed_time_sec=self.total_elapsed_time,
                    eta_sec=eta_seconds,
                    best_macro_f1=self.best_val_macro_f1,
                    best_epoch=self.best_epoch,
                    is_best=is_best,
                    gpu_info=gpu_info
                )
                
                # 9. Early stopping check
                if self.early_stopping_cfg.get("enabled", True):
                    patience = self.early_stopping_cfg.get("patience", 5)
                    if self.patience_counter >= patience:
                        self.logger.info(
                            f"\n[EARLY STOPPING] Validation Macro-F1 did not improve for {patience} consecutive epochs. "
                            f"Terminating training at epoch {epoch}. Best epoch was {self.best_epoch}."
                        )
                        break
                        
        except KeyboardInterrupt:
            self.logger.info("\n[INTERRUPTED] Training manually stopped by user.")
            self.logger.info(f"Latest state saved to {os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')}.")
            self.logger.info("Resume with: python scripts/run_phase2_training.py --resume")
            
        # Generate and save final training curves
        self.plot_training_curves()
        
        return {
            "model_name": model_name,
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_val_accuracy": self.best_val_accuracy,
            "total_epochs_trained": len(self.history),
            "total_training_time_seconds": round(self.total_elapsed_time, 2)
        }

    def plot_training_curves(self) -> None:
        """Generates publication-quality loss, accuracy, and Macro-F1 curves."""
        if not self.history:
            return
            
        df = pd.DataFrame(self.history)
        epochs = df["epoch"]
        
        # 1. Loss Curve
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(epochs, df["val_loss"], label="Val Loss", color="#d62728", lw=2)
        plt.axvline(self.best_epoch, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Best Epoch ({self.best_epoch})")
        plt.title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "loss_curve.png"))
        plt.close()
        
        # 2. Accuracy Curve
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(epochs, df["train_accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
        plt.plot(epochs, df["val_accuracy"], label="Val Accuracy", color="#2ca02c", lw=2)
        plt.axvline(self.best_epoch, color="#d62728", linestyle="--", alpha=0.7, label=f"Best Epoch ({self.best_epoch})")
        plt.title("Classification Accuracy vs. Epochs (%)", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "accuracy_curve.png"))
        plt.close()
        
        # 3. Macro-F1 Curve
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(epochs, df["val_macro_f1"], label="Val Macro-F1", color="#ff7f0e", lw=2)
        plt.plot(epochs, df["val_weighted_f1"], label="Val Weighted-F1", color="#9467bd", lw=1.5, linestyle="--")
        plt.axvline(self.best_epoch, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Best Macro-F1: {self.best_val_macro_f1:.4f}")
        plt.title("Validation Macro-F1 Score vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "macro_f1_curve.png"))
        plt.close()
