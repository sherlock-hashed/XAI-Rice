"""
Experiment Orchestrator for Phase 02 Baseline Training Runs.
"""

import os
import sys
import json
import yaml
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.phase2.seed import seed_everything
from src.phase2.dataloader import create_dataloaders
from src.phase2.model_factory import create_model, count_parameters
from src.phase2.loss import get_loss_function
from src.phase2.optimizer import get_optimizer
from src.phase2.scheduler import get_scheduler
from src.phase2.trainer import Trainer
from src.phase2.logger import ExperimentLogger
from src.phase2.dataset import verify_manifest_firewall, CANONICAL_CLASSES


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_experiment(
    config: Dict[str, Any],
    resume: bool = False,
    device_str: Optional[str] = None,
    base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single model baseline experiment under Phase 02 experimental protocol.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to run baseline experiments.")
        
    model_name = config.get("model", {}).get("name", "efficientnet_b0")
    seed = config.get("seed", 42)
    
    # 1. Deterministic Seeding
    seed_status = seed_everything(seed)
    
    # 2. Output and Checkpoint Directories
    exp_dir = config.get("output_dir", f"experiments/phase2/{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = ExperimentLogger(log_dir=exp_dir, experiment_name=model_name)
    logger.print_banner(f"XAI-RiceGuard — Phase 02 Experiment: {model_name}")
    
    # 3. Verify Manifest Firewalls
    train_manifest = config.get("data", {}).get("train_manifest", "manifests/phase1/primary_train.csv")
    val_manifest = config.get("data", {}).get("validation_manifest", "manifests/phase1/primary_validation.csv")
    
    verify_manifest_firewall(train_manifest, expected_role="train")
    verify_manifest_firewall(val_manifest, expected_role="validation")
    logger.info("Data Firewall Check: PASS (Train & Validation isolated; Calibration & Internal Test LOCKED).")
    
    # 4. Detect Device
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
        
    logger.info(f"Target Compute Device: {device}")
    
    # 5. Create DataLoaders
    image_size = config.get("data", {}).get("image_size", 224)
    batch_size = config.get("training", {}).get("batch_size", 32)
    aug_cfg = config.get("augmentation", {"enabled": True})
    data_dir = base_dir or config.get("data", {}).get("data_dir", None)
    max_train_samples = config.get("data", {}).get("max_train_samples", None)
    max_val_samples = config.get("data", {}).get("max_val_samples", None)
    
    train_loader, val_loader = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        image_size=image_size,
        batch_size=batch_size,
        augmentation_cfg=aug_cfg,
        base_dir=data_dir,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples
    )

    logger.info(f"Loaded DataLoaders: {len(train_loader.dataset)} Train images | {len(val_loader.dataset)} Validation images")

    
    # 6. Instantiate Model
    num_classes = config.get("data", {}).get("num_classes", 6)
    pretrained = config.get("model", {}).get("pretrained", True)
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    
    param_counts = count_parameters(model)
    logger.info(
        f"Model Architecture: {model_name} (Pretrained: {pretrained})\n"
        f"Parameters: Total={param_counts['total_params']:,} | Trainable={param_counts['trainable_params']:,}"
    )
    
    # 7. Loss, Optimizer, Scheduler
    loss_cfg = config.get("loss", {})
    criterion = get_loss_function(
        name=loss_cfg.get("name", "cross_entropy"),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0)
    )
    
    opt_cfg = config.get("optimizer", {})
    optimizer = get_optimizer(
        model=model,
        name=opt_cfg.get("name", "AdamW"),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4)
    )
    
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    sched_cfg = config.get("scheduler", {})
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=sched_cfg.get("name", "cosine"),
        epochs=epochs,
        warmup_epochs=sched_cfg.get("warmup_epochs", 0)
    )
    
    # 8. Setup Trainer
    git_commit = get_git_commit_hash()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=exp_dir,
        logger=logger,
        device=device,
        git_commit=git_commit
    )
    
    if resume:
        last_ckpt = os.path.join(exp_dir, "checkpoint", "last_checkpoint.pt")
        if os.path.exists(last_ckpt):
            trainer.resume_from_checkpoint(last_ckpt)
        else:
            logger.info(f"Resume requested but no checkpoint found at {last_ckpt}. Starting from Epoch 1.")
            
    # 9. Fit Model
    train_results = trainer.fit()
    
    # 10. Record Run Metadata
    metadata = {
        "experiment_id": f"{model_name}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "seed": seed,
        "model": model_name,
        "pretrained": pretrained,
        "image_size": image_size,
        "batch_size": batch_size,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "calibration_samples": 1799,
        "calibration_used": False,
        "internal_test_samples": 1798,
        "internal_test_used": False,
        "sethy_used": False,
        "bd5_used": False,
        "riceseg_used": False,
        "class_mapping": CANONICAL_CLASSES,
        "parameter_counts": param_counts,
        "best_epoch": train_results["best_epoch"],
        "best_val_macro_f1": train_results["best_val_macro_f1"],
        "best_val_accuracy": train_results["best_val_accuracy"],
        "total_training_time_seconds": train_results["total_training_time_seconds"]
    }
    
    meta_path = os.path.join(exp_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # Save config.yaml
    cfg_path = os.path.join(exp_dir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # 11. Write Completion Marker
    completion_data = {
        "status": "SUCCESS",
        "model_name": model_name,
        "best_epoch": train_results["best_epoch"],
        "best_val_macro_f1": train_results["best_val_macro_f1"],
        "best_val_accuracy": train_results["best_val_accuracy"],
        "total_training_time_seconds": train_results["total_training_time_seconds"],
        "checkpoint_path": os.path.join(exp_dir, "checkpoint", "best_model.pt")
    }
    with open(os.path.join(exp_dir, "completion.json"), "w", encoding="utf-8") as f:
        json.dump(completion_data, f, indent=2)
        
    logger.info(f"\n[COMPLETE] Baseline experiment '{model_name}' finished successfully!")
    return completion_data
