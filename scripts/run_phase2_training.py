"""
Command Line Interface for Phase 02 Baseline Training, Model Comparison & Report Compilation.

Usage examples:
  python scripts/run_phase2_training.py --smoke-test
  python scripts/run_phase2_training.py --model efficientnet_b0 --epochs 30 --batch-size 32
  python scripts/run_phase2_training.py --all-baselines
  python scripts/run_phase2_training.py --model resnet50 --resume
"""

import os
import sys
import argparse

# Ensure workspace root is on Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.phase2.experiment import run_experiment
from src.phase2.phase2_report import generate_phase2_reports


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI-RiceGuard — Phase 02 Baseline Architecture & Model Training Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50", "convnext_tiny", "swin_t", "vit_b_16"],
        help="Model architecture to train (default: efficientnet_b0)"
    )
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Sequentially trains all 3 primary baseline models (EfficientNet-B0, ResNet-50, ConvNeXt-Tiny)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Executes a rapid 1-epoch smoke test verifying data loaders, models, forward/backward, metrics, and checkpoints"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumes training from the latest checkpoint (last_checkpoint.pt) in the experiment directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'cuda' or 'cpu' (default: auto-detected)"
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="manifests/phase1/primary_train.csv",
        help="Path to frozen training manifest"
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="manifests/phase1/primary_validation.csv",
        help="Path to frozen validation manifest"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional root directory of primary images on disk/Drive"
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="experiments/phase2",
        help="Base directory for experiment artifacts"
    )
    return parser.parse_args()


def build_config(model_name: str, args) -> dict:
    epochs = 1 if args.smoke_test else args.epochs
    batch_size = 4 if args.smoke_test else args.batch_size
    
    return {
        "experiment": {
            "name": f"{model_name}_seed{args.seed}",
            "phase": "phase2",
        },
        "seed": args.seed,
        "data": {
            "train_manifest": args.train_manifest,
            "validation_manifest": args.val_manifest,
            "data_dir": args.data_dir,
            "image_size": 224,
            "num_classes": 6
        },

        "model": {
            "name": model_name,
            "pretrained": True
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "amp": True,
            "gradient_clip_norm": 1.0
        },
        "loss": {
            "name": "cross_entropy",
            "label_smoothing": 0.0
        },
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 0
        },
        "early_stopping": {
            "enabled": not args.smoke_test,
            "monitor": "val_macro_f1",
            "patience": 5
        },
        "output_dir": os.path.join(args.output_base_dir, model_name)
    }


def main():
    args = parse_args()
    print("=" * 70)
    print("  XAI-RiceGuard — Phase 02 Baseline Architecture Training Pipeline")
    print("=" * 70)
    
    if args.all_baselines:
        baseline_models = ["efficientnet_b0", "resnet50", "convnext_tiny"]
        print(f"\n[MULTI-MODEL PIPELINE] Sequentially running all baselines: {baseline_models}")
        
        for idx, m in enumerate(baseline_models, 1):
            exp_dir = os.path.join(args.output_base_dir, m)
            comp_file = os.path.join(exp_dir, "completion.json")
            
            if os.path.exists(comp_file) and not args.smoke_test:
                print(f"\n[{idx}/{len(baseline_models)}] Model '{m}' is already COMPLETE. Skipping to next.")
                continue
                
            print(f"\n[{idx}/{len(baseline_models)}] Starting Training for: {m}")
            cfg = build_config(m, args)
            run_experiment(cfg, resume=args.resume, device_str=args.device)
            
        print("\nAll baseline models completed! Compiling aggregate reports...")
        generate_phase2_reports(
            experiments_dir=args.output_base_dir,
            results_dir="results/phase2",
            reports_dir="reports/phase2",
            figures_dir="figures/phase2"
        )
        print("Reports and figures generated successfully.")
        
    else:
        cfg = build_config(args.model, args)
        run_experiment(cfg, resume=args.resume, device_str=args.device)
        
        if not args.smoke_test:
            generate_phase2_reports(
                experiments_dir=args.output_base_dir,
                results_dir="results/phase2",
                reports_dir="reports/phase2",
                figures_dir="figures/phase2"
            )


if __name__ == "__main__":
    main()
