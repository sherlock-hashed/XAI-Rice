"""
Learning Rate Scheduler Factory for Phase 02 Training.
"""

from typing import Any, Optional


def get_scheduler(
    optimizer: Any,
    name: str = "cosine",
    epochs: int = 30,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6
) -> Optional[Any]:
    """
    Returns learning rate scheduler. Supports CosineAnnealingLR and warmup+cosine.
    """
    try:
        import torch.optim.lr_scheduler as lr_scheduler
    except ImportError:
        raise ImportError("PyTorch is required for learning rate schedulers.")
        
    name = name.lower()
    if name == "none" or epochs <= 0:
        return None
        
    if warmup_epochs > 0:
        main_epochs = max(1, epochs - warmup_epochs)
        cosine_sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=min_lr)
        warmup_sched = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        return lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )
    else:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
