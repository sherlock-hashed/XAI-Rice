"""
Optimizer Factory for Phase 02 Training.
"""

from typing import Any, Tuple, Optional


def get_optimizer(
    model: Any,
    name: str = "AdamW",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9
) -> Any:
    """
    Returns configured PyTorch optimizer. Default: AdamW (lr=0.0001, weight_decay=0.0001).
    """
    try:
        import torch.optim as optim
    except ImportError:
        raise ImportError("PyTorch is required for optimizers.")
        
    name = name.lower()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if name == "adamw":
        return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == "adam":
        return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == "sgd":
        return optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: '{name}'")
