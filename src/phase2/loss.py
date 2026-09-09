"""
Loss Function Factory for Phase 02 Training.
"""

from typing import Optional, Any


def get_loss_function(
    name: str = "cross_entropy",
    label_smoothing: float = 0.0,
    weight: Optional[Any] = None
) -> Any:
    """
    Returns the configured PyTorch loss function.
    Default: Standard CrossEntropyLoss with no label smoothing (label_smoothing=0.0).
    """
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for loss functions.")
        
    name = name.lower()
    if name in ["cross_entropy", "ce"]:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss function: '{name}'")
