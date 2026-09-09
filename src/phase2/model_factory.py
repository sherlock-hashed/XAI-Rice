"""
PyTorch Model Factory for Instantiating Standardized Baseline Classifiers.
"""

from typing import Dict, Any, Tuple
from src.phase2.model_registry import is_model_registered, MODEL_REGISTRY


def count_parameters(model: Any) -> Dict[str, int]:
    """
    Computes total, trainable, and frozen parameter counts for a PyTorch module.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen
    }


def create_model(
    model_name: str,
    num_classes: int = 6,
    pretrained: bool = True
) -> Any:
    """
    Creates and configures a standardized PyTorch baseline model for 6-class disease classification.
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
    except ImportError:
        raise ImportError("PyTorch and torchvision are required to instantiate models.")
        
    model_name = model_name.lower()
    if not is_model_registered(model_name):
        raise ValueError(
            f"Unknown model architecture: '{model_name}'. "
            f"Registered architectures: {list(MODEL_REGISTRY.keys())}"
        )
        
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
    elif model_name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        
    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        raise NotImplementedError(f"Architecture '{model_name}' registered but constructor not implemented.")
        
    return model
