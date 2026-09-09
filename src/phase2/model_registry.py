"""
Standardized Model Registry for Phase 02 Baseline Architectures.
"""

from typing import Dict, Any

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "efficientnet_b0": {
        "family": "CNN (Compound Scaled)",
        "default_image_size": 224,
        "description": "EfficientNet-B0 (Tan & Le, 2019) with inverted residual MBConv blocks",
        "supports_pretrained": True,
        "is_primary_baseline": True
    },
    "resnet50": {
        "family": "CNN (Residual)",
        "default_image_size": 224,
        "description": "ResNet-50 (He et al., 2016) 50-layer deep residual network",
        "supports_pretrained": True,
        "is_primary_baseline": True
    },
    "convnext_tiny": {
        "family": "Modernized CNN",
        "default_image_size": 224,
        "description": "ConvNeXt-Tiny (Liu et al., 2022) modernized ConvNet architecture",
        "supports_pretrained": True,
        "is_primary_baseline": True
    },
    "swin_t": {
        "family": "Vision Transformer (Hierarchical)",
        "default_image_size": 224,
        "description": "Swin Transformer Tiny (Liu et al., 2021) with shifted windows",
        "supports_pretrained": True,
        "is_primary_baseline": False
    },
    "vit_b_16": {
        "family": "Vision Transformer",
        "default_image_size": 224,
        "description": "Vision Transformer Base (Dosovitskiy et al., 2020) with 16x16 patches",
        "supports_pretrained": True,
        "is_primary_baseline": False
    }
}


def get_registered_models() -> Dict[str, Dict[str, Any]]:
    return MODEL_REGISTRY


def is_model_registered(model_name: str) -> bool:
    return model_name.lower() in MODEL_REGISTRY
