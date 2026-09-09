"""
Data Transforms and Augmentation Pipelines for Phase 02 Training and Validation.
"""

from typing import Dict, Any, Optional

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, augmentation_cfg: Optional[Dict[str, Any]] = None):
    """
    Creates standard training data transform pipeline with moderate augmentation.
    """
    try:
        from torchvision import transforms
    except ImportError:
        return None
        
    cfg = augmentation_cfg or {}
    enabled = cfg.get("enabled", True)
    
    if not enabled:
        return transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
    hflip = cfg.get("horizontal_flip", True)
    vflip = cfg.get("vertical_flip", False)
    rotation_deg = cfg.get("rotation", 10)
    jitter = cfg.get("color_jitter", {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.03})
    
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    ]
    
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if vflip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    if rotation_deg > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_deg))
    if jitter:
        transform_list.append(transforms.ColorJitter(
            brightness=jitter.get("brightness", 0.15),
            contrast=jitter.get("contrast", 0.15),
            saturation=jitter.get("saturation", 0.15),
            hue=jitter.get("hue", 0.03)
        ))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 224):
    """
    Creates deterministic validation transform pipeline.
    """
    try:
        from torchvision import transforms
    except ImportError:
        return None
        
    resize_dim = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
