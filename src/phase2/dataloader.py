"""
PyTorch DataLoader Factory with Environment-Adaptive Worker and Memory Settings.
"""

import os
from typing import Dict, Any, Tuple, Optional
from src.phase2.dataset import RiceLeafDataset
from src.phase2.transforms import get_train_transforms, get_val_transforms


def seed_worker(worker_id):
    import random
    import numpy as np
    worker_seed = (int(os.environ.get("PYTHONHASHSEED", 42)) + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    augmentation_cfg: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[Any, Any]:
    """
    Builds training and validation DataLoaders with deterministic seeding and environment adaptation.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        raise ImportError("PyTorch is required to build DataLoaders.")
        
    train_transform = get_train_transforms(image_size=image_size, augmentation_cfg=augmentation_cfg)
    val_transform = get_val_transforms(image_size=image_size)
    
    train_dataset = RiceLeafDataset(
        manifest_path=train_manifest,
        role="train",
        transform=train_transform,
        base_dir=base_dir
    )
    
    val_dataset = RiceLeafDataset(
        manifest_path=val_manifest,
        role="validation",
        transform=val_transform,
        base_dir=base_dir
    )
    
    if max_train_samples and max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, range(max_train_samples))
    if max_val_samples and max_val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, range(max_val_samples))
    
    # Environment adaptations
    is_cuda = torch.cuda.is_available()
    if pin_memory is None:
        pin_memory = is_cuda
    if num_workers is None:
        num_workers = 2 if is_cuda else 0
        
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    persistent_workers = (num_workers > 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
        worker_init_fn=seed_worker,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
        worker_init_fn=seed_worker
    )
    
    return train_loader, val_loader

