"""
Deterministic Random Seed Management for Phase 02 Training.
"""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> dict:
    """
    Sets deterministic seeds across Python standard library, NumPy, and PyTorch (if available).
    
    Returns a dictionary summarizing the seeded components.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    seeded_status = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": False,
        "cuda": False
    }
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            seeded_status["cuda"] = True
        seeded_status["torch"] = True
    except ImportError:
        pass
        
    return seeded_status
