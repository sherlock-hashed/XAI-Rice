# Reproducibility & Random Seed Protocol

**Project:** XAI-RiceGuard  
**Version:** 0.1.0-phase0  

---

## 1. Deterministic Seeding Policy

Scientific reproducibility requires that any experiment run with the same seed, codebase, and hardware configuration produces numerically reproducible results.

### 1.1 Global Seed Function
The standard seeding routine is defined as:

```python
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic_cudnn: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
```

### 1.2 Multi-Seed Protocol
While the primary development seed is `42`, final IEEE publication tables will evaluate statistical variance across a multi-seed suite: `[42, 123, 456, 789, 1024]`, reporting Mean $\pm$ Standard Deviation.

---

## 2. Experiment ID Specification

Every experimental run must generate a structured, machine-parsable identifier:

$$\text{Format: } \text{XR\_P}\{\text{Phase}\}\_\{\text{Task}\}\_\{\text{Model}\}\_\{\text{Seed}\}\_\{\text{YYYYMMDD\_HHMMSS}\}$$

Examples:
- `XR_P0_FOUNDATION_20260909_041500`
- `XR_P2_CLASSIFICATION_EFFICIENTNETV2_S42_20260915_120000`

---

## 3. Provenance & Metadata Tracing

Every run logs:
1. Exact Git commit hash (`git rev-parse HEAD`).
2. Environment snapshot (Python, PyTorch, CUDA, GPU model).
3. Exact SHA-256 hash of dataset manifests.
4. Full copy of configuration YAML.
