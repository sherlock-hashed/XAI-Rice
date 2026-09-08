# Data Leakage Prevention Protocol

**Project:** XAI-RiceGuard  
**Version:** 0.1.0-phase0  
**Status:** Mandatory Engineering & Research Guidelines  

---

## 1. Fundamental Principles

Data leakage compromises empirical validity and is a frequent cause of unreproducible, artificially inflated metrics in computer vision literature. The **XAI-RiceGuard** project enforces the following rules:

### Rule 1: Partitioning Before Any Image Transformation
- Data splitting (Train / Validation / Internal Test) must occur strictly **before** any data augmentation, normalizations, or resizing operations.
- Augmented variants of any image must reside **only** in the training partition.

### Rule 2: Tracking Source Image IDs & Duplicate Isolation
- Exact duplicate images (identified via SHA-256 hashes during Phase 0) and identical source leaf captures must be tracked with unique identifiers (`source_id`).
- If duplicates or near-duplicates exist, they must never cross split boundaries.

### Rule 3: External Dataset Isolation
- External benchmark datasets (`Sethy` and `RiceLeafDisease-BD5`) must remain completely untouched during:
  - Feature normalization calculation (mean/std must be computed solely on the primary training split).
  - Hyperparameter tuning and model selection.
  - Early stopping decisions.

### Rule 4: Zero Ground-Truth Contamination in XAI
- Ground truth segmentation masks (`RiceSeg-5932`) must never be provided as supervisory signals during standard classification training unless an explicit lesion-grounded multi-task loss is being tested under controlled ablation.

---

## 2. Leakage Verification Checklist

- [ ] Split manifests created on raw source filenames before augmentation.
- [ ] SHA-256 collision check between Train, Val, and Test sets.
- [ ] Zero overlap between Primary Dataset and External Test Datasets.
- [ ] Preprocessing statistics (mean, std) calculated strictly on `train_manifest.csv`.
