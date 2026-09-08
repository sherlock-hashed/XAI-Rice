# XAI-RiceGuard: A Lesion-Grounded Explainable and Uncertainty-Aware Deep Learning Framework for Robust Rice Leaf Blast and Brown Spot Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![IEEE Protocol](https://img.shields.io/badge/Research-IEEE%20Standard-success.svg)](#)
[![Phase](https://img.shields.io/badge/Current%20Phase-Phase%200%20(Foundation)-blueviolet.svg)](#)

---

## 1. Project Overview

**XAI-RiceGuard** is an explainable and uncertainty-aware deep learning framework designed to improve the trustworthiness, interpretability, and robustness of automated rice leaf disease detection (specifically focusing on Blast and Brown Spot) under domain shifts and perturbations.

---

## 2. Research Separation & Immutable Dataset Roles

To preserve IEEE scientific validity and prevent data leakage, all datasets operate under strict, immutable roles:

| Dataset | Role | Primary Function | Prohibited Operations |
| :--- | :--- | :--- | :--- |
| **RiceLeafDiseaseBD** | `PRIMARY_DEVELOPMENT_DATASET` | Model training, validation, internal testing, lesion supervision via bounding boxes | Mixed with external test sets |
| **Sethy et al. (5,932 images)** | `EXTERNAL_DATASET` | Held-out cross-dataset generalization benchmark | Model training, hyperparameter tuning, model selection |
| **RiceLeafDisease-BD5** | `EXTERNAL_FIELD_DATASET` | In-the-wild field robustness & domain shift evaluation | Model training; merging Narrow Brown Spot with Brown Spot |
| **RiceSeg-5932 (5,932 masks)** | `XAI_GROUND_TRUTH` | Ground-truth masks for quantitative XAI attribution faithfulness | Image training dataset |

---

## 3. Computational Architecture & Workflow

```
+------------------------------------+
|  LOCAL WORKSPACE (VS Code / Git)   | -> Development, architecture, version control
+-----------------+------------------+
                  |  Git Push / Pull
                  v
+-----------------+------------------+
|    GOOGLE COLAB FREE TIER          | -> Authoritative ML compute, GPU dynamic detection
+-----------------+------------------+
                  |  Mount
                  v
+-----------------+------------------+
|      GOOGLE DRIVE PERSISTENCE      | -> Datasets, manifests, reports, checkpoints, logs
+------------------------------------+
```

---

## 4. Repository Structure

```
XAI-RiceGuard/
│
├── README.md                                 # Main project documentation
├── .gitignore                                # Excludes raw datasets, checkpoints, temp files
├── requirements.txt                          # Dependency specifications
│
├── configs/
│   ├── project_config.yaml                   # Global project & reproducibility config
│   └── paths_config.yaml                     # Colab & Local paths resolution
│
├── src/
│   └── phase0/                               # Phase 0 verification modules
│       ├── environment.py                    # Dynamic runtime & GPU detection
│       ├── config.py                         # Path and configuration resolution
│       ├── dataset_discovery.py              # Directory traversal & inventory
│       ├── image_audit.py                    # Non-destructive image integrity checker
│       ├── annotation_audit.py               # RiceLeafDiseaseBD YOLO bbox & metadata audit
│       ├── duplicate_audit.py                # Intra & cross-dataset duplicate checker
│       ├── riceseg_audit.py                  # Sethy <-> RiceSeg pairing auditor
│       ├── manifest_generator.py             # CSV/JSON manifest generator
│       └── report_generator.py               # Markdown audit report builder
│
├── scripts/
│   └── run_phase0_audit.py                   # Automated Phase 0 execution CLI
│
├── notebooks/
│   └── 00_phase0_project_foundation.ipynb    # Interactive Colab audit notebook
│
├── manifests/                                # Standardized dataset manifests (.csv, .json)
├── reports/                                  # Audit & completion markdown reports
├── environment/                              # Environment snapshots
├── docs/                                     # Research policies (leakage, seeds, roles)
├── tests/                                    # Phase 0 unit test suite
├── experiments/                              # Experiment registry & tracking
├── checkpoints/                              # Checkpoint policy & storage
├── results/                                  # Metric summaries
├── figures/                                  # Generated plots & visualizations
├── tables/                                   # Formatted publication tables
└── logs/                                     # Execution logs
```

---

## 5. Phase 0 Execution Instructions

### A. Authoritative Execution in Google Colab
1. Upload/clone this repository into your Google Colab environment.
2. Open `notebooks/00_phase0_project_foundation.ipynb`.
3. Set your runtime accelerator (GPU if available, or standard CPU).
4. Run all cells to mount Google Drive, audit datasets, inspect YOLO labels and RiceSeg pairings, and generate persistent reports in Google Drive.

### B. Local Development & Verification
To run the automated audit on local dataset copies:
```bash
python scripts/run_phase0_audit.py
```
To run the unit test suite:
```bash
python -m unittest discover tests
```

---

## 6. Phase 0 Hard Gate & Integrity Notice

> **HARD GATE:** No Phase 1 machine learning training, model fine-tuning, data splitting, augmentation, preprocessing, or XAI generation begins until the Phase 0 audit is verified and approved.
