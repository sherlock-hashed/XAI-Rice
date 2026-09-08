# Phase 0: Comprehensive Dataset & Integrity Audit Report

**Project:** XAI-RiceGuard  
**Title:** A Lesion-Grounded Explainable and Uncertainty-Aware Deep Learning Framework for Robust Rice Leaf Blast and Brown Spot Detection  
**Audit Generated (UTC):** 2026-09-08T23:07:47.613428+00:00  
**Runtime:** Local Development Environment  
**Git Commit:** `Not available (Not a git repo or git not in PATH)`  

---

## 1. Executive Summary & Expected vs. Verified Status

| Dataset | Expected Role | Expected Size/Count | Verified Physical Status | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| **RiceLeafDiseaseBD** | Specified in Protocol | Audited: 17963 files | Verified on disk/Drive | `PASS (17963/17963 readable)` |
| **Sethy_Rice_Leaf_Disease** | Specified in Protocol | Audited: 5932 files | Verified on disk/Drive | `PASS (5932/5932 readable)` |
| **RiceLeafDisease_BD5** | Specified in Protocol | Audited: 3150 files | Verified on disk/Drive | `PASS (3150/3150 readable)` |

---

## 2. Immutable Dataset Role Allocation

```
+------------------------------------------------------------------------------------+
|                                IMMUTABLE DATASET ROLES                             |
+------------------------------------+-----------------------------------------------+
| Dataset                            | Role & Permitted Lifecycle Scope              |
+------------------------------------+-----------------------------------------------+
| RiceLeafDiseaseBD                  | PRIMARY DEVELOPMENT DATASET                   |
|                                    | (Training, Validation, Internal Test, BBox)   |
+------------------------------------+-----------------------------------------------+
| Sethy et al. (5,932 images)        | EXTERNAL TEST DATASET                         |
|                                    | (Cross-Dataset Generalization Only)           |
+------------------------------------+-----------------------------------------------+
| RiceLeafDisease-BD5 (Field)        | EXTERNAL FIELD DOMAIN DATASET                 |
|                                    | (Out-of-Distribution & Robustness Only)       |
+------------------------------------+-----------------------------------------------+
| RiceSeg-5932 (5,932 masks)         | XAI LESION GROUND TRUTH                       |
|                                    | (Quantitative Attribution Mask Evaluation)    |
+------------------------------------+-----------------------------------------------+
```

---

## 3. Class Distribution & Imbalance Audit

### Dataset: RiceLeafDiseaseBD
- **Total Images Audited:** 17963
- **Largest Class:** `Rice Tungro` | **Smallest Class:** `Leaf smut`
- **Imbalance Ratio (Max/Min):** `3.1:1`

| Class Name | Image Count | Percentage (%) |
| :--- | :--- | :--- |
| `Rice Tungro` | 4488 | 24.98% |
| `Brown spot` | 4356 | 24.25% |
| `Sheath blight` | 3444 | 19.17% |
| `Blast` | 2652 | 14.76% |
| `Healthy` | 1575 | 8.77% |
| `Leaf smut` | 1448 | 8.06% |

### Dataset: Sethy_Rice_Leaf_Disease
- **Total Images Audited:** 5932
- **Largest Class:** `Brownspot` | **Smallest Class:** `Tungro`
- **Imbalance Ratio (Max/Min):** `1.22:1`

| Class Name | Image Count | Percentage (%) |
| :--- | :--- | :--- |
| `Brownspot` | 1600 | 26.97% |
| `Bacterialblight` | 1584 | 26.70% |
| `Blast` | 1440 | 24.28% |
| `Tungro` | 1308 | 22.05% |

### Dataset: RiceLeafDisease_BD5
- **Total Images Audited:** 3150
- **Largest Class:** `Normal_Leaf` | **Smallest Class:** `Narrow_Brown_Spot`
- **Imbalance Ratio (Max/Min):** `1.75:1`

| Class Name | Image Count | Percentage (%) |
| :--- | :--- | :--- |
| `Normal_Leaf` | 834 | 26.48% |
| `Blast` | 720 | 22.86% |
| `Sheath_Blight` | 626 | 19.87% |
| `Tungro` | 494 | 15.68% |
| `Narrow_Brown_Spot` | 476 | 15.11% |

---

## 4. Image Properties, Dimensions, and Format Verification

### RiceLeafDiseaseBD - Properties Summary
- **Modes Detected:** `{'RGB': 17963}`
- **Formats Detected:** `{'JPEG': 17963}`
- **Top Dimensions:** `{'1024x1024': 17963}`
- **Total Flagged Anomalies:** 0

### Sethy_Rice_Leaf_Disease - Properties Summary
- **Modes Detected:** `{'RGB': 5776, 'RGBA': 156}`
- **Formats Detected:** `{'JPEG': 5286, 'PNG': 646}`
- **Top Dimensions:** `{'300x300': 4624, '295x221': 12, '499x332': 12, '221x295': 12, '288x216': 11}`
- **Total Flagged Anomalies:** 156

### RiceLeafDisease_BD5 - Properties Summary
- **Modes Detected:** `{'RGB': 3150}`
- **Formats Detected:** `{'JPEG': 3150}`
- **Top Dimensions:** `{'300x300': 3150}`
- **Total Flagged Anomalies:** 0

---

## 5. RiceLeafDiseaseBD: YOLO Bounding Box & Metadata Audit

- **Annotation Directory Located:** `True`
- **Total YOLO Label Files (`.txt`):** `8194`
- **Total Visual Renderings (`.jpg`):** `8194`
- **Total Bounding Boxes Verified:** `70067`
- **YOLO Syntax Errors:** `0`
- **Coordinates Out of Bounds [0, 1]:** `0`

### Class-Level Annotation Breakdown
| Class | YOLO Label Files | Visual Overlays | Total BBoxes | Unique Class IDs | Syntax Errors | Coordinate Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Blast` | 1326 | 1326 | 12279 | `[1]` | 0 | 0 |
| `Brown spot` | 2178 | 2178 | 21885 | `[0]` | 0 | 0 |
| `Leaf smut` | 724 | 724 | 3885 | `[3]` | 0 | 0 |
| `Rice Tungro` | 2244 | 2244 | 14762 | `[3]` | 0 | 0 |
| `Sheath blight` | 1722 | 1722 | 17256 | `[4]` | 0 | 0 |

### Metadata & Documentation Files in RiceLeafDiseaseBD
| Document Name | Present | File Size (Bytes) |
| :--- | :--- | :--- |
| `Dataset metadata.xlsx` | YES | 467612 |
| `Annotation Protocol.pdf` | YES | 137333 |
| `README.md` | YES | 8049 |
| `Data Collection Pipeline.png` | YES | 6331045 |
| `Dataset folder Structure.png` | YES | 4811090 |

---

## 6. Sethy <-> RiceSeg-5932 Mask Pairing Audit

- **Total Sethy Images:** `5932`
- **Total RiceSeg Masks:** `5932`
- **Exact Stem Matches:** `5932`
- **Unmatched Images:** `0`
- **Unmatched Masks:** `0`
- **Dimension Mismatches:** `0`

### Class-Level Image-Mask Alignment
| Class Name | Sethy Images | RiceSeg Masks | Matched Pairs | Unmatched Img | Unmatched Mask | Dim Mismatch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Bacterialblight` | 1584 | 1584 | 1584 | 0 | 0 | 0 |
| `Blast` | 1440 | 1440 | 1440 | 0 | 0 | 0 |
| `Brownspot` | 1600 | 1600 | 1600 | 0 | 0 | 0 |
| `Tungro` | 1308 | 1308 | 1308 | 0 | 0 | 0 |

---

## 7. Duplicate Hashes & Cross-Dataset Contamination Risk

- **Intra-Dataset Duplicate Hashes:** `1158`
- **Cross-Dataset Collision Risk Detected:** `NO`

---

## 8. Machine-Readable Manifests Generated

| Dataset | Role | Records | CSV Path | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| `RiceLeafDiseaseBD` | Verified Role | 17963 | `riceleafdiseasebd_manifest.csv` | `7d178d22d3f5d386...` |
| `Sethy_Rice_Leaf_Disease` | Verified Role | 5932 | `sethy_rice_leaf_disease_manifest.csv` | `b6d6c9fe519ea20b...` |
| `RiceLeafDisease_BD5` | Verified Role | 3150 | `riceleafdisease_bd5_manifest.csv` | `76b59634b64c6359...` |

---

## 9. Conclusion & Research Invariant Summary
1. **RiceLeafDiseaseBD** stands verified as the primary development dataset containing verified YOLO bounding boxes.
2. **Sethy et al.** is verified with 5,932 images and strictly isolated as an external generalization benchmark.
3. **RiceSeg-5932** is verified with 5,932 masks matching Sethy stems 1-to-1 for quantitative XAI evaluation.
4. **RiceLeafDisease-BD5** stands verified for field robustness testing, with `Narrow_Brown_Spot` preserved as a distinct class.
5. **Zero model training, data splitting, or image modification** was performed during this audit.