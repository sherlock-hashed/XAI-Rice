# Phase 1: Comprehensive Dataset Statistics & Profiling Report

**Project:** XAI-RiceGuard  
**Date (UTC):** 2026-09-09T05:10:05.780131+00:00  
**Runtime:** Local Environment  
**Git Commit:** `ceddff4deebcc8cac836939f2b5618b71e87c770`  

---

## 1. Primary Dataset Overview (RiceLeafDiseaseBD)
- **Total Primary Images:** `17,963`
- **Number of Disease Classes:** `6`
- **Largest Class:** `Rice Tungro` | **Smallest Class:** `Leaf smut`
- **Imbalance Ratio:** `3.1:1`

### Class Distribution Table
| Class ID | Class Name | Sample Count | Percentage (%) |
| :---: | :--- | :---: | :---: |
| 0 | `Healthy` | 1,575 | 8.77% |
| 1 | `Blast` | 2,652 | 14.76% |
| 2 | `Brown spot` | 4,356 | 24.25% |
| 3 | `Leaf smut` | 1,448 | 8.06% |
| 4 | `Rice Tungro` | 4,488 | 24.98% |
| 5 | `Sheath blight` | 3,444 | 19.17% |

---

## 2. Duplicate Group Analysis
- **Total Unique SHA-256 Hashes:** `17,862`
- **Duplicate Hash Groups:** `60`
- **Unique (Single-Sample) Groups:** `17,802`
- **Samples in Duplicate Clusters:** `161`
- **Samples in Unique Groups:** `17,802`
- **Max Group Cluster Size:** `11`

---

## 3. Image Dimensions and Resolution Profiling
| Metric | Width (px) | Height (px) | Aspect Ratio | File Size (KB) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | 1024.0 | 1024.0 | 1.0 | 310.56 |
| **Median** | 1024.0 | 1024.0 | 1.0 | 310.43 |
| **Min** | 1024 | 1024 | 1.0 | 128.77 |
| **Max** | 1024 | 1024 | 1.0 | 658.2 |