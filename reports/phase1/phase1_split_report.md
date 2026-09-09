# Phase 1: Group-Aware Stratified Split & Leakage Report

**Project:** XAI-RiceGuard  
**Random Seed:** `42`  
**Splitting Strategy:** Constrained optimization of class distribution alignment subject to atomic duplicate group isolation ($G_i \cap G_j = \emptyset$).

---

## 1. Experimental Split Summary

| Partition | Target Count | Actual Count | Target % | Actual % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 12,574 | 12,568 | 70.0% | 69.97% |
| **Validation** | 1,796 | 1,798 | 10.0% | 10.01% |
| **Calibration** | 1,796 | 1,799 | 10.0% | 10.02% |
| **Internal Test** | 1,796 | 1,798 | 10.0% | 10.01% |

---

## 2. Class Distribution by Partition

| Disease Class | Train | Validation | Calibration | Internal Test | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Healthy` | 1,101 | 158 | 158 | 158 | 1,575 |
| `Blast` | 1,856 | 265 | 266 | 265 | 2,652 |
| `Brown spot` | 3,048 | 436 | 436 | 436 | 4,356 |
| `Leaf smut` | 1,013 | 145 | 145 | 145 | 1,448 |
| `Rice Tungro` | 3,141 | 449 | 449 | 449 | 4,488 |
| `Sheath blight` | 2,409 | 345 | 345 | 345 | 3,444 |

---

## 3. Cryptographic Manifest Checksums (Frozen Split Baseline)

| Manifest File | SHA-256 Checksum | Lifecycle Role |
| :--- | :--- | :--- |
| `primary_all_splits.csv` | `f71f5b6fcca523441c9e8eed54fc16f85be1b53d23a4db6ef391c1ebd07ecd8e` | Experimental Partition |
| `primary_train.csv` | `e8f7281917806aa8f02b74fc99864a2bafe7fc877f8892127f70b228c1f8dda7` | Experimental Partition |
| `primary_validation.csv` | `8c7dff4e323d49eb108b0dc5272dd4fdc7fa45e3819c2bb30276696b09766ba1` | Experimental Partition |
| `primary_calibration.csv` | `81154b79dc3f11e3b5402c5b9f310a68b1420b6deab589eb50a3312c6624f990` | Experimental Partition |
| `primary_internal_test.csv` | `46fbac17a068995072ac6604855bae86c197f43a5cfe0ceed6e385cd1bb7390f` | Frozen Internal Test (LOCKED) |
| `duplicate_groups.csv` | `1d8e88679b6fed4455ab8e801e6f0d31adc5c0beb9afe87e7b90a8939796461d` | Experimental Partition |
| `sethy_external.csv` | `0f1133c0e4f7a58857496fe4c3cc8ae066eda3e609b4fdf41a42b10529a842de` | Experimental Partition |
| `bd5_external.csv` | `8cd9e9ba931e844c7d005d13d2fcd85fc4dd10034a6fbb5a1daab6a5f96a8289` | Experimental Partition |
| `riceseg_ground_truth.csv` | `cc308c8184bb35236c48a53d9deafb354581f3ff5e4d612fdafecdc7215918e7` | Experimental Partition |