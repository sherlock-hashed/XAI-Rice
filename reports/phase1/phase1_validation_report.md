# Phase 1: Experimental Split & Leakage Validation Report

**Project:** XAI-RiceGuard  
**Validation Status:** `PASS`  

---

## 1. Hard-Gate Integrity Checklist

| Integrity Check | Status | Verification Detail |
| :--- | :---: | :--- |
| **Primary Completeness** | `PASS` | Total samples: 17963 (Expected: 17963) |
| **Mutual Exclusivity** | `PASS` | Splits are strictly pairwise disjoint |
| **Duplicate Group Isolation** | `PASS` | 0 duplicate groups cross splits (17862 groups audited) |
| **Class Representation** | `PASS` | All 6 disease classes represented across Train, Val, Cal, and Test |
| **Cross Dataset Leakage** | `PASS` | Primary intersect Sethy: 0, Primary intersect BD5: 0, Sethy intersect BD5: 0 |
| **External Dataset Firewall** | `PASS` | Sethy and BD5 are 100% firewalled from the primary training/validation/calibration/test splits |

---

## 2. Partition Distribution Drift Metrics

| Partition | Max Class % Deviation | $L_1$ Total Variation Distance |
| :--- | :---: | :---: |
| **Train** | 0.01% | 0.0001 |
| **Validation** | 0.03% | 0.0004 |
| **Calibration** | 0.03% | 0.0004 |
| **Internal Test** | 0.03% | 0.0004 |