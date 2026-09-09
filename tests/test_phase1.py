"""Comprehensive Unit Test Suite for Phase 1 (12 IEEE Research Tests)."""

import unittest
import pandas as pd
import numpy as np
from src.phase1.manifest_loader import load_phase0_manifests, validate_manifest_completeness
from src.phase1.duplicate_groups import build_duplicate_groups, validate_duplicate_group_isolation
from src.phase1.label_analysis import analyze_primary_labels, PRIMARY_CLASS_NAMES
from src.phase1.split_generator import generate_group_aware_stratified_splits
from src.phase1.split_validator import validate_phase1_splits
from src.phase1.eda import select_representative_samples

class TestPhase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifests = load_phase0_manifests()
        cls.df_primary = cls.manifests["primary"]
        cls.df_sethy = cls.manifests["sethy"]
        cls.df_bd5 = cls.manifests["bd5"]

    def test_01_manifest_loading(self):
        """Test 1: Manifest loading success and basic columns present."""
        self.assertIn("sample_id", self.df_primary.columns)
        self.assertIn("sha256", self.df_primary.columns)
        self.assertIn("class_name", self.df_primary.columns)
        self.assertIn("annotation_path", self.df_primary.columns)
        self.assertIn("mask_path", self.df_sethy.columns)

    def test_02_manifest_completeness(self):
        """Test 2: Manifest completeness against Phase 0 audited baselines."""
        res = validate_manifest_completeness(self.manifests)
        self.assertTrue(res["all_passed"], f"Manifest completeness failed: {res['details']}")
        self.assertEqual(len(self.df_primary), 17963)
        self.assertEqual(len(self.df_sethy), 5932)
        self.assertEqual(len(self.df_bd5), 3150)

    def test_03_duplicate_grouping(self):
        """Test 3: Deterministic duplicate grouping strictly by SHA-256."""
        df_grouped, df_groups, summary = build_duplicate_groups(self.df_primary)
        self.assertEqual(len(df_grouped), 17963)
        self.assertIn("duplicate_group_id", df_grouped.columns)
        self.assertEqual(summary["total_unique_hashes"], len(df_groups))
        self.assertTrue(summary["total_images_in_duplicate_groups"] > 0)

    def test_04_duplicate_isolation(self):
        """Test 4: Duplicate isolation verification engine."""
        df_grouped, _, _ = build_duplicate_groups(self.df_primary)
        df_splits, _, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        iso_res = validate_duplicate_group_isolation(df_splits)
        self.assertTrue(iso_res["passed"], f"Duplicate group isolation failed: {iso_res['leaked_groups_sample']}")

    def test_05_split_coverage(self):
        """Test 5: 100% split coverage (every primary image assigned)."""
        df_grouped, _, _ = build_duplicate_groups(self.df_primary)
        df_splits, split_dfs, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        self.assertEqual(len(df_splits), 17963)
        self.assertEqual(df_splits["split"].isna().sum(), 0)
        total_split_rows = sum(len(df) for df in split_dfs.values())
        self.assertEqual(total_split_rows, 17963)

    def test_06_split_exclusivity(self):
        """Test 6: Mutual exclusivity across Train, Val, Cal, Test."""
        df_grouped, _, _ = build_duplicate_groups(self.df_primary)
        _, split_dfs, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        
        train_ids = set(split_dfs["train"]["sample_id"])
        val_ids = set(split_dfs["validation"]["sample_id"])
        cal_ids = set(split_dfs["calibration"]["sample_id"])
        test_ids = set(split_dfs["internal_test"]["sample_id"])
        
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(cal_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(cal_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)
        self.assertEqual(len(cal_ids.intersection(test_ids)), 0)

    def test_07_class_representation(self):
        """Test 7: All 6 disease classes represented across all 4 partitions."""
        df_grouped, _, _ = build_duplicate_groups(self.df_primary)
        _, split_dfs, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        
        for s_name, s_df in split_dfs.items():
            classes_in_split = set(s_df["class_name"].unique())
            for c in PRIMARY_CLASS_NAMES:
                self.assertIn(c, classes_in_split, f"Class {c} missing in split {s_name}")

    def test_08_cross_dataset_collision_detection(self):
        """Test 8: Cross-dataset SHA-256 collision verification."""
        p_hashes = set(self.df_primary["sha256"])
        s_hashes = set(self.df_sethy["sha256"])
        b_hashes = set(self.df_bd5["sha256"])
        
        self.assertEqual(len(p_hashes.intersection(s_hashes)), 0)
        self.assertEqual(len(p_hashes.intersection(b_hashes)), 0)
        self.assertEqual(len(s_hashes.intersection(b_hashes)), 0)

    def test_09_external_dataset_firewall(self):
        """Test 9: External dataset firewall constraint."""
        df_grouped, _, _ = build_duplicate_groups(self.df_primary)
        df_splits, split_dfs, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        val_res = validate_phase1_splits(df_splits, split_dfs, self.df_sethy, self.df_bd5)
        self.assertTrue(val_res["checks"]["external_dataset_firewall"]["passed"])

    def test_10_manifest_reproducibility(self):
        """Test 10: Deterministic reproducibility (Run A vs Run B produce exact same split assignments)."""
        df_grouped, _, _ = build_duplicate_groups(self.df_primary)
        df_splits_a, _, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        df_splits_b, _, _ = generate_group_aware_stratified_splits(df_grouped, seed=42)
        
        pd.testing.assert_series_equal(df_splits_a["split"], df_splits_b["split"])

    def test_11_deterministic_sample_selection(self):
        """Test 11: Sample selection for EDA is deterministic with fixed seed."""
        df_samples_1 = select_representative_samples(self.df_primary, self.df_sethy, self.df_bd5, seed=42)
        df_samples_2 = select_representative_samples(self.df_primary, self.df_sethy, self.df_bd5, seed=42)
        pd.testing.assert_frame_equal(df_samples_1, df_samples_2)

    def test_12_riceseg_pairing_preservation(self):
        """Test 12: RiceSeg-5932 mask pairing is 100% preserved (5932/5932)."""
        matched_masks = int(self.df_sethy["mask_path"].notna().sum())
        self.assertEqual(matched_masks, 5932)

if __name__ == "__main__":
    unittest.main()
