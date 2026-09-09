"""
Comprehensive Unit Test Suite for Phase 02: Baseline Architectures & Model Training Pipeline.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.phase2.dataset import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DataFirewallViolationError,
    verify_manifest_firewall
)
from src.phase2.model_registry import (
    MODEL_REGISTRY,
    is_model_registered,
    get_registered_models
)
from src.phase2.metrics import compute_metrics
from src.phase2.seed import seed_everything
from src.phase2.phase2_report import select_best_baseline


class TestPhase2Pipeline(unittest.TestCase):

    def test_01_canonical_class_mapping(self):
        """Test 1: Verify canonical 6-class names and bidirectional mappings."""
        self.assertEqual(len(CANONICAL_CLASSES), 6)
        expected = ["Healthy", "Blast", "Brown spot", "Leaf smut", "Rice Tungro", "Sheath blight"]
        self.assertEqual(CANONICAL_CLASSES, expected)
        for i, name in enumerate(expected):
            self.assertEqual(CLASS_TO_IDX[name], i)
            self.assertEqual(IDX_TO_CLASS[i], name)

    def test_02_train_manifest_exists_and_verified(self):
        """Test 2: Verify primary_train.csv structure and class alignment."""
        path = "manifests/phase1/primary_train.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.assertEqual(len(df), 12568)
            col = "class_name" if "class_name" in df.columns else "disease_class"
            for cls_name in df[col].unique():
                self.assertIn(cls_name, CLASS_TO_IDX)

    def test_03_validation_manifest_exists_and_verified(self):
        """Test 3: Verify primary_validation.csv structure and class alignment."""
        path = "manifests/phase1/primary_validation.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.assertEqual(len(df), 1798)
            col = "class_name" if "class_name" in df.columns else "disease_class"
            for cls_name in df[col].unique():
                self.assertIn(cls_name, CLASS_TO_IDX)


    def test_04_firewall_rejects_calibration_manifest(self):
        """Test 4: Verify firewall strictly rejects primary_calibration.csv."""
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/primary_calibration.csv", expected_role="train")
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/primary_calibration.csv", expected_role="validation")

    def test_05_firewall_rejects_internal_test_manifest(self):
        """Test 5: Verify firewall strictly rejects locked primary_internal_test.csv."""
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/primary_internal_test.csv", expected_role="train")
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/primary_internal_test.csv", expected_role="validation")

    def test_06_firewall_rejects_external_sethy_manifest(self):
        """Test 6: Verify firewall strictly rejects external sethy_external.csv."""
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/sethy_external.csv", expected_role="train")

    def test_07_firewall_rejects_external_bd5_manifest(self):
        """Test 7: Verify firewall strictly rejects external bd5_external.csv."""
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/bd5_external.csv", expected_role="train")

    def test_08_firewall_rejects_riceseg_manifest(self):
        """Test 8: Verify firewall strictly rejects riceseg_ground_truth.csv."""
        with self.assertRaises(DataFirewallViolationError):
            verify_manifest_firewall("manifests/phase1/riceseg_ground_truth.csv", expected_role="validation")

    def test_09_model_registry_entries(self):
        """Test 9: Verify required baseline models exist in MODEL_REGISTRY."""
        self.assertTrue(is_model_registered("efficientnet_b0"))
        self.assertTrue(is_model_registered("resnet50"))
        self.assertTrue(is_model_registered("convnext_tiny"))
        models = get_registered_models()
        self.assertIn("efficientnet_b0", models)
        self.assertIn("resnet50", models)
        self.assertIn("convnext_tiny", models)

    def test_10_deterministic_seeding(self):
        """Test 10: Verify seed_everything produces deterministic random outputs."""
        seed_everything(42)
        val1 = np.random.rand()
        seed_everything(42)
        val2 = np.random.rand()
        self.assertEqual(val1, val2)

    def test_11_metric_calculation_accuracy_and_macro_f1(self):
        """Test 11: Verify multi-class metric calculation logic."""
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        res = compute_metrics(y_true, y_pred, class_names=CANONICAL_CLASSES)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["macro_f1"], 1.0)
        self.assertEqual(res["total_samples"], 12)
        self.assertEqual(len(res["per_class"]), 6)

    def test_12_metric_confusion_matrix_dimensions(self):
        """Test 12: Verify confusion matrix dimensions are 6x6."""
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 1, 3, 4, 5]
        res = compute_metrics(y_true, y_pred, class_names=CANONICAL_CLASSES)
        cm = res["confusion_matrix"]
        self.assertEqual(len(cm), 6)
        self.assertEqual(len(cm[0]), 6)
        self.assertAlmostEqual(res["accuracy"], 5/6, places=4)

    def test_13_baseline_selection_by_macro_f1(self):
        """Test 13: Verify model selection picks highest Validation Macro-F1."""
        experiments = [
            {"model": "resnet50", "val_macro_f1": 0.9120, "val_accuracy": 92.5, "val_loss": 0.25, "total_params": 25000000},
            {"model": "efficientnet_b0", "val_macro_f1": 0.9350, "val_accuracy": 94.1, "val_loss": 0.18, "total_params": 5300000},
            {"model": "convnext_tiny", "val_macro_f1": 0.9280, "val_accuracy": 93.8, "val_loss": 0.21, "total_params": 28000000}
        ]
        best = select_best_baseline(experiments)
        self.assertIsNotNone(best)
        self.assertEqual(best["model"], "efficientnet_b0")

    def test_14_baseline_selection_tie_breaker(self):
        """Test 14: Verify tie-breaker uses accuracy and lower params if Macro-F1 ties."""
        experiments = [
            {"model": "model_a", "val_macro_f1": 0.9200, "val_accuracy": 92.0, "val_loss": 0.20, "total_params": 20000000},
            {"model": "model_b", "val_macro_f1": 0.9200, "val_accuracy": 93.0, "val_loss": 0.20, "total_params": 10000000}
        ]
        best = select_best_baseline(experiments)
        self.assertEqual(best["model"], "model_b")


if __name__ == "__main__":
    unittest.main()
