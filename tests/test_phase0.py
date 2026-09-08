"""Unit tests for Phase 0 modules using Python standard unittest."""

import os
import unittest
import tempfile
from pathlib import Path
from src.phase0.environment import detect_environment, is_google_colab, generate_environment_report
from src.phase0.config import load_project_config, load_paths_config, resolve_paths
from src.phase0.dataset_discovery import discover_dataset_structure
from src.phase0.image_audit import compute_file_sha256
from src.phase0.annotation_audit import audit_single_yolo_file

class TestPhase0(unittest.TestCase):

    def test_environment_detection(self):
        env = detect_environment()
        self.assertIn("timestamp", env)
        self.assertIn("python_version", env)
        self.assertIn("cuda_available", env)
        self.assertIn("packages", env)
        self.assertIsInstance(env["is_colab"], bool)
        
        report = generate_environment_report(env)
        self.assertIn("XAI-RiceGuard: Phase 0 Environment Snapshot", report)

    def test_config_loading(self):
        proj_cfg = load_project_config("configs/project_config.yaml")
        self.assertEqual(proj_cfg["project"]["name"], "XAI-RiceGuard")
        self.assertEqual(proj_cfg["project"]["current_phase"], 0)
        self.assertIn("reproducibility", proj_cfg)
        self.assertEqual(proj_cfg["reproducibility"]["default_seed"], 42)
        self.assertIn("datasets", proj_cfg)
        self.assertEqual(proj_cfg["datasets"]["primary"]["role"], "PRIMARY_DEVELOPMENT_DATASET")

        paths_cfg = load_paths_config("configs/paths_config.yaml")
        self.assertIn("colab", paths_cfg)
        self.assertIn("local", paths_cfg)

    def test_path_resolution(self):
        resolved = resolve_paths("configs/paths_config.yaml", force_env="local")
        self.assertEqual(resolved["environment"], "local")
        self.assertIn("primary_dataset", resolved["dataset_roots"])
        self.assertIn("manifests", resolved["artifacts"])
        self.assertTrue(os.path.exists(resolved["artifacts"]["manifests"]))

    def test_sha256_computation(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("XAI-RiceGuard-Phase0-Verification")
            temp_file = f.name

        try:
            sha = compute_file_sha256(temp_file)
            self.assertIsInstance(sha, str)
            self.assertEqual(len(sha), 64)
        finally:
            os.remove(temp_file)

    def test_yolo_annotation_parsing(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.05 0.05\n")
            temp_file = f.name

        try:
            res = audit_single_yolo_file(temp_file)
            self.assertTrue(res["readable"])
            self.assertEqual(res["bbox_count"], 2)
            self.assertEqual(len(res["syntax_errors"]), 0)
            self.assertEqual(len(res["out_of_bounds"]), 0)
        finally:
            os.remove(temp_file)

if __name__ == "__main__":
    unittest.main()
