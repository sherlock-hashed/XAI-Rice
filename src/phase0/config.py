"""Configuration Loader and Path Resolution Module for Phase 0."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from src.phase0.environment import is_google_colab

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_paths_config(config_path: str = "configs/paths_config.yaml") -> Dict[str, Any]:
    """Load paths configuration."""
    return load_yaml_config(config_path)

def load_project_config(config_path: str = "configs/project_config.yaml") -> Dict[str, Any]:
    """Load project configuration."""
    return load_yaml_config(config_path)

def resolve_paths(
    paths_config_path: str = "configs/paths_config.yaml",
    force_env: str = None
) -> Dict[str, Any]:
    """
    Resolve dataset and artifact paths based on the active environment.
    Automatically detects Google Colab vs Local unless overridden.
    """
    cfg = load_paths_config(paths_config_path)
    mode = cfg.get("environment_mode", "auto")

    if force_env:
        env_key = force_env
    elif mode == "auto":
        env_key = "colab" if is_google_colab() else "local"
    else:
        env_key = mode

    if env_key not in cfg:
        raise ValueError(f"Invalid environment key '{env_key}' in paths config.")

    env_cfg = cfg[env_key]
    project_root = Path(env_cfg["project_root"]).resolve()

    # Resolve dataset roots
    dataset_roots = {}
    for key, rel_or_abs in env_cfg.get("dataset_roots", {}).items():
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        dataset_roots[key] = str(p)

    # Resolve artifact directories and ensure they exist
    artifact_dirs = {}
    for key, rel_or_abs in env_cfg.get("artifacts", {}).items():
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        artifact_dirs[key] = str(p)
        # Create directory safely if needed
        p.mkdir(parents=True, exist_ok=True)

    return {
        "environment": env_key,
        "is_colab": (env_key == "colab"),
        "project_root": str(project_root),
        "dataset_roots": dataset_roots,
        "artifacts": artifact_dirs,
        "supported_extensions": cfg.get("supported_extensions", {})
    }
