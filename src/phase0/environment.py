"""Environment Detection and Package Snapshot Module for Phase 0."""

import os
import sys
import platform
import subprocess
import datetime
from typing import Dict, Any, Optional

def is_google_colab() -> bool:
    """Detect whether the current runtime is Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return 'COLAB_GPU' in os.environ or 'GCS_READ_CACHE' in os.environ

def get_git_commit_hash() -> Optional[str]:
    """Retrieve the current Git commit hash if running in a Git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "Not available (Not a git repo or git not in PATH)"

def detect_environment() -> Dict[str, Any]:
    """Dynamically probe operating system, Python, PyTorch, GPU/accelerator, and libraries."""
    env_info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_colab": is_google_colab(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "git_commit": get_git_commit_hash(),
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "packages": {}
    }

    # Deep learning & accelerator probing
    try:
        import torch
        env_info["packages"]["torch"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024 ** 3)
                env_info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(total_mem_gb, 2),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
        else:
            env_info["cuda_version"] = None
    except ImportError:
        env_info["packages"]["torch"] = "Not installed"

    # Core Scientific & Computer Vision Libraries
    package_names = [
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "pillow"),
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("yaml", "pyyaml"),
        ("openpyxl", "openpyxl"),
    ]

    for mod_name, pkg_label in package_names:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed (version unknown)")
            env_info["packages"][pkg_label] = ver
        except ImportError:
            env_info["packages"][pkg_label] = "Not installed"

    return env_info

def generate_environment_report(env_info: Optional[Dict[str, Any]] = None) -> str:
    """Format environment information into an IEEE-ready text snapshot."""
    if env_info is None:
        env_info = detect_environment()

    lines = [
        "=" * 70,
        "XAI-RiceGuard: Phase 0 Environment Snapshot",
        "=" * 70,
        f"Timestamp (UTC): {env_info.get('timestamp')}",
        f"Execution Environment: {'Google Colab' if env_info.get('is_colab') else 'Local / VS Code'}",
        f"Operating System: {env_info.get('system')} ({env_info.get('platform')})",
        f"Architecture: {env_info.get('architecture')} / Processor: {env_info.get('processor')}",
        f"Python Version: {env_info.get('python_version')}",
        f"Git Commit Hash: {env_info.get('git_commit')}",
        "-" * 70,
        "Hardware & Accelerator Status:",
    ]

    if env_info.get("cuda_available"):
        lines.append(f"CUDA Available: YES (CUDA Version: {env_info.get('cuda_version')})")
        lines.append(f"Detected GPU Count: {env_info.get('gpu_count')}")
        for gpu in env_info.get("gpus", []):
            lines.append(f"  [GPU {gpu['index']}] {gpu['name']} | VRAM: {gpu['total_memory_gb']} GB | Compute Capability: {gpu['compute_capability']}")
    else:
        lines.append("CUDA Available: NO")
        lines.append("Note: Running on CPU / Integrated Graphics. (Expected for local dev / unassigned Colab free tier).")

    lines.extend([
        "-" * 70,
        "Core Package Versions:",
    ])

    for pkg, ver in env_info.get("packages", {}).items():
        lines.append(f"  - {pkg:<18}: {ver}")

    lines.append("=" * 70)
    return "\n".join(lines)
