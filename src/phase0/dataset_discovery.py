"""Dataset Discovery and Directory Traversal Module for Phase 0."""

import os
from pathlib import Path
from typing import Dict, Any, List, Set

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def discover_dataset_structure(
    root_path: str,
    supported_image_exts: Set[str] = None
) -> Dict[str, Any]:
    """
    Safely and non-destructively inspect a dataset directory on disk/Drive.
    Categorizes subdirectories, image files, non-image files, and extensions.
    """
    if supported_image_exts is None:
        supported_image_exts = SUPPORTED_IMAGE_EXTS
    else:
        supported_image_exts = {ext.lower() for ext in supported_image_exts}

    path = Path(root_path)
    if not path.exists():
        return {
            "exists": False,
            "root_path": str(path.resolve()) if path.is_absolute() else str(path),
            "error": "Directory does not exist."
        }

    total_files = 0
    total_dirs = 0
    total_size_bytes = 0
    image_files: List[str] = []
    non_image_files: List[str] = []
    extension_counts: Dict[str, int] = {}
    subdirectories: List[str] = []
    immediate_subdirs: List[str] = []

    try:
        immediate_subdirs = [d.name for d in path.iterdir() if d.is_dir()]
    except Exception as e:
        immediate_subdirs = []

    for dirpath, dirnames, filenames in os.walk(path):
        rel_dir = os.path.relpath(dirpath, path)
        if rel_dir != ".":
            subdirectories.append(rel_dir)
        total_dirs += len(dirnames)

        for filename in filenames:
            total_files += 1
            ext = os.path.splitext(filename)[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            full_file_path = os.path.join(dirpath, filename)
            
            try:
                size = os.path.getsize(full_file_path)
                total_size_bytes += size
            except OSError:
                size = 0

            if ext in supported_image_exts:
                image_files.append(os.path.relpath(full_file_path, path))
            else:
                non_image_files.append(os.path.relpath(full_file_path, path))

    return {
        "exists": True,
        "root_path": str(path.resolve()),
        "immediate_subdirs": sorted(immediate_subdirs),
        "total_subdirectories": total_dirs,
        "subdirectories_sample": sorted(subdirectories)[:20],
        "total_files": total_files,
        "total_images": len(image_files),
        "total_non_images": len(non_image_files),
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "extension_counts": extension_counts,
        "sample_image_paths": sorted(image_files)[:10],
        "sample_non_image_paths": sorted(non_image_files)[:10]
    }
