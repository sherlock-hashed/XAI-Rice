"""Non-Destructive Image Integrity Audit Module for Phase 0.
Optimized with Multi-Threaded Parallel I/O for high-latency Google Drive FUSE filesystems.
"""

import os
import io
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

def compute_file_sha256_from_bytes(data: bytes) -> str:
    """Compute SHA-256 hash directly from in-memory byte buffer."""
    return hashlib.sha256(data).hexdigest()

def compute_file_sha256(file_path: str, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file efficiently in chunked blocks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def inspect_single_image(
    file_path: str,
    dataset_root: str,
    compute_hash: bool = True
) -> Dict[str, Any]:
    """
    Non-destructively inspect image properties, dimensions, mode, and integrity.
    Reads file once into memory to minimize network round-trips on Google Drive FUSE.
    """
    path = Path(file_path)
    rel_path = os.path.relpath(str(path), dataset_root)
    file_size = 0

    record: Dict[str, Any] = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "relative_path": rel_path.replace("\\", "/"),
        "file_size_bytes": 0,
        "readable": False,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "mode": None,
        "channels": None,
        "format": None,
        "sha256": None,
        "anomalies": []
    }

    try:
        # Single network I/O read into memory
        data = path.read_bytes()
        file_size = len(data)
        record["file_size_bytes"] = file_size
    except Exception as e:
        record["anomalies"].append(f"read_error: {str(e)}")
        return record

    if file_size == 0:
        record["anomalies"].append("zero_byte_file")
        return record

    # Compute SHA-256 in-memory (zero extra network I/O)
    if compute_hash:
        try:
            record["sha256"] = compute_file_sha256_from_bytes(data)
        except Exception as e:
            record["anomalies"].append(f"hash_error: {str(e)}")

    # Non-destructive header & content readability check from in-memory stream
    try:
        with Image.open(io.BytesIO(data)) as img:
            record["width"], record["height"] = img.size
            record["mode"] = img.mode
            record["format"] = img.format
            record["readable"] = True

            # Determine channel count
            mode_channel_map = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "I": 1, "F": 1}
            record["channels"] = mode_channel_map.get(img.mode, len(img.getbands()))

            if record["height"] > 0:
                record["aspect_ratio"] = round(record["width"] / record["height"], 4)

            # Check for anomalies
            if record["channels"] == 1:
                record["anomalies"].append("single_channel_grayscale")
            elif record["channels"] == 4:
                record["anomalies"].append("four_channel_rgba")

            if file_size < 1024:
                record["anomalies"].append("suspiciously_small_file (<1KB)")

            if record["aspect_ratio"] and (record["aspect_ratio"] > 4.0 or record["aspect_ratio"] < 0.25):
                record["anomalies"].append("extreme_aspect_ratio")

    except Exception as e:
        record["readable"] = False
        record["anomalies"].append(f"corrupted_or_unreadable: {str(e)}")

    return record

def _process_image_worker(args):
    """Helper worker for multi-threaded parallel execution."""
    full_path_str, root_str, compute_hashes, rel_p, class_name, dataset_name = args
    rec = inspect_single_image(full_path_str, root_str, compute_hash=compute_hashes)
    rec["class_name"] = class_name
    rec["dataset_name"] = dataset_name
    return rec

def audit_images(
    dataset_name: str,
    root_path: str,
    image_relative_paths: List[str],
    compute_hashes: bool = True,
    class_extractor: Optional[callable] = None,
    max_workers: int = 16
) -> Dict[str, Any]:
    """
    Perform comprehensive integrity audit on a collection of image paths
    using multi-threaded parallel execution.
    """
    records = []
    class_counts: Dict[str, int] = {}
    dimensions_summary: Dict[str, int] = {}
    modes_summary: Dict[str, int] = {}
    formats_summary: Dict[str, int] = {}
    anomalies_detected: List[Dict[str, Any]] = []
    unreadable_count = 0

    root = Path(root_path)

    tasks = []
    for rel_p in image_relative_paths:
        full_path = root / rel_p
        if class_extractor:
            class_name = class_extractor(rel_p)
        else:
            parts = Path(rel_p).parts
            class_name = parts[0] if len(parts) > 1 else "root"
        
        tasks.append((str(full_path), str(root), compute_hashes, rel_p, class_name, dataset_name))

    # Multi-threaded parallel execution
    num_workers = min(max_workers, len(tasks)) if tasks else 1
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for rec in executor.map(_process_image_worker, tasks):
            records.append(rec)
            class_name = rec["class_name"]

            if not rec["readable"]:
                unreadable_count += 1

            if rec["anomalies"]:
                anomalies_detected.append({
                    "relative_path": rec["relative_path"],
                    "anomalies": rec["anomalies"]
                })

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if rec["width"] and rec["height"]:
                dim_key = f"{rec['width']}x{rec['height']}"
                dimensions_summary[dim_key] = dimensions_summary.get(dim_key, 0) + 1

            if rec["mode"]:
                modes_summary[rec["mode"]] = modes_summary.get(rec["mode"], 0) + 1

            if rec["format"]:
                formats_summary[rec["format"]] = formats_summary.get(rec["format"], 0) + 1

    total_images = len(records)
    class_percentages = {
        cls: round((cnt / total_images) * 100, 2) if total_images > 0 else 0
        for cls, cnt in class_counts.items()
    }

    counts_list = list(class_counts.values()) if class_counts else [1]
    max_count = max(counts_list) if counts_list else 0
    min_count = min(counts_list) if counts_list else 0
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else 0

    return {
        "dataset_name": dataset_name,
        "total_images": total_images,
        "readable_count": total_images - unreadable_count,
        "unreadable_count": unreadable_count,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "largest_class": max(class_counts, key=class_counts.get) if class_counts else "None",
        "smallest_class": min(class_counts, key=class_counts.get) if class_counts else "None",
        "imbalance_ratio": imbalance_ratio,
        "dimensions_summary": dict(sorted(dimensions_summary.items(), key=lambda x: x[1], reverse=True)[:10]),
        "modes_summary": modes_summary,
        "formats_summary": formats_summary,
        "total_anomalies_flagged": len(anomalies_detected),
        "anomalies_sample": anomalies_detected[:20],
        "image_records": records
    }
