"""Dataset Dimension Statistics and Resolution Profiling Module for Phase 1."""

from typing import Dict, Any
import pandas as pd
import numpy as np

def compute_image_dimension_statistics(df_manifest: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate resolution, aspect ratio, mode, and channel statistics from manifest records.
    """
    df = df_manifest.copy()
    
    widths = df["width"].dropna().astype(float)
    heights = df["height"].dropna().astype(float)
    aspect_ratios = df["aspect_ratio"].dropna().astype(float)
    file_sizes_kb = (df["file_size_bytes"].dropna() / 1024.0).astype(float)

    stats = {
        "total_records": len(df),
        "dimensions": {
            "width": {
                "mean": round(float(widths.mean()), 2) if not widths.empty else 0.0,
                "std": round(float(widths.std()), 2) if not widths.empty else 0.0,
                "median": round(float(widths.median()), 2) if not widths.empty else 0.0,
                "min": int(widths.min()) if not widths.empty else 0,
                "max": int(widths.max()) if not widths.empty else 0,
            },
            "height": {
                "mean": round(float(heights.mean()), 2) if not heights.empty else 0.0,
                "std": round(float(heights.std()), 2) if not heights.empty else 0.0,
                "median": round(float(heights.median()), 2) if not heights.empty else 0.0,
                "min": int(heights.min()) if not heights.empty else 0,
                "max": int(heights.max()) if not heights.empty else 0,
            },
            "aspect_ratio": {
                "mean": round(float(aspect_ratios.mean()), 4) if not aspect_ratios.empty else 0.0,
                "std": round(float(aspect_ratios.std()), 4) if not aspect_ratios.empty else 0.0,
                "median": round(float(aspect_ratios.median()), 4) if not aspect_ratios.empty else 0.0,
                "min": round(float(aspect_ratios.min()), 4) if not aspect_ratios.empty else 0.0,
                "max": round(float(aspect_ratios.max()), 4) if not aspect_ratios.empty else 0.0,
            },
            "file_size_kb": {
                "mean": round(float(file_sizes_kb.mean()), 2) if not file_sizes_kb.empty else 0.0,
                "median": round(float(file_sizes_kb.median()), 2) if not file_sizes_kb.empty else 0.0,
                "min": round(float(file_sizes_kb.min()), 2) if not file_sizes_kb.empty else 0.0,
                "max": round(float(file_sizes_kb.max()), 2) if not file_sizes_kb.empty else 0.0,
            }
        },
        "top_resolutions": (df["width"].astype(str) + "x" + df["height"].astype(str)).value_counts().head(5).to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "formats": df["format"].value_counts().to_dict(),
        "channels": df["channels"].value_counts().to_dict()
    }

    return stats
