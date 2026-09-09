"""
Computational Profiler for Hardware Resource and Inference Throughput Monitoring.
"""

import time
from typing import Dict, Any, Optional


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Returns GPU memory usage statistics if CUDA is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
            total = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(dev),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2)
            }
    except Exception:
        pass
        
    return {
        "available": False,
        "device_name": "CPU",
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "total_gb": 0.0
    }


def profile_inference(
    model: Any,
    dataloader: Any,
    device: str = "cpu",
    warmup_batches: int = 2
) -> Dict[str, Any]:
    """
    Measures evaluation inference latency, throughput (images/sec), and memory footprint.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for inference profiling.")
        
    model.eval()
    model.to(device)
    
    total_images = 0
    total_time = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            bs = images.size(0)
            
            # Warmup
            if i < warmup_batches:
                _ = model(images)
                if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
                    torch.cuda.synchronize()
                continue
                
            start_t = time.perf_counter()
            _ = model(images)
            if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
                torch.cuda.synchronize()
            batch_t = time.perf_counter() - start_t
            
            total_images += bs
            total_time += batch_t
            
    avg_per_image_ms = (total_time / max(1, total_images)) * 1000.0
    throughput = total_images / max(1e-6, total_time)
    
    return {
        "device": str(device),
        "total_profiled_images": total_images,
        "total_time_seconds": round(total_time, 4),
        "avg_latency_ms_per_image": round(avg_per_image_ms, 2),
        "throughput_images_per_sec": round(throughput, 2)
    }
