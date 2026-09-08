# Google Colab Free-Tier Resilience & Checkpointing Policy

**Project:** XAI-RiceGuard  
**Version:** 0.1.0-phase0  

---

## 1. Operating Assumptions & Constraints

1. **Non-Guaranteed Hardware**: Runtime may assign T4, K80, V100, P100, or CPU only.
2. **Ephemeral Disk**: Local storage (`/content/`) is wiped upon session termination or timeout.
3. **Session Disconnects**: Sessions terminate after idle periods or max continuous compute limits (typically 12 hours).
4. **Memory Caps**: System RAM is typically ~12 GB; GPU VRAM varies from 12 GB to 16 GB.

---

## 2. Engineering Directives for Colab-First Research

### Directive 1: Persistent Google Drive Storage
- All critical artifacts (manifests, audit reports, checkpoints, logs, plots) must be written directly to the mounted Google Drive directory: `/content/drive/MyDrive/XAI-RiceGuard/`.
- Local `/content/` can only be used as a fast temporary read cache if needed.

### Directive 2: Atomic Checkpoint Architecture
Every checkpoint must save comprehensive state dictionaries to permit seamless resumption:

```python
checkpoint = {
    'experiment_id': experiment_id,
    'phase': phase_number,
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_metric_val': best_metric,
    'config': full_config_dict,
    'seed': random_seed,
    'git_commit': git_commit_hash,
    'timestamp': datetime.utcnow().isoformat()
}
```

### Directive 3: Dynamic Device & Memory Safeguards
- Code must query `torch.cuda.is_available()`, query `torch.cuda.get_device_name(0)` dynamically, and never hardcode CUDA device indexes.
- Batch sizes and workers in config must be conservative to prevent Colab CUDA Out-Of-Memory (OOM) fatal crashes.
