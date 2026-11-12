"""Validation-only diagnostics and synchronized latency measurements."""
from __future__ import annotations

import time

import numpy as np
import torch

from common import HORIZON, LENGTH


def _validation_batch(model, arrays: dict, limit: int = 32):
    origins = np.asarray(arrays["origins_val"], dtype=np.int64)
    if not len(origins):
        raise ValueError("validation split is empty")
    selected = origins[: min(limit, len(origins))]
    values = arrays["data"]
    x = np.stack([values[t - LENGTH : t] for t in selected]).astype(np.float32, copy=False)
    device = next(model.parameters()).device
    return torch.from_numpy(x).to(device), torch.as_tensor(selected, device=device)


@torch.inference_mode()
def diagnostics(model, bank, arrays: dict, weights, graph: dict) -> dict:
    """Summarize support coverage, gates and retrieval using validation histories only."""
    was_training = model.training
    model.eval()
    x, origins = _validation_batch(model, arrays)
    _, details = model(x, origins, bank, residual_weight=weights)
    incidence = graph["incidence"]
    degree = incidence.sum(dim=1).float()
    support = details["support"].float()
    gate = details["gate"].float()
    variance = details["edge_var"].float()
    memory_origins = bank.origins.detach().cpu().numpy()
    first_val_history = int(np.asarray(arrays["origins_val"])[0]) - LENGTH
    report = {
        "split_used": "validation",
        "validation_windows_sampled": int(len(origins)),
        "test_labels_used": False,
        "nodes": int(incidence.shape[0]),
        "supports": int(incidence.shape[1]),
        "covered_nodes": int((degree > 0).sum().item()),
        "support_degree_mean": float(degree.mean().item()),
        "support_degree_max": int(degree.max().item()),
        "memory_episodes": int(len(memory_origins)),
        "memory_train_only": bool(len(memory_origins) and memory_origins.max() + HORIZON <= first_val_history),
        "memory_origin_min": int(memory_origins.min()) if len(memory_origins) else None,
        "memory_origin_max": int(memory_origins.max()) if len(memory_origins) else None,
        "gate_mean": float(gate.mean().item()),
        "gate_min": float(gate.min().item()),
        "gate_max": float(gate.max().item()),
        "retrieval_support_mean": float(support.mean().item()),
        "retrieval_variance_mean": float(variance.mean().item()),
    }
    if was_training:
        model.train()
    return report


@torch.inference_mode()
def latency(model, bank, arrays: dict, weights, warmup: int = 5, repeats: int = 30) -> dict:
    """Measure synchronized inference on a fixed validation batch."""
    was_training = model.training
    model.eval()
    x, origins = _validation_batch(model, arrays, limit=16)
    device = x.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(x, origins, bank, residual_weight=weights)
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        model(x, origins, bank, residual_weight=weights)
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if was_training:
        model.train()
    return {
        "split_used": "validation",
        "test_labels_used": False,
        "device": str(device),
        "batch_size": int(len(origins)),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(repeats),
        "synchronized": device.type == "cuda",
        "median_batch_ms": float(np.median(samples)),
        "p95_batch_ms": float(np.quantile(samples, 0.95)),
        "mean_batch_ms": float(np.mean(samples)),
    }
