"""Train-only memory sampling and analytic normalization of clock features.

The episode count is identical to ``train_origins[::stride]``.  Selection uses
timestamps only: balanced daily-phase quotas, then one origin per chronological
stratum within each phase.  Values, validation data and labels are never read.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


SAMPLING_VERSION = "phase-balanced-chronological-v1"
NORMALIZATION_VERSION = "analytic-clock-sin-cos-v1"
PHASE_FEATURE_START = 11
KEY_DIM = 13


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validated_origins(origins, *, horizon, train_end, seq_len):
    x = np.asarray(origins)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError("origins must be a one-dimensional integer array")
    x = x.astype(np.int64, copy=False)
    if not len(x):
        raise ValueError("training origins cannot be empty")
    if np.any(x[1:] <= x[:-1]):
        raise ValueError("origins must be strictly increasing without duplicates")
    h = _positive_integer(horizon, "horizon")
    end = _positive_integer(train_end, "train_end")
    length = _positive_integer(seq_len, "seq_len")
    # Subtraction avoids overflow in an unchecked origin + horizon expression.
    if x[0] < length or x[-1] > end - h:
        raise ValueError("origin history or forecast target lies outside the training range")
    return x


def _phase_order(period):
    """A fixed, widely spaced permutation; remainder quotas do not favor phase 0."""
    if period == 1:
        return np.zeros(1, dtype=np.int64)
    golden_target = period * (math.sqrt(5.0) - 1.0) / 2.0
    steps = (s for s in range(1, period) if math.gcd(s, period) == 1)
    step = min(steps, key=lambda s: (abs(s - golden_target), s))
    return (period // 2 + np.arange(period, dtype=np.int64) * step) % period


def select_memory_origins(train_origins, stride, daily_period, *, horizon, train_end, seq_len=96):
    """Return sorted, unique origins with the old stride sampler's exact budget.

    Every available daily phase is represented if the budget permits.  Counts
    differ by at most one unless a phase has too few eligible training origins.
    Quotas use a fixed dispersed phase order.  Within each phase, midpoint-like
    stratification spans its full training chronology, with dispersed offsets so
    phases do not all choose the same position in their temporal strata.
    """
    x = _validated_origins(train_origins, horizon=horizon, train_end=train_end, seq_len=seq_len)
    stride = _positive_integer(stride, "stride")
    period = _positive_integer(daily_period, "daily_period")
    budget = (len(x) + stride - 1) // stride
    if budget == len(x):
        return x.copy()
    phase = x % period
    capacities = np.bincount(phase, minlength=period)
    order = _phase_order(period)
    order = order[capacities[order] > 0]
    quotas = np.zeros(period, dtype=np.int64)
    remaining = budget
    while remaining:
        eligible = order[quotas[order] < capacities[order]]
        take = eligible[:remaining]
        if not len(take):
            raise AssertionError("memory budget exceeds available origins")
        quotas[take] += 1
        remaining -= len(take)
    picked = []
    golden_fraction = (math.sqrt(5.0) - 1.0) / 2.0
    for rank, p in enumerate(order):
        q = int(quotas[p])
        if not q:
            continue
        eligible = x[phase == p]
        # Offset in (0, 1): nonidentical chronological positions across phases.
        offset = ((rank + 0.5) * golden_fraction) % 1.0
        slots = np.floor((np.arange(q) + offset) * len(eligible) / q).astype(np.int64)
        picked.append(eligible[slots])
    result = np.sort(np.concatenate(picked))
    if len(result) != budget or np.any(result[1:] <= result[:-1]):
        raise AssertionError("memory sampling violated exact budget or uniqueness")
    return result


def sampling_metadata(train_origins, selected, daily_period, stride, *, horizon, train_end, seq_len=96):
    """Record enough timestamp-only evidence to audit sampling and chronology."""
    x = _validated_origins(train_origins, horizon=horizon, train_end=train_end, seq_len=seq_len)
    chosen = _validated_origins(selected, horizon=horizon, train_end=train_end, seq_len=seq_len)
    period = _positive_integer(daily_period, "daily_period")
    stride = _positive_integer(stride, "stride")
    if not np.isin(chosen, x).all() or len(chosen) != len(x[::stride]):
        raise ValueError("selected origins must be a training subset with the original stride budget")
    counts = np.bincount(chosen % period, minlength=period)
    available = np.bincount(x % period, minlength=period)
    # Explicit little-endian serialization makes this hash portable.
    origins_hash = hashlib.sha256(chosen.astype("<i8", copy=False).tobytes()).hexdigest()
    return {
        "sampling_version": SAMPLING_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "train_origins": int(len(x)),
        "memory_episodes": int(len(chosen)),
        "legacy_stride_budget": int(len(x[::stride])),
        "memory_stride_budget_only": stride,
        "daily_period": period,
        "available_phases": int(np.count_nonzero(available)),
        "covered_phases": int(np.count_nonzero(counts)),
        "phase_counts": counts.tolist(),
        "first_origin": int(chosen[0]),
        "last_origin": int(chosen[-1]),
        "last_target_exclusive": int(chosen[-1] + horizon),
        "train_end": int(train_end),
        "selected_origins_sha256": origins_hash,
        "selection_inputs": "training timestamps only; no values or labels",
        "retrieval_chronology": "memory_origin + horizon <= query_origin - seq_len",
    }


def normalize_phase_features(bank):
    """Keep fitted statistics for dimensions 0..10; analytically scale sin/cos.

    For a uniform daily clock, each phase feature has mean 0 and variance 1/2.
    Those constants define the metric even when memory phases are incomplete.
    This prevents a fitted near-zero standard deviation from exploding query
    distances.  No query or validation statistics are needed.
    """
    if bank.keys_raw.shape[-1] != KEY_DIM or bank.keys_raw.shape[0] == 0:
        raise ValueError("bank requires nonempty memory with 13 descriptor dimensions")
    bank.mean = bank.mean.clone()
    bank.std = bank.std.clone()
    bank.mean[..., PHASE_FEATURE_START:] = 0.0
    bank.std[..., PHASE_FEATURE_START:] = 1.0 / math.sqrt(2.0)
    bank.keys = (bank.keys_raw - bank.mean) / bank.std
    bank.normalization_metadata = {
        "version": NORMALIZATION_VERSION,
        "state_dimensions": [0, PHASE_FEATURE_START],
        "state_statistics": "training memory fitted; unchanged",
        "clock_dimensions": [PHASE_FEATURE_START, KEY_DIM],
        "clock_mean": 0.0,
        "clock_std": 1.0 / math.sqrt(2.0),
    }
    return bank


def build_phase_balanced_bank(origins, keys, values, device, topk=8, temperature=0.5):
    """Build the existing chronology-aware bank with corrected clock scaling."""
    from physics_kv import PhysicalKVBank

    bank = PhysicalKVBank(origins, keys, values, device, topk=topk, temperature=temperature)
    return normalize_phase_features(bank)
