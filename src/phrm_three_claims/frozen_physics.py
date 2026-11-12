"""Physical hypergraph residual Key--Value adapter for a frozen TQNet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


KEY_DIM = 13


def load_graph(path: Path, num_nodes: int) -> Dict[str, torch.Tensor]:
    """Load the physical hyperedges generated from the PEMS08 road topology."""
    data = json.loads(path.read_text(encoding="utf-8"))
    edges = []
    for edge in data.get("edges", []):
        members = sorted({int(x) for x in edge if 0 <= int(x) < num_nodes})
        if len(members) >= 2:
            edges.append(members)
    # Remove duplicate sets while retaining the JSON order and its physical provenance.
    seen = set()
    unique = []
    for edge in edges:
        key = tuple(edge)
        if key not in seen:
            unique.append(edge)
            seen.add(key)
    incidence = torch.zeros(num_nodes, len(unique), dtype=torch.float32)
    for e, members in enumerate(unique):
        incidence[members, e] = 1.0
    degree = incidence.sum(1).clamp_min(1.0)
    pool = incidence / degree[:, None]
    edge_pool = incidence / incidence.sum(0, keepdim=True).clamp_min(1.0)
    return {
        "incidence": incidence,
        "pool": pool,
        "edge_pool": edge_pool,
        "edges": unique,
        "num_nodes": torch.tensor(num_nodes),
    }


def periodic_prior(train_norm: np.ndarray, daily_period: int = 288, weekly_period: int = 2016):
    """Estimate daily and weekly phase means using training rows only."""
    x = torch.as_tensor(train_norm, dtype=torch.float32)
    n, nodes = x.shape
    daily = torch.zeros(daily_period, nodes)
    for phase in range(daily_period):
        rows = x[phase:n:daily_period]
        daily[phase] = rows.mean(0) if rows.numel() else 0.0
    residual = x - daily[torch.arange(n) % daily_period]
    weekly = torch.zeros(weekly_period, nodes)
    for phase in range(weekly_period):
        rows = residual[phase:n:weekly_period]
        weekly[phase] = rows.mean(0) if rows.numel() else 0.0
    return daily, weekly


def _phase_value(daily, weekly, origins, steps):
    ids = origins[:, None] + steps[None, :]
    return daily[ids.remainder(daily.shape[0])] + weekly[ids.remainder(weekly.shape[0])]


def residual_history(x, origins, daily, weekly):
    steps = torch.arange(-x.shape[1], 0, device=x.device)
    phase = _phase_value(daily, weekly, origins, steps)
    return x - phase


def residual_future(y, origins, daily, weekly):
    steps = torch.arange(y.shape[1], device=y.device)
    phase = _phase_value(daily, weekly, origins, steps)
    return y - phase


def edge_descriptors(residual, origins, edge_pool, period=288):
    """Return 13 interpretable descriptors for every physical hyperedge."""
    # residual [B, S, N] -> edge signal [B, S, E]
    series = torch.einsum("bsn,ne->bse", residual, edge_pool)
    s = series.shape[1]
    last = series[:, -1]
    def mean_last(k):
        return series[:, -min(k, s):].mean(1)
    def std_last(k):
        return series[:, -min(k, s):].std(1, unbiased=False)

    m6, m12, m24 = mean_last(6), mean_last(12), mean_last(24)
    sd24 = std_last(24)
    slope24 = last - series[:, -min(24, s)]
    diff = series[:, 1:] - series[:, :-1]
    dlast = diff[:, -1] if diff.shape[1] else torch.zeros_like(last)
    dmean = diff[:, -min(12, diff.shape[1]):].mean(1) if diff.shape[1] else dlast
    dstd = diff[:, -min(24, diff.shape[1]):].std(1, unbiased=False) if diff.shape[1] else dlast
    abs_last = last.abs()
    abs_mean = series[:, -min(24, s):].abs().mean(1)
    phase = origins.to(series.dtype).remainder(float(period)) * (2.0 * np.pi / float(period))
    sin_phase, cos_phase = phase.sin()[:, None], phase.cos()[:, None]
    # [B, E, 13]
    return torch.cat(
        [
            torch.stack(
                [last, m6, m12, m24, sd24, slope24, dlast, dmean, dstd, abs_last, abs_mean],
                dim=-1,
            ),
            sin_phase.expand(-1, last.shape[1])[:, :, None],
            cos_phase.expand(-1, last.shape[1])[:, :, None],
        ],
        dim=-1,
    )


class PhysicalKVBank:
    """Train-only episode memory with chronology-aware retrieval."""

    def __init__(self, origins, keys, values, device, topk=8, temperature=0.5):
        self.origins = origins.to(device=device, dtype=torch.long)
        self.values = values.to(device=device, dtype=torch.float32)
        self.keys_raw = keys.to(device=device, dtype=torch.float32)
        self.mean = self.keys_raw.mean(0, keepdim=True)
        self.std = self.keys_raw.std(0, unbiased=False, keepdim=True).clamp_min(1e-3)
        self.keys = (self.keys_raw - self.mean) / self.std
        self.topk = int(topk)
        self.temperature = float(temperature)

    def retrieve(self, query, origins, seq_len):
        # query [B, E, D], keys [M, E, D]
        bsz, edges, _ = query.shape
        if self.origins.numel() == 0:
            z = query.new_zeros(bsz, edges, self.values.shape[-1])
            return z, z, query.new_zeros(bsz, edges), query.new_zeros(bsz, edges, 1)
        q = (query - self.mean) / self.std
        # A fixed metric keeps the first experiment deterministic and avoids test-label fitting.
        dist = (
            q.square().sum(-1).unsqueeze(-1)
            + self.keys.square().sum(-1).transpose(0, 1).unsqueeze(0)
            - 2.0 * torch.einsum("bed,med->bem", q, self.keys)
        ).clamp_min(0.0)
        allowed = self.origins[None, :] + self.values.shape[-1] <= origins[:, None] - seq_len
        dist = dist.masked_fill(~allowed[:, None, :], float("inf"))
        k = min(self.topk, self.origins.numel())
        nearest, ids = dist.topk(k, largest=False, dim=-1)
        valid = torch.isfinite(nearest)
        weights = torch.softmax(-nearest.masked_fill(~valid, 1e6) / self.temperature, dim=-1)
        weights = weights * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        # Gather values edge by edge: values_e has shape [E, M, H].
        values_e = self.values.permute(1, 0, 2)
        values_e = values_e.unsqueeze(0).expand(bsz, -1, -1, -1)
        gather_ids = ids.unsqueeze(-1).expand(-1, -1, -1, values_e.shape[-1])
        selected = torch.gather(values_e, 2, gather_ids)
        retrieved = (weights.unsqueeze(-1) * selected).sum(2)
        variance = (weights.unsqueeze(-1) * (selected - retrieved.unsqueeze(2)).square()).sum(2)
        support = valid.any(-1).to(query.dtype)
        nearest = nearest[:, :, :1].masked_fill(~support.bool().unsqueeze(-1), 0.0)
        return retrieved, variance, support, nearest


class TQNetPhysicsKV(nn.Module):
    """Frozen official TQNet plus a physical residual memory adapter."""

    def __init__(self, base, graph, daily, weekly, horizon, hidden=96):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.horizon = horizon
        self.register_buffer("incidence", graph["incidence"])
        self.register_buffer("edge_pool", graph["edge_pool"])
        self.register_buffer("daily", daily)
        self.register_buffer("weekly", weekly)
        # The final layer is initialized to zero: the adapter starts exactly at TQNet.
        self.gate = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.residual_scale = nn.Parameter(torch.zeros(horizon))

    def forward(
        self,
        x,
        origins,
        bank: PhysicalKVBank,
        disable_memory=False,
        residual_weight=None,
    ):
        """Return the base forecast plus a physical residual correction.

        ``residual_weight`` is an inference-time calibration hook.  It is
        intentionally left as ``None`` during training and for h48/h96, so
        those runs retain the original adapter exactly.  For h12/h24 the
        caller may pass a validation-selected vector of length ``horizon`` to
        down-weight individual forecast steps without using test labels.
        """
        cycle = origins.remainder(self.base.cycle_len).to(dtype=torch.long)
        with torch.no_grad():
            base = self.base(x, cycle)
        rh = residual_history(x, origins, self.daily, self.weekly)
        query = edge_descriptors(rh, origins, self.edge_pool, period=self.daily.shape[0])
        edge_memory, edge_var, support, nearest = bank.retrieve(query, origins, x.shape[1])
        # Incidence normalization injects one edge trajectory to every physical member.
        edge_weight = support / (1.0 + nearest.squeeze(-1))
        edge_weight = edge_weight / edge_weight.sum(-1, keepdim=True).clamp_min(1e-6)
        node_denom = torch.einsum("ne,be->bn", self.incidence, edge_weight).clamp_min(1e-6)
        # Use the retrieval confidence to reweight edges before node aggregation.
        weighted = torch.einsum("ne,beh,be->bhn", self.incidence, edge_memory, edge_weight)
        node_memory = weighted / node_denom[:, None, :]
        future_steps = torch.arange(self.horizon, device=x.device)
        periodic = _phase_value(self.daily, self.weekly, origins, future_steps)
        physical = periodic + node_memory
        diff = physical - base
        last = x[:, -1]
        features = torch.stack(
            [
                base.mean(1),
                base.std(1, unbiased=False),
                diff.mean(1),
                diff.std(1, unbiased=False),
                last,
                node_memory.mean(1),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(features)).transpose(1, 2)
        scale = torch.tanh(self.residual_scale)[None, :, None]
        if disable_memory:
            gate = gate * 0.0
        raw_residual = scale * gate * diff
        if residual_weight is None:
            weighted_residual = raw_residual
        else:
            weight = torch.as_tensor(
                residual_weight, dtype=raw_residual.dtype, device=raw_residual.device
            )
            if weight.ndim == 0:
                weight = weight.view(1, 1, 1)
            elif weight.ndim == 1:
                if weight.numel() != self.horizon:
                    raise ValueError(
                        f"residual_weight length {weight.numel()} != horizon {self.horizon}"
                    )
                weight = weight.view(1, self.horizon, 1)
            elif weight.ndim == 2:
                if weight.shape != (raw_residual.shape[0], self.horizon):
                    raise ValueError(
                        "2-D residual_weight must have shape "
                        f"[batch, horizon], got {tuple(weight.shape)}"
                    )
                weight = weight.unsqueeze(-1)
            else:
                raise ValueError("residual_weight must be scalar, [horizon], or [batch,horizon]")
            weighted_residual = raw_residual * weight
        prediction = base + weighted_residual
        diagnostics = {
            "base": base,
            "physical": physical,
            "diff": diff,
            "gate": gate,
            "support": support,
            "edge_var": edge_var,
            "nearest": nearest,
            "raw_residual": raw_residual,
            "residual": weighted_residual,
        }
        return prediction, diagnostics


def trainable_parameters(model: TQNetPhysicsKV):
    return [p for p in model.parameters() if p.requires_grad]
