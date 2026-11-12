"""Physical hyperedge memory and utility-supervised forecasting, independent of HUR.

All public inputs/outputs use original flow units. Memory keys describe observed
local histories. Memory values and utility targets come from rolling out-of-fold
teachers. A binary edge mask preserves every retained candidate's membership.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def physical_candidates(csv_path, nodes, budget=256, max_size=16):
    """Use undirected physical connectivity; CSV row order is not road direction."""
    adjacency = [dict() for _ in range(nodes)]
    with Path(csv_path).open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            i, j = int(row["from"]), int(row["to"])
            if not 0 <= i < nodes or not 0 <= j < nodes:
                raise ValueError("CSV node indices are not aligned with NPZ variables")
            if i == j:
                continue
            cost = float(row["cost"])
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("physical edge cost must be finite and positive")
            adjacency[i][j] = min(cost, adjacency[i].get(j, float("inf")))
            adjacency[j][i] = min(cost, adjacency[j].get(i, float("inf")))
    candidates = {}

    def add(members, family, anchor):
        key = tuple(sorted(set(members)))
        if not 2 <= len(key) <= max_size:
            return
        entry = candidates.setdefault(key, {"members": list(key), "families": set(), "anchors": set()})
        entry["families"].add(family)
        entry["anchors"].add(anchor)

    for anchor in range(nodes):
        neighbors = sorted(adjacency[anchor], key=lambda n: (adjacency[anchor][n], n))
        add([anchor] + neighbors[:max_size-1], "local_neighborhood", anchor)
        for n in neighbors:
            add([anchor, n], "adjacent_pair", anchor)
        # Weighted shortest paths stay connected and require no assumed compass
        # direction, downstream label, or undocumented distance/speed units.
        queue = [(0.0, (anchor,))]
        best = {anchor: 0.0}
        reached = []
        while queue:
            cost, path = heapq.heappop(queue)
            last = path[-1]
            if cost > best[last]:
                continue
            if last != anchor:
                reached.append(last)
            if 3 <= len(path) <= 5:
                add(path, "connected_path", anchor)
            if len(path) >= 5:
                continue
            for n, step_cost in adjacency[last].items():
                new_cost = cost + step_cost
                if n not in path and new_cost < best.get(n, float("inf")):
                    best[n] = new_cost
                    heapq.heappush(queue, (new_cost, path + (n,)))
        add([anchor] + reached[:max_size-1], "bounded_neighborhood", anchor)
    entries = sorted(candidates.values(), key=lambda e: (min(e["anchors"]), len(e["members"]), e["members"]))
    if not entries:
        raise ValueError("physical topology produced no candidates")
    # Coverage then stratified round-robin; no correlation-derived memberships.
    selected, covered = [], set()
    available = list(range(len(entries)))
    while len(selected) < budget:
        gains = [(len(set(entries[i]["members"]) - covered), i) for i in available]
        gain, idx = max(gains, default=(0, -1), key=lambda x: (x[0], -x[1]))
        if gain == 0:
            break
        selected.append(idx)
        covered.update(entries[idx]["members"])
        available.remove(idx)
    families = sorted({f for e in entries for f in e["families"]})
    queues = {f: deque(i for i in available if f in entries[i]["families"]) for f in families}
    chosen = set(selected)
    while len(selected) < min(budget, len(entries)):
        progressed = False
        for family in families:
            q = queues[family]
            while q and q[0] in chosen:
                q.popleft()
            if q and len(selected) < budget:
                idx = q.popleft()
                selected.append(idx)
                chosen.add(idx)
                progressed = True
        if not progressed:
            break
    result = [entries[i] for i in selected]
    return {
        "num_nodes": nodes, "edges": [e["members"] for e in result],
        "edge_metadata": [{"families": sorted(e["families"]), "anchor": min(e["anchors"])} for e in result],
        "families": families, "source_sha256": file_hash(csv_path),
        "direction_status": "unverified; use undirected connectivity",
        "candidate_count_before_budget": len(entries),
        "coverage": len(set(n for e in result for n in e["members"])),
    }


def graph_tensors(graph):
    n, e = graph["num_nodes"], len(graph["edges"])
    h = torch.zeros(n, e)
    family = torch.zeros(e, len(graph["families"]))
    anchors = []
    for i, (members, meta) in enumerate(zip(graph["edges"], graph["edge_metadata"])):
        h[members, i] = 1
        anchors.append(meta["anchor"])
        for name in meta["families"]:
            family[i, graph["families"].index(name)] = 1
    family /= family.sum(-1, keepdim=True).clamp_min(1)
    return h, family, torch.tensor(anchors, dtype=torch.long)


def periodic_prior(raw_prefix, mean, std, day=288, week=2016):
    z = (np.asarray(raw_prefix, dtype=np.float32) - mean) / std
    def phase_average(period):
        total = np.zeros((period, z.shape[1]), dtype=np.float64)
        count = np.zeros(period, dtype=np.float64)
        phase = np.arange(len(z)) % period
        np.add.at(total, phase, z)
        np.add.at(count, phase, 1)
        return total / np.maximum(count[:, None], 1), count
    daily, _ = phase_average(day)
    weekly, count = phase_average(week)
    # Shrink poorly observed week phases. Absolute day-of-week labels are not
    # claimed because this NPZ lacks a timestamp/calendar mapping.
    deviation = (weekly - daily[np.arange(week) % day]) * (count / (count + 8))[:, None]
    return torch.tensor(daily, dtype=torch.float32), torch.tensor(deviation, dtype=torch.float32)


def lead_profiles(tensor, bins):
    return torch.stack([part.mean(dim=1) for part in torch.tensor_split(tensor, bins, dim=1)], dim=1)


def interpolate_profiles(profile, horizon):
    return F.interpolate(profile.transpose(1, 2), size=horizon, mode="linear", align_corners=True).transpose(1, 2)


def local_keys(raw, origins, pool):
    """[B,S,N] -> [B,E,13], in observed raw-flow units with ordered history."""
    width = min(48, raw.size(1))
    chunks = torch.tensor_split(raw[:, -width:], 8, dim=1)
    trajectory = torch.stack([c.mean(1) for c in chunks], dim=1)
    pooled = torch.einsum("bqn,ne->beq", trajectory, pool)
    last = raw[:, -6:].mean(1)
    spatial_var = torch.einsum("bn,ne->be", last.square(), pool) - torch.einsum("bn,ne->be", last, pool).square()
    spread = (spatial_var.clamp_min(0) + 1e-5).sqrt()
    vol = torch.einsum("bn,ne->be", raw[:, -min(24,raw.size(1)):].std(1, unbiased=False), pool)
    trend = pooled[:, :, -1] - pooled[:, :, -2]
    phase = origins.float() % 288
    sin = torch.sin(2*math.pi*phase/288)[:, None].expand(-1, pool.size(1))
    cos = torch.cos(2*math.pi*phase/288)[:, None].expand_as(sin)
    return torch.cat([pooled, spread.unsqueeze(-1), vol.unsqueeze(-1), trend.unsqueeze(-1),
                      sin.unsqueeze(-1), cos.unsqueeze(-1)], dim=-1)


class EpisodicMemory:
    def __init__(self, payload, device, edges, bins):
        self.origins = payload["origins"].to(device)
        self.keys = payload["keys"].to(device)
        self.values = payload["values"].to(device)
        self.bins = bins
        if self.keys.size(0):
            self.mean = self.keys.mean(0, keepdim=True)
            self.std = self.keys.std(0, unbiased=False, keepdim=True).clamp_min(1e-3)
            self.keys_z = (self.keys - self.mean) / self.std
        else:
            self.mean = torch.zeros(1, edges, 13, device=device)
            self.std = torch.ones_like(self.mean)
            self.keys_z = self.keys

    def retrieve(self, query, origins, metric, seq_len, horizon, topk=8, temperature=0.5):
        b, e, _ = query.shape
        if not self.origins.numel():
            zero = query.new_zeros(b, e, self.bins)
            return zero, zero, query.new_zeros(b,e,1), query.new_zeros(b,e,1)
        q = (query - self.mean) / self.std
        w = F.softplus(metric) + 0.01
        w = w / w.sum(-1, keepdim=True)
        distance = ((q.square()*w).sum(-1).unsqueeze(-1)
                    + (self.keys_z.square()*w).sum(-1).transpose(0,1).unsqueeze(0)
                    - 2*torch.einsum("bef,mef,ef->bem", q, self.keys_z, w)).clamp_min(0)
        # A memory's complete history+target window must finish before the
        # current query history starts. This also excludes the query itself.
        allowed = self.origins[None,:] + horizon <= origins[:,None] - seq_len
        distance = distance.masked_fill(~allowed[:,None,:], float("inf"))
        d, ids = distance.topk(min(topk, self.origins.numel()), largest=False, dim=-1)
        valid = torch.isfinite(d)
        logits = -d.masked_fill(~valid, 1e6)/temperature
        weights = torch.softmax(logits, -1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        edge_id = torch.arange(e, device=query.device)[None,:,None].expand_as(ids)
        values = self.values[ids, edge_id]
        result = (weights.unsqueeze(-1)*values).sum(-2)
        variance = (weights.unsqueeze(-1)*(values-result.unsqueeze(-2)).square()).sum(-2)
        support = valid.any(-1, keepdim=True).to(query.dtype)
        nearest = d[:, :, :1].masked_fill(~support.bool(), 0)
        return result, variance, nearest, support


def empty_bank(edges, bins):
    return {"origins": torch.empty(0,dtype=torch.long), "keys": torch.empty(0,edges,13),
            "values": torch.empty(0,edges,bins), "utility": torch.empty(0,edges,bins),
            "utility_known": torch.empty(0,edges,bins,dtype=torch.bool)}


class PhysicalCFMemory(nn.Module):
    def __init__(self, graph, mean, std, daily, weekly, seq_len=96, horizon=96,
                 hidden=96, bins=8, topk=8, temperature=0.5):
        super().__init__()
        self.seq_len, self.horizon, self.bins = seq_len, horizon, min(bins,horizon)
        self.topk, self.temperature = topk, temperature
        h, family, anchors = graph_tensors(graph)
        self.register_buffer("incidence", h)
        self.register_buffer("pool", h/h.sum(0,keepdim=True))
        self.register_buffer("family_membership", family)
        self.register_buffer("anchors", anchors)
        self.register_buffer("norm_mean", torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("norm_std", torch.as_tensor(std,dtype=torch.float32))
        self.register_buffer("daily_reference", daily.clone())
        self.register_buffer("weekly_reference", weekly.clone())
        self.daily = nn.Parameter(daily.clone())
        self.weekly = nn.Parameter(weekly.clone())
        self.temporal_encoder = nn.Sequential(nn.Linear(seq_len,hidden),nn.GELU(),
                                               nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.temporal_head = nn.Linear(hidden,horizon)
        self.family_embedding = nn.Parameter(torch.randn(family.size(1),hidden)*0.02)
        self.size_embedding = nn.Embedding(33,hidden)
        self.edge_encoder = nn.Sequential(nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),
                                          nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.edge_head = nn.Linear(hidden,self.bins)
        nn.init.zeros_(self.edge_head.weight)
        nn.init.zeros_(self.edge_head.bias)
        self.retrieval_gate = nn.Linear(hidden,self.bins)
        nn.init.zeros_(self.retrieval_gate.weight)
        nn.init.zeros_(self.retrieval_gate.bias)
        self.node_gate = nn.Linear(hidden,self.bins)
        nn.init.zeros_(self.node_gate.weight)
        nn.init.zeros_(self.node_gate.bias)
        self.metric = nn.Parameter(torch.zeros(family.size(1),13))
        self.lead_embedding = nn.Embedding(self.bins,hidden)
        self.utility_net = nn.Sequential(nn.LayerNorm(hidden+4),nn.Linear(hidden+4,hidden),
                                        nn.GELU(),nn.Linear(hidden,1))
        self.threshold = nn.Parameter(torch.zeros(self.bins))
        self.mask_temperature = 0.2

    def temporal_features(self, raw, origins):
        z = (raw-self.norm_mean)/self.norm_std
        history_ids = origins[:,None] + torch.arange(-self.seq_len,0,device=raw.device)[None,:]
        future_ids = origins[:,None] + torch.arange(self.horizon,device=raw.device)[None,:]
        history_period = self.daily[history_ids%288] + self.weekly[history_ids%2016]
        future_period = self.daily[future_ids%288] + self.weekly[future_ids%2016]
        residual = z-history_period
        center = residual.mean(1,keepdim=True)
        scale = (residual.var(1,unbiased=False,keepdim=True)+1e-4).sqrt()
        encoded = self.temporal_encoder(((residual-center)/scale).transpose(1,2))
        base_z = future_period + self.temporal_head(encoded).transpose(1,2)*scale + center
        return base_z*self.norm_std+self.norm_mean, encoded

    def features(self, raw, origins, memory, no_memory=False):
        base, nodes = self.temporal_features(raw,origins)
        pooled = torch.einsum("bnd,ne->bed",nodes,self.pool)
        variance = (torch.einsum("bnd,ne->bed",nodes.square(),self.pool)-pooled.square()).clamp_min(0)
        metadata = self.family_membership@self.family_embedding
        metadata = metadata + self.size_embedding(self.incidence.sum(0).long().clamp_max(32))
        edge = self.edge_encoder(torch.cat([pooled+metadata,nodes[:,self.anchors],variance],-1))
        keys = local_keys(raw,origins,self.pool)
        ret,var,dist,support = memory.retrieve(keys,origins,self.family_membership@self.metric,
                                             self.seq_len,self.horizon,self.topk,self.temperature)
        if no_memory:
            ret,var,support = torch.zeros_like(ret),torch.zeros_like(var),torch.zeros_like(support)
        edge_scale = (raw.var(1,unbiased=False)+1e-4).sqrt()@self.pool
        contribution = self.edge_head(edge)*edge_scale.unsqueeze(-1)
        contribution = contribution + torch.sigmoid(self.retrieval_gate(edge))*ret
        denominator=edge_scale.unsqueeze(-1).clamp_min(1)
        aux = torch.stack([ret/denominator,(var+1e-6).sqrt()/denominator,
                           torch.log1p(dist).expand_as(ret),support.expand_as(ret)],-1)
        lead_features=edge.unsqueeze(2)+self.lead_embedding.weight[None,None,:,:]
        utility = torch.tanh(self.utility_net(torch.cat([lead_features,aux],-1)).squeeze(-1))
        soft = torch.sigmoid((utility-self.threshold)/self.mask_temperature)
        hard = (soft>=0.5).to(soft.dtype)
        mask = hard+soft-soft.detach() if self.training else hard
        return {"base":base,"edge_contribution":contribution,"node_gain":2*torch.sigmoid(self.node_gate(nodes)),
                "utility":utility,"soft_mask":soft,"mask":mask,"retrieval_support":support}

    def compose(self, cache, mask):
        # Recompute degrees for the actual retained edges; do not preserve full
        # graph denominators, which confound deletion with shrinking amplitude.
        numerator = torch.einsum("ne,beq->bqn",self.incidence,mask*cache["edge_contribution"])
        degree = torch.einsum("ne,beq->bqn",self.incidence,mask)
        correction = numerator/degree.clamp_min(1)
        correction = correction*cache["node_gain"].transpose(1,2)
        return cache["base"] + interpolate_profiles(correction,self.horizon)

    def forward(self, raw, origins, memory, mask=None, no_memory=False):
        cache = self.features(raw,origins,memory,no_memory)
        return self.compose(cache,cache["mask"] if mask is None else mask),cache

    def prior_penalty(self):
        return (self.daily-self.daily_reference).square().mean() + (self.weekly-self.weekly_reference).square().mean()


@torch.no_grad()
def paired_utilities(model, cache, target, sampled_edges, contexts=2):
    """Frozen-model on/off marginal utility, local nodes and forecast lead bins."""
    b,e,q = cache["edge_contribution"].shape
    labels = target.new_zeros(b,e,q)
    known = torch.zeros_like(labels,dtype=torch.bool)
    for _ in range(contexts):
        coalition = (torch.rand(b,e,q,device=target.device)<0.5).to(target.dtype)
        for edge_id in sampled_edges:
            off,on = coalition.clone(),coalition.clone()
            off[:,edge_id] = 0
            on[:,edge_id] = 1
            error_off = ((model.compose(cache,off)-target)/model.norm_std).square()
            error_on = ((model.compose(cache,on)-target)/model.norm_std).square()
            members = model.incidence[:,edge_id].bool()
            loss_off = lead_profiles(error_off[:,:,members],q).mean(-1)
            loss_on = lead_profiles(error_on[:,:,members],q).mean(-1)
            utility = (loss_off-loss_on)/(loss_off+loss_on+1e-6)
            labels[:,edge_id] += utility/contexts
            known[:,edge_id] = True
    return labels,known
