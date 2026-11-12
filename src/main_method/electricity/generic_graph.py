"""Deterministic graph-to-hypergraph conversion for the unified runner."""

import csv
import json
from pathlib import Path

import torch


def load_graph(path: Path, num_nodes: int, max_edges: int = 512, max_edge_size: int = 32):
    if path.suffix=='.json':
        payload=json.loads(path.read_text())
        assert payload['variable_names']==['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'] and num_nodes==7
        edges=payload['edges']; incidence=torch.zeros(num_nodes,len(edges),dtype=torch.float32)
        for k,e in enumerate(edges):
            assert 2<=len(set(e))==len(e)<=num_nodes and all(0<=n<num_nodes for n in e)
            incidence[e,k]=1
        return dict(incidence=incidence,pool=incidence/incidence.sum(1,keepdim=True).clamp_min(1),edge_pool=incidence/incidence.sum(0,keepdim=True),edges=edges,num_nodes=num_nodes)
    pairs = set()
    neighbors = [set() for _ in range(num_nodes)]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if header is None:
            raise ValueError(f"empty edge file: {path}")
        for row in rows:
            if len(row) < 2:
                continue
            try:
                u, v = int(float(row[0])), int(float(row[1]))
            except ValueError:
                continue
            if not (0 <= u < num_nodes and 0 <= v < num_nodes) or u == v:
                continue
            a, b = sorted((u, v))
            pairs.add((a, b))
            neighbors[a].add(b)
            neighbors[b].add(a)

    edges = [list(pair) for pair in sorted(pairs)]
    for node, values in enumerate(neighbors):
        if values:
            members = [node] + sorted(values)
            edges.append(members[:max_edge_size])

    unique = []
    seen = set()
    for members in edges:
        key = tuple(sorted(set(members)))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            unique.append(list(key))
    unique.sort(key=lambda item: (len(item), tuple(item)))
    unique = unique[:max_edges]

    incidence = torch.zeros(num_nodes, len(unique), dtype=torch.float32)
    for edge_id, members in enumerate(unique):
        incidence[members, edge_id] = 1.0
    degree = incidence.sum(1).clamp_min(1.0)
    edge_size = incidence.sum(0).clamp_min(1.0)
    return {
        "incidence": incidence,
        "pool": incidence / degree[:, None],
        "edge_pool": incidence / edge_size[None, :],
        "edges": unique,
        "num_nodes": num_nodes,
    }
