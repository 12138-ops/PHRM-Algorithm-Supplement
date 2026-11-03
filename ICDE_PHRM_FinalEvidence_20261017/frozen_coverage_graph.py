"""Build physical candidates with a deterministic budget that preserves coverage."""
import csv
import hashlib
import json
from pathlib import Path

import torch

GRAPH_VERSION='physical-coverage-before-budget-v1'


def select_edges(candidates, num_nodes, max_edges):
    candidates=sorted(set(tuple(sorted(set(e))) for e in candidates),key=lambda e:(len(e),e))
    if not candidates or any(len(e)<2 for e in candidates):
        raise ValueError('physical candidates must contain at least two distinct nodes')
    universe=set(range(num_nodes));available=set().union(*map(set,candidates))
    if available!=universe:
        raise ValueError('road candidates do not cover all nodes; missing='+str(sorted(universe-available)))
    if max_edges<1:raise ValueError('edge budget must be positive')
    if len(candidates)<=max_edges:return candidates
    high=[e for e in candidates if len(e)>2]
    legacy=candidates[:max_edges]
    # Preserve prior results whenever the old subset already satisfies both invariants.
    if set().union(*map(set,legacy))==universe and (not high or any(len(e)>2 for e in legacy)):
        return legacy
    pairs=[e for e in candidates if len(e)==2]
    selected=[];seen=set();uncovered=set(universe)
    # Cover nodes with genuine higher-order neighborhoods first. Pair-only components
    # remain eligible when no higher-order candidate can reach an uncovered node.
    while uncovered and len(selected)<max_edges:
        pool=[e for e in high if e not in seen and uncovered.intersection(e)]
        if not pool:pool=[e for e in candidates if e not in seen and uncovered.intersection(e)]
        if not pool:break
        edge=min(pool,key=lambda e:(-len(uncovered.intersection(e)),len(e),e))
        selected.append(edge);seen.add(edge);uncovered.difference_update(edge)
    if uncovered:
        raise ValueError(f'budget={max_edges} cannot preserve this coverage; missing={sorted(uncovered)}')
    # Fill remaining capacity from both families rather than sorting all pairs first.
    high_remaining=iter(e for e in high if e not in seen)
    pair_remaining=iter(e for e in pairs if e not in seen)
    pools=[high_remaining,pair_remaining]
    while len(selected)<max_edges and pools:
        keep=[]
        for pool in pools:
            if len(selected)>=max_edges:break
            edge=next(pool,None)
            if edge is not None:selected.append(edge);keep.append(pool)
        pools=keep
    return sorted(selected,key=lambda e:(len(e),e))


def load_graph(path: Path,num_nodes: int,max_edges: int=512,max_edge_size: int=32):
    path=Path(path)
    if max_edge_size<2:raise ValueError('max_edge_size must be at least two')
    neighbors=[set() for _ in range(num_nodes)];pairs=set();raw_rows=0
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f);header=next(reader,None)
        if not header or header[:2]!=['from','to']:raise ValueError('expected indexed from/to road CSV')
        for row in reader:
            if not row:continue
            if len(row)<2:raise ValueError('incomplete road row')
            u,v=map(int,row[:2]);raw_rows+=1
            if not (0<=u<num_nodes and 0<=v<num_nodes):raise ValueError(f'road endpoint outside variable indices: {(u,v)}')
            if u==v:continue
            edge=tuple(sorted((u,v)));pairs.add(edge);neighbors[u].add(v);neighbors[v].add(u)
    candidates=set(pairs)
    for node,values in enumerate(neighbors):
        if values:candidates.add(tuple(sorted(([node]+sorted(values))[:max_edge_size])))
    chosen=select_edges(candidates,num_nodes,max_edges)
    incidence=torch.zeros(num_nodes,len(chosen),dtype=torch.float32)
    for edge_id,members in enumerate(chosen):incidence[list(members),edge_id]=1
    assert (incidence.sum(1)>0).all()
    if any(len(e)>2 for e in candidates):assert any(len(e)>2 for e in chosen)
    edge_hash=hashlib.sha256(json.dumps(chosen,separators=(',',':')).encode()).hexdigest()
    metadata=dict(version=GRAPH_VERSION,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),num_nodes=num_nodes,raw_rows=raw_rows,candidate_edges=len(candidates),candidate_higher_order_edges=sum(len(e)>2 for e in candidates),edge_budget=max_edges,max_edge_size=max_edge_size,selected_edges=len(chosen),selected_higher_order_edges=sum(len(e)>2 for e in chosen),covered_variables=int((incidence.sum(1)>0).sum()),min_node_degree=float(incidence.sum(1).min()),min_edge_size=min(map(len,chosen)),max_selected_edge_size=max(map(len,chosen)),edges_sha256=edge_hash)
    metadata['fingerprint']=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    return dict(incidence=incidence,pool=incidence/incidence.sum(1,keepdim=True),edge_pool=incidence/incidence.sum(0,keepdim=True),edges=[list(e) for e in chosen],num_nodes=num_nodes,metadata=metadata)
