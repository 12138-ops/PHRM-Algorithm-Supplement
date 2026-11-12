"""Deterministic physical candidates for LargeST-SD's ordered road graph.

There is one bounded road ego candidate for every sensor.  When metadata
permits, a second candidate follows a connected road neighborhood restricted
to the anchor's highway and direction.  Affinities choose physical neighbors;
the sensor's integer position is never interpreted as physical distance.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


GRAPH_VERSION = "largest-road-ego-and-connected-corridor-v1"


def _column(records, names, required=False):
    lookup = {str(k).strip().lower(): k for k in records[0]}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise ValueError(f"metadata requires one of these columns: {names}")
    return None


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _source_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def build_candidates(adjacency, records, *, expected_sensor_ids=None, max_edge_size=32, max_edges=1500):
    """Return edge sets/provenance without importing a tensor runtime.

    Rows in ``records`` must already have the exact same order as the data and
    adjacency.  ``expected_sensor_ids`` makes that correspondence an executable
    check when the data preparer supplies its ordered sensor IDs.
    """
    affinity = np.asarray(adjacency)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("road affinity must be a square matrix")
    nodes = affinity.shape[0]
    if nodes < 1 or len(records) != nodes:
        raise ValueError("metadata rows must match road matrix size")
    if not np.isfinite(affinity).all() or np.any(affinity < 0):
        raise ValueError("road affinity must be finite and nonnegative")
    if max_edge_size < 2 or max_edges < 1:
        raise ValueError("invalid graph budget")
    id_key = _column(records, ("ID", "sensor_id", "station_id"), required=True)
    ids = [str(row[id_key]).strip() for row in records]
    if any(not value for value in ids) or len(set(ids)) != nodes:
        raise ValueError("metadata sensor IDs must be nonempty and unique")
    if expected_sensor_ids is not None:
        expected = [str(value).strip() for value in expected_sensor_ids]
        if ids != expected:
            raise ValueError("metadata sensor order differs from data/adjacency sensor order")
    highway_key = _column(records, ("Fwy", "highway", "freeway"))
    direction_key = _column(records, ("Direction", "direction"))
    groups = [
        (str(row.get(highway_key, "")).strip(), str(row.get(direction_key, "")).strip().upper())
        for row in records
    ]
    # The official matrix contains road-distance affinities: greater means
    # nearer/stronger.  Ignore diagonal identity links when finding neighbors.
    affinity = affinity.astype(np.float64, copy=True)
    np.fill_diagonal(affinity, 0.0)
    undirected = np.maximum(affinity, affinity.T)
    neighbors = [np.flatnonzero(undirected[node] > 0).tolist() for node in range(nodes)]

    def physical_order(anchor, candidates):
        # IDs break equal-affinity ties reproducibly; ties carry no distance
        # information and never create a link absent from the road matrix.
        return sorted(candidates, key=lambda node: (-float(undirected[anchor, node]), ids[node]))

    candidates = {}
    mandatory = []
    isolated = []

    def add(members, provenance):
        edge = tuple(sorted(set(members)))
        candidates.setdefault(edge, []).append(provenance)
        return edge

    for anchor in range(nodes):
        chosen = physical_order(anchor, neighbors[anchor])[: max_edge_size - 1]
        if not chosen:
            isolated.append(anchor)
            edge = add([anchor], dict(family="isolated_identity", anchor=anchor,
                                     sensor_id=ids[anchor], physical_relation=False,
                                     reason="no off-diagonal road affinity; identity only"))
        else:
            edge = add([anchor] + chosen, dict(
                family="road_ego", anchor=anchor, sensor_id=ids[anchor],
                physical_relation=True, direction="incoming_and_outgoing_union",
                outgoing_members=[node for node in chosen if affinity[anchor, node] > 0],
                incoming_members=[node for node in chosen if affinity[node, anchor] > 0],
                neighborhood_capped=len(neighbors[anchor]) > max_edge_size - 1))
        if edge not in mandatory:
            mandatory.append(edge)

    for anchor in range(nodes):
        highway, direction = groups[anchor]
        if not highway or not direction or anchor in isolated:
            continue
        # Bounded breadth-first expansion preserves a connected road-induced
        # set.  Within a frontier, prefer the strongest actual connecting link.
        selected = [anchor]
        seen = {anchor}
        frontier = {anchor}
        while frontier and len(selected) < max_edge_size:
            links = {}
            for parent in frontier:
                for node in neighbors[parent]:
                    if node not in seen and groups[node] == (highway, direction):
                        links[node] = max(links.get(node, 0.0), float(undirected[parent, node]))
            next_frontier = sorted(links, key=lambda node: (-links[node], ids[node]))
            next_frontier = next_frontier[: max_edge_size - len(selected)]
            selected.extend(next_frontier)
            seen.update(next_frontier)
            frontier = set(next_frontier)
        if len(selected) >= 2:
            add(selected, dict(family="same_highway_direction_connected", anchor=anchor,
                               sensor_id=ids[anchor], physical_relation=True,
                               highway=highway, direction=direction,
                               construction="bounded road-connected breadth-first expansion"))

    # Mandatory ego sets cover every anchor before any optional budget is
    # applied.  Never sort all pair edges ahead of higher-order candidates.
    if len(mandatory) > max_edges:
        raise ValueError(f"edge budget {max_edges} cannot retain {len(mandatory)} anchor candidates")
    selected = list(mandatory)
    selected_set = set(selected)
    optional = sorted((edge for edge in candidates if edge not in selected_set),
                      key=lambda edge: (-len(edge), tuple(ids[node] for node in edge)))
    selected.extend(optional[: max_edges - len(selected)])
    selected = sorted(selected, key=lambda edge: (len(edge), edge))
    covered = set().union(*map(set, selected))
    if covered != set(range(nodes)):
        raise AssertionError("candidate budget lost node coverage")
    provenance = [candidates[edge] for edge in selected]
    families = Counter(item["family"] for entries in provenance for item in entries)
    metadata = dict(
        version=GRAPH_VERSION, num_nodes=nodes, selected_edges=len(selected),
        candidate_edges=len(candidates), selected_higher_order_edges=sum(len(edge) > 2 for edge in selected),
        singleton_identity_edges=sum(len(edge) == 1 for edge in selected),
        isolated_node_indices=isolated, isolated_sensor_ids=[ids[node] for node in isolated],
        covered_variables=len(covered), min_edge_size=min(map(len, selected)),
        max_selected_edge_size=max(map(len, selected)), max_edge_size=max_edge_size,
        edge_budget=max_edges, mandatory_anchor_edges=len(mandatory),
        optional_edges_truncated=max(0, len(candidates) - len(selected)),
        family_counts=dict(families), sensor_ids=ids, sensor_order_sha256=_hash_json(ids),
        expected_sensor_order_checked=expected_sensor_ids is not None,
        edges_sha256=_hash_json(selected),
        affinity_semantics="official road-distance-derived affinity; larger is stronger",
        source_directions_preserved_in_provenance=True,
        isolation_policy="explicit identity singleton; no fabricated physical neighbors",
    )
    return [list(edge) for edge in selected], provenance, metadata


def load_graph(adjacency_path, metadata_path, num_nodes=716, *, expected_sensor_ids=None,
               max_edge_size=32, max_edges=1500):
    """Load an ordered official SD matrix/metadata pair for the existing KV."""
    import torch

    adjacency_path, metadata_path = Path(adjacency_path), Path(metadata_path)
    adjacency = np.load(adjacency_path, allow_pickle=False)
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("road matrix does not match configured number of sensors")
    edges, provenance, metadata = build_candidates(
        adjacency, records, expected_sensor_ids=expected_sensor_ids,
        max_edge_size=max_edge_size, max_edges=max_edges)
    incidence = torch.zeros(num_nodes, len(edges), dtype=torch.float32)
    for edge_index, members in enumerate(edges):
        incidence[members, edge_index] = 1.0
    degree = incidence.sum(1, keepdim=True)
    if not bool((degree > 0).all()):
        raise AssertionError("incidence graph lost node coverage")
    metadata.update(adjacency_source=str(adjacency_path), metadata_source=str(metadata_path),
                    adjacency_source_sha256=_source_hash(adjacency_path),
                    metadata_source_sha256=_source_hash(metadata_path),
                    min_node_degree=float(degree.min()))
    metadata["fingerprint"] = _hash_json(metadata)
    return dict(incidence=incidence, pool=incidence / degree,
                edge_pool=incidence / incidence.sum(0, keepdim=True), edges=edges,
                edge_metadata=provenance, num_nodes=num_nodes, metadata=metadata)
