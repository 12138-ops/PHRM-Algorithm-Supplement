"""Strict loader for the named static Weather candidate JSON schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from weather_graph import SCHEMA_VERSION, canonical_variable_name, validate_variable_names


def load_graph(path: str | Path, num_nodes: int, variable_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Load validated semantic candidates without dropping singleton identities.

    Supplying the actual data-column ``variable_names`` additionally verifies
    exact alignment with the model's node axis.  There is deliberately no CSV,
    sequential-neighbor, truncation, or missing-node fallback.
    """
    if Path(path).suffix.lower() != ".json":
        raise ValueError("Weather graph loader accepts named candidate JSON only")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("dataset") != "Weather":
        raise ValueError("Unsupported named Weather graph schema")
    names, code_index = validate_variable_names(payload.get("variable_names", []))
    if type(num_nodes) is not int or payload.get("num_variables") != num_nodes or len(names) != num_nodes:
        raise ValueError("Weather graph node count does not match the data")
    named_index = payload.get("variable_to_index")
    if (not isinstance(named_index, dict)
            or any(type(index) is not int for index in named_index.values())
            or named_index != {name: i for i, name in enumerate(names)}):
        raise ValueError("Graph variable-to-index mapping does not match variable_names order")
    if variable_names is not None and list(variable_names) != names:
        raise ValueError("Graph variable_names order differs from the data-column order")

    construction = payload.get("construction", {})
    required_false = ("uses_data_values", "uses_labels", "uses_column_order_adjacency",
                      "causal_effect_claimed", "exact_physical_law_enforced")
    if any(construction.get(flag) is not False for flag in required_false):
        raise ValueError("Candidate construction provenance is missing or incompatible")
    edges = payload.get("edges")
    metadata = payload.get("edge_metadata")
    if not isinstance(edges, list) or not edges or not isinstance(metadata, list) or len(edges) != len(metadata):
        raise ValueError("Edges and aligned edge_metadata must be nonempty lists")
    seen: set[tuple[int, ...]] = set()
    covered: set[int] = set()
    identity_ids = {code_index["OT"], code_index["Tlog"]}
    for edge_index, (edge, item) in enumerate(zip(edges, metadata)):
        if not isinstance(edge, list) or not edge or any(type(i) is not int or not 0 <= i < num_nodes for i in edge):
            raise ValueError("Invalid or out-of-range Weather edge membership")
        if edge != sorted(set(edge)):
            raise ValueError("Edge members must be sorted unique node indices")
        members = tuple(edge)
        if members in seen:
            raise ValueError("Duplicate edge sets must have their metadata families merged")
        seen.add(members)
        covered.update(edge)
        if not isinstance(item, dict) or type(item.get("edge_index")) is not int or item["edge_index"] != edge_index:
            raise ValueError("Misaligned Weather edge metadata index")
        if item.get("member_names") != [names[i] for i in edge]:
            raise ValueError("Edge member_names do not match named node indices")
        if item.get("canonical_members") != [canonical_variable_name(names[i]) for i in edge]:
            raise ValueError("Canonical edge member metadata does not match the names")
        if type(item.get("anchor")) is not int or item["anchor"] not in edge or item.get("anchor_name") != names[item["anchor"]]:
            raise ValueError("Edge anchor was dropped or is misaligned")
        if not isinstance(item.get("families"), list) or not item["families"] or any(not isinstance(f, str) or not f for f in item["families"]):
            raise ValueError("Every candidate must retain its semantic families")
        allowed_evidence = {"domain_relation_hypothesis", "measurement_group"}
        evidence_types = item.get("evidence_types")
        if (item.get("evidence_type") not in allowed_evidence
                or not isinstance(evidence_types, list) or not evidence_types
                or any(value not in allowed_evidence for value in evidence_types)
                or item["evidence_type"] not in evidence_types):
            raise ValueError("Candidate must be tagged as hypothesis or measurement group")
        if item.get("causal_effect_claimed") is not False or item.get("exact_physical_law_enforced") is not False:
            raise ValueError("This schema does not support causal or exact-law claims")
        if identity_ids.intersection(edge):
            if len(edge) != 1 or item.get("identity_only") is not True or item.get("relation_class") != "identity_singleton":
                raise ValueError("OT and Tlog may occur only as identity singletons")
        elif len(edge) == 1 or item.get("identity_only") is not False:
            raise ValueError("Only OT and Tlog may use identity singleton candidates")
    if covered != set(range(num_nodes)):
        raise ValueError("Weather graph must cover every data variable")

    incidence = torch.zeros(num_nodes, len(edges), dtype=torch.float32)
    for edge_index, members in enumerate(edges):
        incidence[members, edge_index] = 1.0
    pool = incidence / incidence.sum(1, keepdim=True).clamp_min(1.0)
    edge_pool = incidence / incidence.sum(0, keepdim=True).clamp_min(1.0)
    return {
        "incidence": incidence,
        "edge_pool": edge_pool,
        "pool": pool,
        "edges": edges,
        "metadata": metadata,
        "edge_metadata": metadata,
        "variable_names": names,
        "construction": construction,
        "schema_version": SCHEMA_VERSION,
        "num_nodes": torch.tensor(num_nodes),
    }
