"""Static Weather domain-relation candidates, constructed from column names only.

The edges are hypotheses about related measurements, not discovered causal
relations or an assertion that a numerical conservation law is enforced.  No
observed values, forecasts, labels, or train/validation/test rows enter this
construction.  OT is intentionally left uninterpreted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = "weather-semantic-candidates-v1"
CANONICAL_VARIABLES = frozenset(
    ("H2OC", "PAR", "SWDR", "T", "Tdew", "Tlog", "Tpot", "VPact",
     "VPdef", "VPmax", "max.PAR", "max.wv", "p", "rain", "raining",
     "rh", "rho", "sh", "wd", "wv", "OT")
)

# The first listed member is a declared anchor.  Membership is never capped or
# inferred from neighboring column indices.  These groups are fixed before any
# forecasting experiment and are independent of all measured values.
RELATION_SPECS = (
    ("potential_temperature", ("p", "T", "Tpot"), "domain_relation_hypothesis"),
    ("vapor_pressure_humidity", ("VPmax", "VPact", "VPdef", "rh"), "domain_relation_hypothesis"),
    ("temperature_saturation_pressure", ("T", "VPmax"), "domain_relation_hypothesis"),
    ("dewpoint_actual_vapor_pressure", ("Tdew", "VPact"), "domain_relation_hypothesis"),
    ("specific_humidity_pressure", ("p", "VPact", "sh"), "domain_relation_hypothesis"),
    ("water_concentration_pressure", ("p", "VPact", "H2OC"), "domain_relation_hypothesis"),
    ("density_temperature_water", ("p", "T", "rho", "H2OC"), "domain_relation_hypothesis"),
    ("wind_measurements", ("wv", "max.wv", "wd"), "measurement_group"),
    ("precipitation_measurements", ("rain", "raining"), "measurement_group"),
    ("radiation_measurements", ("SWDR", "PAR", "max.PAR"), "measurement_group"),
)


def canonical_variable_name(name: str) -> str:
    """Keep the stable measurement prefix; tolerate damaged unit encodings."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every variable name must be a nonempty string")
    prefix=name.strip().split("(", 1)[0].strip()
    return re.sub(r'^max\.\s+', 'max.', prefix)


def validate_variable_names(variable_names: Sequence[str]) -> tuple[list[str], dict[str, int]]:
    if isinstance(variable_names, (str, bytes)):
        raise ValueError("variable_names must be an ordered sequence of column names")
    names = list(variable_names)
    codes = [canonical_variable_name(name) for name in names]
    if len(names) != len(set(names)) or len(codes) != len(set(codes)):
        raise ValueError("Duplicate Weather variable names or measurement prefixes")
    found = set(codes)
    if found != CANONICAL_VARIABLES:
        raise ValueError(
            f"Weather requires exactly 21 known measurements; "
            f"missing={sorted(CANONICAL_VARIABLES - found)}, "
            f"unexpected={sorted(found - CANONICAL_VARIABLES)}"
        )
    return names, {code: index for index, code in enumerate(codes)}


def build_weather_graph(variable_names: Sequence[str]) -> dict:
    """Build static named candidate edges in the supplied data-column order.

    OT and Tlog are represented by identity singleton edges only.  In
    particular, OT is not interpreted as carbon dioxide or any other sensor.
    """
    names, index = validate_variable_names(variable_names)
    edges: list[list[int]] = []
    metadata: list[dict] = []
    by_members: dict[tuple[int, ...], int] = {}

    def add(family: str, members: tuple[str, ...], evidence_type: str, *, identity: bool = False):
        ids = tuple(sorted(index[code] for code in members))
        if ids in by_members:
            item = metadata[by_members[ids]]
            if family not in item["families"]:
                item["families"].append(family)
            if evidence_type not in item["evidence_types"]:
                item["evidence_types"].append(evidence_type)
            return
        by_members[ids] = len(edges)
        edges.append(list(ids))
        metadata.append({
            "edge_index": len(edges) - 1,
            "families": [family],
            "evidence_type": evidence_type,
            "evidence_types": [evidence_type],
            "relation_class": "identity_singleton" if identity else evidence_type,
            "member_names": [names[i] for i in ids],
            "canonical_members": [canonical_variable_name(names[i]) for i in ids],
            "anchor": index[members[0]],
            "anchor_name": names[index[members[0]]],
            "direction": "identity" if identity else "undirected_semantic_candidate",
            "causal_effect_claimed": False,
            "exact_physical_law_enforced": False,
            "identity_only": identity,
        })

    for family, members, evidence in RELATION_SPECS:
        add(family, members, evidence)
    for name in ("Tlog", "OT"):
        # A singleton is bookkeeping for a measurement, not a physical link.
        add("identity_singleton", (name,), "measurement_group", identity=True)

    covered = {member for edge in edges for member in edge}
    if covered != set(range(len(names))):
        raise ValueError("Static Weather candidates do not cover all variables")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "Weather",
        "num_variables": len(names),
        "variable_names": names,
        "variable_to_index": {name: i for i, name in enumerate(names)},
        "edges": edges,
        "edge_metadata": metadata,
        "construction": {
            "source": "column_headers_and_predeclared_domain_relation_hypotheses",
            "uses_data_values": False,
            "uses_labels": False,
            "uses_column_order_adjacency": False,
            "causal_effect_claimed": False,
            "exact_physical_law_enforced": False,
            "identity_only_variables": [names[index[code]] for code in ("Tlog", "OT")],
            "note": "Semantic candidates and measurement groups; no causal or exact-law claim. OT remains uninterpreted.",
        },
    }


def ordered_weather_csv_columns(csv_path: str | Path) -> list[str]:
    """Read only the CSV header, matching Dataset_Custom's OT-last order."""
    with Path(csv_path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        try:
            columns = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError("Weather CSV is empty") from exc
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV column names")
    if "date" not in columns or "OT" not in columns:
        raise ValueError("Weather CSV must contain date and OT columns")
    names = [name for name in columns if name not in ("date", "OT")] + ["OT"]
    validate_variable_names(names)
    return names


def write_weather_graph(csv_path: str | Path, output_path: str | Path) -> dict:
    """Write deterministic JSON without reading any time-series data rows."""
    payload = build_weather_graph(ordered_weather_csv_columns(csv_path))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    built = write_weather_graph(args.csv_path, args.output_path)
    print(json.dumps({"output": str(args.output_path), "num_variables": built["num_variables"],
                      "candidate_edges": len(built["edges"]), "covered_variables": len(built["variable_names"]),
                      "uses_data_values": False, "causal_effect_claimed": False}))
