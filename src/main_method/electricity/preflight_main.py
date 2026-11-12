"""Validate the Electricity main-method inputs and freeze its protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generic_graph import load_graph
from protocol import load_data, sha256_file
from settings import SPECS
from train_phase_balanced import PHASE_VERSION


def main() -> None:
    spec = SPECS["Electricity"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=spec["source"])
    parser.add_argument("--edge-path", type=Path, default=spec["edge"])
    args = parser.parse_args()

    prepared = load_data("Electricity", args.data_path)
    graph = load_graph(args.edge_path, spec["nodes"])
    covered = int((graph["incidence"].sum(1) > 0).sum().item())
    if covered != spec["nodes"] or not graph["edges"]:
        raise RuntimeError("physical graph does not cover every Electricity variable")

    root = Path(__file__).resolve().parent
    audit = root / "audit" / "preflight.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "repair_version": PHASE_VERSION,
        "test_predictions_evaluated": False,
        "datasets": {
            "Electricity": {
                "protocol": prepared.metadata,
                "graph": {
                    "edge_count": len(graph["edges"]),
                    "covered_variables": covered,
                    "sha256": sha256_file(args.edge_path),
                },
            }
        },
    }
    audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: Electricity inputs and protocol frozen in {audit}")


if __name__ == "__main__":
    main()
