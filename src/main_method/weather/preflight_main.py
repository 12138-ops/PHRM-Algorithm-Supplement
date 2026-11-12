"""Validate Weather data and named semantic supports before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generic_graph import load_graph
from protocol import load_data
from settings import SPECS
from train_phase_balanced import PHASE_VERSION


def main() -> None:
    spec = SPECS["Weather"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=spec["source"])
    parser.add_argument("--edge-path", type=Path, default=spec["edge"])
    args = parser.parse_args()

    prepared = load_data("Weather", args.data_path)
    graph = load_graph(args.edge_path, spec["nodes"], variable_names=prepared.metadata["variable_names"])
    if int(graph["metadata"]["covered_variables"]) != spec["nodes"]:
        raise RuntimeError("semantic graph does not cover all Weather variables")

    root = Path(__file__).resolve().parent
    audit = root / "audit" / "preflight.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "repair_version": PHASE_VERSION,
        "test_predictions_evaluated": False,
        "datasets": {
            "Weather": {
                "protocol": prepared.metadata,
                "graph": graph["metadata"],
            }
        },
    }
    audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: Weather inputs and protocol frozen in {audit}")


if __name__ == "__main__":
    main()
