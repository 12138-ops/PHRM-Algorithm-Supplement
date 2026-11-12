#!/usr/bin/env python3
"""Validate and summarize saved baseline/PHRM main-table results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from repro_utils import REPO, load_config


def result_path(project_root: Path, dataset: str, spec: dict, horizon: int, seed: int) -> Path:
    if spec["method"] == "pems08":
        return project_root / "pems08" / "runs_main" / f"h{horizon}" / f"seed{seed}" / "result.json"
    return REPO / "src" / "main_method" / spec["source_root"] / "runs_main" / dataset / f"h{horizon}" / f"seed{seed}" / "result.json"


def paired_metrics(payload: dict) -> tuple[dict, dict]:
    test = payload.get("test", {})
    if "base" not in test or "trained" not in test:
        raise ValueError("result does not contain paired base and trained test metrics")
    return test["base"], test["trained"]


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO / "data")
    parser.add_argument("--datasets", nargs="+", choices=list(config["datasets"]))
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    datasets = args.datasets or list(config["datasets"])

    cell_rows = []
    seed_values: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    missing = []
    for dataset in datasets:
        spec = config["datasets"][dataset]
        for horizon in spec["horizons"]:
            for seed in config["seeds"]:
                path = result_path(project_root, dataset, spec, horizon, seed)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("test_labels_used_for_selection", False):
                    raise ValueError(f"test labels were used for selection: {path}")
                base, phrm = paired_metrics(payload)
                for method, metrics in (("TQNet", base), ("PHRM", phrm)):
                    row = {"dataset": dataset, "horizon": horizon, "seed": seed, "method": method}
                    for metric in ("mse", "mae"):
                        value = float(metrics[metric])
                        if not math.isfinite(value):
                            raise ValueError(f"non-finite {metric} in {path}")
                        row[metric] = value
                        seed_values.setdefault((dataset, method, metric), {}).setdefault(seed, []).append(value)
                    cell_rows.append(row)
    if missing:
        raise FileNotFoundError("missing result file(s):\n" + "\n".join(missing[:40]))

    output = project_root / "results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "seed_and_horizon_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "horizon", "seed", "method", "mse", "mae"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cell_rows)

    summary_rows = []
    for (dataset, method, metric), by_seed in sorted(seed_values.items()):
        per_seed = [sum(values) / len(values) for _, values in sorted(by_seed.items())]
        summary_rows.append({
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "mean_over_seed_horizon_means": statistics.mean(per_seed),
            "sample_std_across_seeds": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
            "seeds": len(per_seed),
            "horizons_per_seed": len(next(iter(by_seed.values()))),
        })
    with (output / "main_table_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "method", "metric", "mean_over_seed_horizon_means", "sample_std_across_seeds", "seeds", "horizons_per_seed"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summarized {len(cell_rows) // 2} paired cells across {len(datasets)} dataset(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
