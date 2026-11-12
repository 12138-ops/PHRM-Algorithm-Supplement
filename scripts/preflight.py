#!/usr/bin/env python3
"""Check exact paper-table data and physical-support files before a run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from repro_utils import REPO, load_config, select_datasets, verify_assets


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO / "data")
    parser.add_argument("--datasets", nargs="+", choices=list(config["datasets"]))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    selected = select_datasets(config, args.datasets)

    report = {
        "profile": config["profile"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "status": "PASS",
        "datasets": {},
    }
    for name in selected:
        spec = config["datasets"][name]
        hashes = verify_assets(root, name, spec)
        report["datasets"][name] = {
            "data": {"path": spec["data"], "sha256": hashes["data"]},
            "physical_support": {"path": spec["graph"], "sha256": hashes["graph"]},
            "horizons": spec["horizons"],
            "seeds": config["seeds"],
        }

    output = args.report.expanduser().resolve() if args.report else root / "input_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: checked {len(selected)} dataset(s); report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
