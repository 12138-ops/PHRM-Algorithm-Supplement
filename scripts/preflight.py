#!/usr/bin/env python3
"""Check the reference PEMS08 environment and input file contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        parser.error(f"missing dependency {exc.name!r}; install environment.yml first")

    root = args.project_root.expanduser().resolve()
    data_path = root / "dataset" / "PEMS08_data.npz"
    graph_path = root / "dataset" / "physical" / "PEMS08_physical_candidates.json"
    missing = [str(path) for path in (data_path, graph_path) if not path.is_file()]
    if missing:
        parser.error("required input file(s) missing: " + ", ".join(missing))

    with np.load(data_path, allow_pickle=False) as archive:
        if "data" not in archive:
            parser.error("PEMS08_data.npz must contain an array named 'data'")
        data = archive["data"]
        if data.ndim != 3 or data.shape[1:] != (170, 1):
            parser.error(f"expected data shape [time, 170, 1], got {data.shape}")
        if not np.isfinite(data).all():
            parser.error("PEMS08 data contains NaN or infinite values")

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        edges = graph["edges"]
        if not isinstance(edges, list) or not edges:
            raise ValueError("'edges' must be a non-empty list")
        for index, edge in enumerate(edges):
            if not isinstance(edge, list) or len(edge) < 2:
                raise ValueError(f"edge {index} must contain at least two node indices")
            if any(not isinstance(node, int) or node < 0 or node >= 170 for node in edge):
                raise ValueError(f"edge {index} has a node index outside [0, 169]")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid physical graph JSON: {exc}")

    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")

    result = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "project_root": str(root),
        "data_shape": list(data.shape),
        "data_sha256": sha256(data_path),
        "graph_edges": len(edges),
        "graph_sha256": sha256(graph_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
