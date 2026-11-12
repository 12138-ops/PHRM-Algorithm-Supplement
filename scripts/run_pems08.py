#!/usr/bin/env python3
"""Run the paired PEMS08 strict70 TQNet and PHRM experiments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "pems08_strict70.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "torch", "ptflops", "PyYAML")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> int:
    with CONFIG.open(encoding="utf-8") as stream:
        config = json.load(stream)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=config["horizons"])
    parser.add_argument("--seeds", type=int, nargs="+", default=config["seeds"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    if not args.dry_run:
        required = [root / rel for rel in config["inputs"].values()]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            parser.error("required input file(s) missing: " + ", ".join(missing))
        try:
            import torch
        except ImportError:
            parser.error("PyTorch is not installed; create the environment from environment.yml")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("the requested CUDA device is unavailable; use a CUDA-enabled environment")

    source = REPO / config["source_root"]
    base = config["baseline"]
    phrm = config["phrm"]
    jobs = []
    for horizon in args.horizons:
        if horizon not in config["horizons"]:
            parser.error(f"unsupported horizon {horizon}; choose from {config['horizons']}")
        for seed in args.seeds:
            if seed not in config["seeds"]:
                parser.error(f"unsupported seed {seed}; choose from {config['seeds']}")
            checkpoint = root / base["output_root"] / f"h{horizon}" / f"seed{seed}" / "checkpoint.pth"
            common = [
                "--root", str(root), "--horizon", str(horizon), "--seed", str(seed),
                "--split", config["split"], "--device", args.device,
            ]
            baseline_command = [
                args.python, str(source / "train_tqnet_base_70.py"), *common,
                "--epochs", str(base["epochs"]), "--patience", str(base["patience"]),
                "--batch_size", str(base["batch_size"]), "--lr", str(base["learning_rate"]),
                "--output_root", base["output_root"],
            ]
            phrm_command = [
                args.python, str(source / "train_tqnet_physicskv.py"), *common,
                "--epochs", str(phrm["epochs"]), "--patience", str(phrm["patience"]),
                "--batch_size", str(phrm["batch_size"]), "--bank_batch_size", str(phrm["bank_batch_size"]),
                "--memory_stride", str(phrm["memory_stride"]), "--topk", str(phrm["topk"]),
                "--temperature", str(phrm["temperature"]), "--hidden", str(phrm["hidden"]),
                "--lr", str(phrm["learning_rate"]), "--runs_root", phrm["runs_root"],
                "--base_checkpoint", str(checkpoint),
            ]
            jobs.extend((baseline_command, phrm_command))

    if args.dry_run:
        for command in jobs:
            print(shlex.join(command))
        return 0

    data_path, graph_path = (root / rel for rel in config["inputs"].values())
    manifest_path = root / phrm["runs_root"] / "reproduction_run.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": config["profile"],
        "started_unix": time.time(),
        "repository_revision": git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "data_sha256": sha256(data_path),
        "graph_sha256": sha256(graph_path),
        "source_sha256": {
            path.relative_to(REPO).as_posix(): sha256(path)
            for path in sorted(source.rglob("*.py"))
        },
        "commands": [],
        "status": "RUNNING",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        for command in jobs:
            record = {"command": command, "command_display": shlex.join(command), "started_unix": time.time()}
            manifest["commands"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            subprocess.run(command, cwd=source, check=True)
            record["status"] = "COMPLETE"
            record["finished_unix"] = time.time()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["status"] = "COMPLETE"
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "FAILED"
        manifest["failed_return_code"] = exc.returncode
        raise
    finally:
        manifest["finished_unix"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run complete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
