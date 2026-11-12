#!/usr/bin/env python3
"""Run the seven-dataset PHRM main-table workflow."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from repro_utils import REPO, load_config, select_datasets, sha256_file, source_hashes, verify_assets


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint_for(source_root: Path, project_root: Path, dataset: str, spec: dict, horizon: int, seed: int, use_reference: bool = False) -> Path:
    baseline = spec["baseline"]
    if baseline["mode"] == "external" or use_reference:
        return project_root / "checkpoints" / dataset / f"h{horizon}" / f"seed{seed}.pth"
    if spec["method"] == "pems08":
        return project_root / "pems08" / "base_checkpoints_60" / f"h{horizon}" / f"seed{seed}" / "checkpoint.pth"
    if dataset == "Weather":
        return source_root / "base_checkpoints" / "Weather" / f"h{horizon}" / f"seed{seed}.pth"
    return source_root / "base_checkpoints" / dataset / f"h{horizon}" / f"seed{seed}.pth"


def make_commands(
    python: str,
    device: str,
    project_root: Path,
    source_root: Path,
    dataset: str,
    spec: dict,
    horizon: int,
    seed: int,
    checkpoint: Path,
) -> tuple[list[str] | None, list[str]]:
    data_path, graph_path = (project_root / spec[key] for key in ("data", "graph"))
    common = [
        "--dataset", dataset,
        "--data_path", str(data_path),
        "--num_nodes", str(spec["nodes"]),
        "--horizon", str(horizon),
        "--cycle", str(spec["cycle"]),
        "--seed", str(seed),
        "--split", spec["split"],
        "--device", device,
    ]

    if spec["method"] == "generic":
        baseline = spec["baseline"]
        baseline_command = [
            python, str(source_root / "train_tqnet_base_generic.py"), *common,
            "--output", str(checkpoint),
            "--epochs", str(baseline["epochs"]),
            "--batch_size", str(baseline["batch_size"]),
        ]
        phrm_command = [
            python, str(source_root / "train_tqnet_physicskv_generic.py"),
            "--root", str(source_root), *common,
            "--edge_csv", str(graph_path),
            "--base_checkpoint", str(checkpoint),
            "--epochs", str(spec["phrm"]["epochs"]),
            "--patience", str(spec["phrm"]["patience"]),
            "--batch_size", str(spec["phrm"]["batch_size"]),
            "--memory_stride", str(spec["phrm"]["memory_stride"]),
            "--runs_root", "runs_main",
        ]
        return baseline_command, phrm_command

    if spec["method"] == "phase_balanced":
        phrm_command = [
            python, str(source_root / "train_phase_balanced.py"),
            "--root", str(source_root), *common,
            "--edge_csv", str(graph_path),
            "--base_checkpoint", str(checkpoint),
            "--epochs", str(spec["phrm"]["epochs"]),
            "--patience", str(spec["phrm"]["patience"]),
            "--batch_size", str(spec["phrm"]["batch_size"]),
            "--memory_stride", str(spec["phrm"]["memory_stride"]),
            "--runs_root", "runs_main",
        ]
        baseline_command = None
        if dataset == "Weather":
            baseline = spec["baseline"]
            baseline_command = [
                python, str(source_root / "train_tqnet_base_generic.py"), *common,
                "--output", str(checkpoint),
                "--epochs", str(baseline["epochs"]),
                "--batch_size", str(baseline["batch_size"]),
            ]
        return baseline_command, phrm_command

    if spec["method"] == "pems08":
        data_root = project_root / "pems08"
        baseline = spec["baseline"]
        baseline_command = [
            python, str(source_root / "train_tqnet_base.py"),
            "--root", str(data_root), "--horizon", str(horizon), "--seed", str(seed),
            "--split", spec["split"], "--epochs", str(baseline["epochs"]),
            "--patience", str(baseline["patience"]), "--batch_size", str(baseline["batch_size"]),
            "--workers", str(baseline["workers"]), "--lr", str(baseline["learning_rate"]),
            "--device", device, "--output_root", "base_checkpoints_60",
        ]
        phrm = spec["phrm"]
        phrm_command = [
            python, str(source_root / "train_tqnet_physicskv.py"),
            "--root", str(data_root), "--split", spec["split"],
            "--horizon", str(horizon), "--seed", str(seed),
            "--epochs", str(phrm["epochs"]), "--patience", str(phrm["patience"]),
            "--batch_size", str(phrm["batch_size"]),
            "--bank_batch_size", str(phrm["bank_batch_size"]),
            "--workers", str(phrm["workers"]), "--memory_stride", str(phrm["memory_stride"]),
            "--topk", str(phrm["topk"]), "--temperature", str(phrm["temperature"]),
            "--hidden", str(phrm["hidden"]), "--lr", str(phrm["learning_rate"]),
            "--device", device, "--base_checkpoint", str(checkpoint),
            "--runs_root", "runs_main",
        ]
        return baseline_command, phrm_command

    raise ValueError(f"unsupported method kind: {spec['method']}")


def environment_snapshot(device: str) -> dict:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": device,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": {
            name: (importlib.metadata.version(name) if _installed(name) else None)
            for name in ("numpy", "pandas", "scikit-learn", "scipy")
        },
    }


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def run_step(command: list[str], cwd: Path, log_path: Path, dry_run: bool, env: dict[str, str] | None = None) -> dict:
    record = {"command": command, "command_display": shlex.join(command), "cwd": str(cwd)}
    if dry_run:
        return {**record, "status": "NOT_RUN"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["log"] = str(log_path)
    record["started_at_utc"] = iso_now()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False, env=env)
    record["finished_at_utc"] = iso_now()
    record["return_code"] = completed.returncode
    record["status"] = "COMPLETE" if completed.returncode == 0 else "FAILED"
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return record


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO / "data")
    parser.add_argument("--datasets", nargs="+", choices=list(config["datasets"]), default=list(config["datasets"]))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-baselines", action="store_true", help="reuse existing checkpoints instead of training local baselines")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    selected = select_datasets(config, args.datasets)
    seeds = args.seeds or config["seeds"]
    for seed in seeds:
        if seed not in config["seeds"]:
            parser.error(f"unsupported seed {seed}; choose from {config['seeds']}")

    cells: list[tuple[str, int, int, dict]] = []
    for dataset in selected:
        spec = config["datasets"][dataset]
        horizons = args.horizons or spec["horizons"]
        for horizon in horizons:
            if horizon not in spec["horizons"]:
                parser.error(f"unsupported {dataset} horizon {horizon}; choose from {spec['horizons']}")
            for seed in seeds:
                cells.append((dataset, horizon, seed, spec))

    if args.dry_run:
        print(f"Profile: {config['profile']}; cells: {len(cells)}; seeds: {seeds}")
        for dataset in selected:
            spec = config["datasets"][dataset]
            hs = args.horizons or spec["horizons"]
            count = len(hs) * len(seeds)
            print(f"{dataset}: {count} cells, {spec['method']}, baseline={spec['baseline']['mode']}")
        print("Dry run only; no data files or training packages were read.")
        return 0

    try:
        env = environment_snapshot(args.device)
    except ImportError as exc:
        parser.error(f"PyTorch is required; install the environment in environment.yml: {exc}")
    if not env["cuda_available"] and args.device.startswith("cuda"):
        parser.error("CUDA is unavailable; select an available device or install a CUDA-enabled PyTorch build")

    input_hashes = {}
    for dataset in selected:
        try:
            input_hashes[dataset] = verify_assets(project_root, dataset, config["datasets"][dataset])
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    runtime_env = os.environ.copy()
    runtime_env["PHRM_DATA_ROOT"] = str(project_root)

    reference_checkpoints = json.loads((REPO / "configs" / "reference_checkpoints.json").read_text(encoding="utf-8"))
    external_missing = []
    for dataset, horizon, seed, spec in cells:
        source_root = REPO / "src" / "main_method" / spec["source_root"]
        checkpoint = checkpoint_for(source_root, project_root, dataset, spec, horizon, seed, args.reuse_baselines)
        if spec["baseline"]["mode"] == "external" and not checkpoint.is_file():
            external_missing.append(str(checkpoint))
        elif args.reuse_baselines and not checkpoint.is_file():
            external_missing.append(str(checkpoint))
        elif not args.reuse_baselines and spec["baseline"]["mode"] != "external" and checkpoint.exists():
            parser.error(f"baseline checkpoint already exists: {checkpoint}; use --reuse-baselines or move it aside")
    if external_missing:
        parser.error("required baseline checkpoint(s) missing:\n" + "\n".join(external_missing))

    for dataset, horizon, seed, spec in cells:
        if spec["baseline"]["mode"] != "external" and not args.reuse_baselines:
            continue
        checkpoint = checkpoint_for(
            REPO / "src" / "main_method" / spec["source_root"],
            project_root, dataset, spec, horizon, seed, args.reuse_baselines,
        )
        reference = reference_checkpoints[dataset][f"h{horizon}_seed{seed}"]
        actual = sha256_file(checkpoint)
        if actual != reference["sha256"]:
            parser.error(
                f"{dataset} baseline checkpoint SHA256 mismatch for {checkpoint.name}: "
                f"expected {reference['sha256']}, got {actual}"
            )

    manifest_path = project_root / "run_manifest.json"
    manifest = {
        "profile": config["profile"],
        "started_at_utc": iso_now(),
        "project_root": str(project_root),
        "repository_revision": _git_revision(),
        "datasets": selected,
        "seeds": seeds,
        "device": args.device,
        "environment": env,
        "input_sha256": input_hashes,
        "source_sha256": {
            dataset: source_hashes(REPO / "src" / "main_method" / config["datasets"][dataset]["source_root"])
            for dataset in selected
        },
        "jobs": [],
        "status": "RUNNING",
    }
    write_manifest(manifest_path, manifest)

    try:
        for dataset in selected:
            spec = config["datasets"][dataset]
            source_root = REPO / "src" / "main_method" / spec["source_root"]
            data_path, graph_path = (project_root / spec[key] for key in ("data", "graph"))
            if spec["method"] == "phase_balanced":
                preflight = [
                    args.python, str(source_root / "preflight_main.py"),
                    "--data-path", str(data_path), "--edge-path", str(graph_path),
                ]
                job = run_step(
                    preflight, source_root,
                    project_root / "logs" / f"{dataset.lower()}_preflight.log", False, runtime_env,
                )
                manifest["jobs"].append({"dataset": dataset, "stage": "preflight", **job})
                write_manifest(manifest_path, manifest)

            for dset, horizon, seed, cell_spec in (x for x in cells if x[0] == dataset):
                checkpoint = checkpoint_for(source_root, project_root, dataset, spec, horizon, seed, args.reuse_baselines)
                baseline_command, phrm_command = make_commands(
                    args.python, args.device, project_root, source_root, dataset, spec,
                    horizon, seed, checkpoint,
                )
                if baseline_command and not args.reuse_baselines:
                    job = run_step(
                        baseline_command, source_root,
                        project_root / "logs" / f"{dataset.lower()}_h{horizon}_seed{seed}_base.log", False, runtime_env,
                    )
                    manifest["jobs"].append({"dataset": dataset, "horizon": horizon, "seed": seed, "stage": "baseline", **job})
                    write_manifest(manifest_path, manifest)
                job = run_step(
                    phrm_command, source_root,
                    project_root / "logs" / f"{dataset.lower()}_h{horizon}_seed{seed}_phrm.log", False, runtime_env,
                )
                result_path = _result_path(source_root, project_root, dataset, spec, horizon, seed)
                result_record = {
                    "dataset": dataset, "horizon": horizon, "seed": seed, "stage": "phrm", **job,
                    "result_path": str(result_path),
                    "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
                    "base_checkpoint_sha256": sha256_file(checkpoint),
                }
                manifest["jobs"].append(result_record)
                write_manifest(manifest_path, manifest)
        manifest["status"] = "COMPLETE"
    except BaseException as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at_utc"] = iso_now()
        write_manifest(manifest_path, manifest)

    print(f"Completed {len(cells)} cells. Run manifest: {manifest_path}")
    return 0


def _result_path(source_root: Path, project_root: Path, dataset: str, spec: dict, horizon: int, seed: int) -> Path:
    if spec["method"] == "pems08":
        return project_root / "pems08" / "runs_main" / f"h{horizon}" / f"seed{seed}" / "result.json"
    return source_root / "runs_main" / dataset / f"h{horizon}" / f"seed{seed}" / "result.json"


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
