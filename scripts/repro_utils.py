"""Shared helpers for the paper main-table workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "main_experiments.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def select_datasets(config: dict, names: Iterable[str] | None) -> list[str]:
    selected = list(names) if names else list(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return selected


def asset_paths(project_root: Path, spec: dict) -> tuple[Path, Path]:
    return project_root / spec["data"], project_root / spec["graph"]


def verify_assets(project_root: Path, dataset: str, spec: dict) -> dict[str, str]:
    data_path, graph_path = asset_paths(project_root, spec)
    actual = {}
    missing = [str(p) for p in (data_path, graph_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing input file(s): " + ", ".join(missing))
    for label, path, expected in (
        ("data", data_path, spec["data_sha256"]),
        ("graph", graph_path, spec["graph_sha256"]),
    ):
        digest = sha256_file(path)
        actual[label] = digest
        if digest != expected:
            raise ValueError(
                f"{dataset} {label} SHA256 mismatch for {path.name}: "
                f"expected {expected}, got {digest}"
            )
    return actual


def source_hashes(source_root: Path) -> dict[str, str]:
    return {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in sorted(source_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
