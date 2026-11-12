#!/usr/bin/env python3
"""Regenerate repository-relative source inventory and SHA256 manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_packaged_file(path: Path, checksum_path: Path) -> bool:
    generated_parts = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
    if (
        not path.is_file()
        or path == checksum_path
        or any(part in generated_parts for part in path.parts)
        or path.suffix in {".pyc", ".pyo"}
    ):
        return False
    relative = path.relative_to(REPO)
    if relative.parts[:1] == ("data",):
        return (
            len(relative.parts) == 5
            and relative.parts[:3] == ("data", "checkpoints", "Electricity")
            and relative.suffix == ".pth"
        )
    return True


def main() -> None:
    MANIFESTS.mkdir(exist_ok=True)
    inventory_path = MANIFESTS / "source_inventory.csv"
    source_files = sorted((REPO / "src").rglob("*.py"))
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("path", "size_bytes", "sha256", "kind"))
        for path in source_files:
            writer.writerow((path.relative_to(REPO).as_posix(), path.stat().st_size, sha256(path), "python_source"))

    checksum_path = MANIFESTS / "SHA256SUMS.txt"
    files = sorted(
        path for path in REPO.rglob("*")
        if is_packaged_file(path, checksum_path)
    )
    with checksum_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(
            f"{sha256(path)}  {path.relative_to(REPO).as_posix()}\n" for path in files
        ))
    print(f"Inventoried {len(source_files)} Python source files; hashed {len(files)} files.")


if __name__ == "__main__":
    main()
