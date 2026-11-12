#!/usr/bin/env python3
"""Run dependency-free checks on package names, Python syntax and hashes."""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
EXTERNAL_MODULES = {"matplotlib", "numpy", "pandas", "ptflops", "sklearn", "torch", "vendor_baselines"}
EXPECTED_SOURCE_ROOTS = {
    "tqnet_physics_residual_kv_all",
    "tqnet_physics_residual_kv_pems08_final",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_roots = {path.name for path in SRC.iterdir() if path.is_dir()}
    if source_roots != EXPECTED_SOURCE_ROOTS:
        print(
            "Unexpected source roots; expected only main-method code: "
            f"found={sorted(source_roots)}, expected={sorted(EXPECTED_SOURCE_ROOTS)}"
        )
        return 1

    bad_names = [
        path for path in REPO.rglob("*")
        if "icde" in path.name.lower() or re.search(r"(?<!\d)20\d{6,8}(?!\d)", path.name)
    ]
    if bad_names:
        print("Found a path name containing ICDE or a date stamp:")
        for path in bad_names:
            print(path.relative_to(REPO).as_posix())
        return 1

    python_files = sorted(REPO.rglob("*.py"))
    for path in python_files:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            compile(tree, str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"Syntax error in {path.relative_to(REPO)}: {exc}")
            return 1

        if path.is_relative_to(SRC):
            source_root = SRC / path.relative_to(SRC).parts[0]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".", 1)[0]
                    if top in sys.stdlib_module_names or top in EXTERNAL_MODULES:
                        continue
                    if (
                        any(source_root.rglob(f"{top}.py"))
                        or any(source_root.rglob(f"{top}/__init__.py"))
                        or any(source_root.rglob(f"{top}/*.py"))
                    ):
                        continue
                    print(f"Unresolved local import {name!r} in {path.relative_to(REPO)}")
                    return 1

    checksum_file = REPO / "manifests" / "SHA256SUMS.txt"
    if checksum_file.is_file():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            expected, rel = line.split("  ", 1)
            path = REPO / Path(rel)
            if not path.is_file() or sha256(path) != expected:
                print(f"Checksum mismatch: {rel}")
                return 1
    print(f"PASS: {len(python_files)} Python files compile; only main-method source roots remain; local imports resolve; paths contain no ICDE or date stamps; checksums match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
