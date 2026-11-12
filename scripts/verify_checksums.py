#!/usr/bin/env python3
"""Verify the repository snapshot listed in manifests/SHA256SUMS.txt."""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path, PurePosixPath


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests" / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"malformed line {line_number}") from error
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"invalid SHA256 on line {line_number}")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or ":" in relative
            or "\\" in relative
        ):
            raise ValueError(f"unsafe relative path on line {line_number}")
        if relative == MANIFEST.relative_to(REPO).as_posix():
            raise ValueError("the checksum manifest must not hash itself")
        if relative in seen:
            raise ValueError(f"duplicate path on line {line_number}: {relative}")
        seen.add(relative)
        entries.append((expected, relative))
    return entries


def main() -> int:
    try:
        entries = manifest_entries()
    except (OSError, ValueError) as error:
        print(f"Invalid checksum manifest: {error}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for expected, relative in entries:
        target = REPO.joinpath(*PurePosixPath(relative).parts).resolve()
        if not target.is_relative_to(REPO.resolve()):
            failures.append(f"UNSAFE PATH {relative}")
            continue
        try:
            actual = sha256(target)
        except OSError:
            failures.append(f"MISSING {relative}")
            continue
        if actual != expected:
            failures.append(f"MISMATCH {relative}")

    if failures:
        print("Checksum verification failed:")
        print("\n".join(failures))
        return 1
    print(f"Verified {len(entries)} files against {MANIFEST.relative_to(REPO).as_posix()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
