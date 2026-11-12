"""Launch a three-claim experiment command with one physical GPU visible."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from common import ROOT


def call(gpu: int, job_id: str, argv: list[str], wait: bool = False) -> int:
    if not argv or argv[0] not in {"fit", "replay", "efficiency", "cases", "smoke", "largest_efficiency"}:
        raise ValueError(f"unsupported three-claim task: {argv[0] if argv else None}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{safe_id}.log").open("ab")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    command = [sys.executable, str(ROOT / "three_claims.py"), *argv]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    if wait:
        return process.wait()
    return process.pid
