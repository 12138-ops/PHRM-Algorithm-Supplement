"""Small, explicit GPU process launcher for the evidence CLI."""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT


def activity(gpu: int) -> list[dict[str, str]]:
    """Return compute processes on a physical GPU index; fail closed if unknown."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("nvidia-smi is required to check GPU activity")
    gpu_rows = subprocess.check_output(
        [executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    uuids = {
        row[1].strip()
        for row in csv.reader(gpu_rows.splitlines())
        if len(row) >= 2 and int(row[0].strip()) == int(gpu)
    }
    if not uuids:
        raise ValueError(f"physical GPU index {gpu} was not reported by nvidia-smi")
    process_rows = subprocess.check_output(
        [executable, "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
    )
    active = []
    for row in csv.reader(process_rows.splitlines()):
        if len(row) >= 3 and row[0].strip() in uuids:
            active.append({"gpu_uuid": row[0].strip(), "pid": row[1].strip(), "process_name": row[2].strip()})
    return active


def call(gpu: int, job_id: str, argv: list[str], wait: bool = False) -> int:
    """Run `evidence.py` with one physical GPU visible; return PID or exit code."""
    if not argv or argv[0] not in {"largest_efficiency", "efficiency", "cases", "smoke"}:
        raise ValueError(f"unsupported evidence task: {argv[0] if argv else None}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{safe_id}.log").open("ab")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    command = [sys.executable, str(ROOT / "evidence.py"), *argv]
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
