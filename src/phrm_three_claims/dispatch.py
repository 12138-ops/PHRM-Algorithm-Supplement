"""GPU process launcher used by the three-claim efficiency queue."""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys

from common import ROOT


def activity(gpu: int) -> list[dict[str, str]]:
    """Return compute processes on a physical GPU index; fail if it cannot be checked."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("nvidia-smi is required to check GPU activity")
    rows = subprocess.check_output(
        [executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    uuids = {
        row[1].strip()
        for row in csv.reader(rows.splitlines())
        if len(row) >= 2 and int(row[0].strip()) == int(gpu)
    }
    if not uuids:
        raise ValueError(f"physical GPU index {gpu} was not reported by nvidia-smi")
    processes = subprocess.check_output(
        [executable, "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [
        {"gpu_uuid": row[0].strip(), "pid": row[1].strip(), "process_name": row[2].strip()}
        for row in csv.reader(processes.splitlines())
        if len(row) >= 3 and row[0].strip() in uuids
    ]


def call(gpu: int, job_id: str, argv: list[str], wait: bool = False) -> int:
    """Run `three_claims.py` with one physical GPU visible; return PID or exit code."""
    if not argv or argv[0] not in {"largest_efficiency", "efficiency", "cases", "smoke", "fit", "replay"}:
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
