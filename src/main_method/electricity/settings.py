"""Electricity main-table settings with repository-relative defaults."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
PROJECTS = PROJECT_ROOT
OFFICIAL = PROJECT_ROOT / "references" / "TQNet"
UPSTREAM = ROOT
DATA_ROOT = Path(os.environ.get("PHRM_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()

SEEDS = (2022, 2023, 2024)
SPECS = {
    "Electricity": dict(
        nodes=321, cycle=168, daily=24, weekly=168, split="70/10/20",
        horizons=(96, 192, 336, 720),
        source=DATA_ROOT / "raw" / "electricity_OT.csv",
        edge=DATA_ROOT / "physical" / "Electricity.csv",
        loader="Dataset_Custom", topology="ordered_channel_proxy",
        use_revin=1, dropout=0.0, fresh_base=False, adapter_batch=64,
    ),
}


def checkpoint_path(root, dataset, horizon, seed):
    if dataset not in SPECS:
        raise ValueError(dataset)
    return Path(root) / "base_checkpoints" / dataset / f"h{int(horizon)}" / f"seed{int(seed)}.pth"
