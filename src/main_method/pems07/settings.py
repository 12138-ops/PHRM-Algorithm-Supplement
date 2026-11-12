"""PEMS07 main-table settings with repository-relative defaults."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
PROJECTS = PROJECT_ROOT
OFFICIAL = PROJECT_ROOT / "references" / "TQNet"
DATA_ROOT = Path(os.environ.get("PHRM_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()

SEEDS = (2022, 2023, 2024)
SPECS = {
    "PEMS07": dict(
        nodes=883, cycle=288, daily=288, weekly=2016, split="60/20/20",
        horizons=(12, 24, 48, 96), source=DATA_ROOT / "raw" / "PEMS07.npz",
        edge=DATA_ROOT / "physical" / "PEMS07.csv", loader="Dataset_PEMS",
        topology="road_network", use_revin=0, dropout=0.0, base_lr=0.003,
        base_batch=32, base_epochs=30, patience=5, lradj="type3",
        optimizer="Adam", fresh_base=True, adapter_batch=4,
        official_script="scripts/run_main.py",
    ),
}


def checkpoint_path(root, dataset, horizon, seed):
    if dataset not in SPECS:
        raise ValueError(dataset)
    return Path(root) / "base_checkpoints" / dataset / f"h{int(horizon)}" / f"seed{int(seed)}.pth"
