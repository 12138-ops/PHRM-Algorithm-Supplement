"""Weather main-table settings with repository-relative defaults."""
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
    "Weather": dict(
        nodes=21, cycle=144, daily=144, weekly=1008, split="70/10/20",
        horizons=(96, 192, 336, 720),
        source=DATA_ROOT / "raw" / "weather_wide.csv",
        edge=DATA_ROOT / "physical" / "Weather_semantic.json",
        loader="Dataset_Custom", topology="measurement_semantic_hypotheses",
        use_revin=1, dropout=0.5, base_lr=0.001, base_batch=64,
        base_epochs=30, adapter_batch=32, patience=5,
    ),
}


def checkpoint_path(root, dataset, horizon, seed):
    if dataset not in SPECS:
        raise ValueError(dataset)
    return Path(root) / "base_checkpoints" / dataset / f"h{int(horizon)}" / f"seed{int(seed)}.pth"
