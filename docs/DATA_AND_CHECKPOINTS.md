# Data, physical supports, and checkpoints

The Git repository contains code and reference hashes, not large data or checkpoint binaries. Put assets under `data/` and retain the filenames and checksums in [`manifests/reference_artifact_hashes.csv`](../manifests/reference_artifact_hashes.csv). The expected hashes are also embedded for each data/support input in [`configs/main_experiments.json`](../configs/main_experiments.json).

| Dataset | Raw input path | Physical-support path | Nodes | Split | Horizons |
| --- | --- | --- | ---: | --- | --- |
| PEMS03 | `data/raw/PEMS03.npz` | `data/physical/PEMS03_indexed.csv` | 358 | 60/20/20 | 12, 24, 48, 96 |
| PEMS04 | `data/raw/PEMS04.npz` | `data/physical/PEMS04.csv` | 307 | 60/20/20 | 12, 24, 48, 96 |
| PEMS07 | `data/raw/PEMS07.npz` | `data/physical/PEMS07.csv` | 883 | 60/20/20 | 12, 24, 48, 96 |
| PEMS08 | `data/pems08/dataset/PEMS08_data.npz` | `data/pems08/dataset/physical/PEMS08_physical_candidates.json` | 170 | 60/20/20 | 12, 24, 48, 96 |
| Solar | `data/raw/solar_AL.txt` | `data/physical/Solar.csv` | 137 | 70/10/20 | 96, 192, 336, 720 |
| Electricity | `data/raw/electricity_OT.csv` | `data/physical/Electricity.csv` | 321 | 70/10/20 | 96, 192, 336, 720 |
| Weather | `data/raw/weather_wide.csv` | `data/physical/Weather_semantic.json` | 21 | 70/10/20 | 96, 192, 336, 720 |

All runs use lookback 96 and seeds 2022, 2023, and 2024. The archive formats, channel ordering, chronological splits, periodic settings, and main-method training parameters are recorded in the dataset source roots and `configs/main_experiments.json`.

## Electricity baseline checkpoints

Electricity needs 12 paired TQNet checkpoints. Store them as:

```text
data/checkpoints/Electricity/h{horizon}/seed{seed}.pth
```

For example, `h96/seed2022.pth`. Each checkpoint must match the TQNet architecture used by the Electricity source root. Expected SHA256 values and byte sizes are listed in the reference artifact manifest.

The root runner trains paired TQNet baselines for the other six datasets by default. To use the exact 84 historical paired baselines instead, place them under `data/checkpoints/{dataset}/h{horizon}/seed{seed}.pth` using the entries in the reference artifact manifest and pass `--reuse-baselines`. The runner checks each file against its expected SHA256 before using it. Electricity always uses its 12 reference checkpoints.

## Validation

Run a strict data/support hash check before training:

```bash
python scripts/preflight.py --project-root data
```

The runner refuses to start if an input is missing or differs from the reference hash. Do not put data, graphs, weights, logs, or generated results in Git. Keep the generated `data/run_manifest.json`, the per-cell `result.json` files, and their hashes with the experiment record.
