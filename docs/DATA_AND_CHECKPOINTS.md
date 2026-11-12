# Data, physical supports, and checkpoints

The Git repository contains code, reference hashes, all 14 raw-data/physical-support assets, and the 12 matched Electricity baseline checkpoints. The other 72 historical checkpoints remain external. Expected hashes for all data/support inputs are embedded in [`configs/main_experiments.json`](../configs/main_experiments.json), and all reference artifacts are listed in [`manifests/reference_artifact_hashes.csv`](../manifests/reference_artifact_hashes.csv).

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

The repository includes all 12 paired Electricity TQNet checkpoints at:

```text
data/checkpoints/Electricity/h{horizon}/seed{seed}.pth
```

Each checkpoint must match the TQNet architecture used by the Electricity source root. Their SHA256 values and byte sizes are listed in the reference artifact manifest, and `scripts/run_main.py` verifies them before use.

The root runner trains paired TQNet baselines for the other six datasets by default. To use all 84 historical paired baselines, add the remaining 72 files under `data/checkpoints/{dataset}/h{horizon}/seed{seed}.pth` using the entries in the reference artifact manifest and pass `--reuse-baselines`. The runner checks each file against its expected SHA256 before using it. Electricity always uses its 12 included reference checkpoints.

## Validation

Run a strict data/support hash check before training:

```bash
python scripts/preflight.py --project-root data
```

The runner refuses to start if an input is missing or differs from the reference hash. Raw data, physical-support files, and the 12 reference Electricity checkpoints are tracked in Git. Training logs, generated results, and optional baseline weights for other datasets remain outside Git by default. Keep the generated `data/run_manifest.json`, per-cell `result.json` files, and their hashes with the experiment record.
