# Reproducibility guide

## Reference profile in this repository

`configs/pems08_strict70.json` defines the included end-to-end reference profile. It uses the PEMS08 source root under `src/tqnet_physics_residual_kv_pems08_final/`, lookback 96, a chronological `70/10/20` split, horizons 12/24/48/96, seeds 2022/2023/2024, a validation-selected paired TQNet checkpoint, and a PHRM adapter trained from that checkpoint. Values match the CLI defaults in the referenced scripts and are passed explicitly by `scripts/run_pems08.py`.

`python scripts/preflight.py --project-root data/pems08 --require-cuda` checks the environment and the two required inputs. `python scripts/run_pems08.py --project-root data/pems08` runs the full profile sequentially. To inspect commands without starting jobs, add `--dry-run`.

For each baseline/PHRM pair, the runner records the command, source revision when available, source-file hashes, dataset and graph hashes, package versions, GPU model, and CUDA/PyTorch versions in `data/pems08/reproduction_run.json`. Training scripts also save their own checkpoints, histories, validation/test metrics, and protocol metadata below the project root.

## Determinism

The reference runner fixes seeds, does not shuffle validation/test windows, uses the source scripts' chronological splits, and runs one job at a time. The source implementations set deterministic cuDNN flags where provided. CUDA kernels can still differ across GPU architectures, driver versions, and library builds; compare saved protocol metadata and environment manifests before interpreting small metric differences.

## Main-method code for other datasets

`src/tqnet_physics_residual_kv_all/` contains the dataset-parameterized main method and paired TQNet baseline. The baseline command accepts an NPZ `data` array shaped `[time, nodes, channels]`. PHRM additionally needs an edge CSV and the matching baseline checkpoint. Consult each CLI's `--help` output and provide the dataset-specific horizon, cycle, split, seed, node count, and paths. This code path is included for the main method; it does not include ablation or sensitivity jobs.

## Reproducing the paper tables

The dataset/horizon index is in `configs/paper_datasets.json`. It is not a complete executable experiment plan. The archive did not provide the per-table job manifests, baseline recipes, processed data, physical support files, checkpoints, or result files described in the paper. Before claiming table-level reproduction, restore and hash those exact items, and run the source snapshot tied to the target experiment. In particular, the repository contains multiple PEMS08 split variants; use the one tied to the target table rather than substituting the `strict70` profile.

The package intentionally focuses on main-method experiments. It does not include ablation, sensitivity, support-counterfactual, efficiency, or retrieval-diagnostic jobs.
