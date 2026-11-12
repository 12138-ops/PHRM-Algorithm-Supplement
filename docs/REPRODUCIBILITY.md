# Reproducibility guide

## Reference profile in this repository

`configs/pems08_strict70.json` defines the included end-to-end reference profile. It uses the PEMS08 source root under `src/tqnet_physics_residual_kv_pems08_final/`, lookback 96, a chronological `70/10/20` split, horizons 12/24/48/96, seeds 2022/2023/2024, a validation-selected paired TQNet checkpoint, and a PHRM adapter trained from that checkpoint. Values match the CLI defaults in the referenced scripts and are passed explicitly by `scripts/run_pems08.py`.

`python scripts/preflight.py --project-root data/pems08 --require-cuda` checks the environment and the two required inputs. `python scripts/run_pems08.py --project-root data/pems08` runs the full profile sequentially. To inspect commands without starting jobs, add `--dry-run`.

For each baseline/PHRM pair, the runner records the command, source revision when available, source-file hashes, dataset and graph hashes, package versions, GPU model, and CUDA/PyTorch versions in `data/pems08/reproduction_run.json`. Training scripts also save their own checkpoints, histories, validation/test metrics, and protocol metadata below the project root.

## Determinism

The reference runner fixes seeds, does not shuffle validation/test windows, uses the source scripts' chronological splits, and runs one job at a time. The source implementations set deterministic cuDNN flags where provided. CUDA kernels can still differ across GPU architectures, driver versions, and library builds; compare saved protocol metadata and environment manifests before interpreting small metric differences.

## Reproducing the paper tables

The dataset/horizon index is in `configs/paper_datasets.json`. It is not a complete executable experiment plan. The archive did not provide the per-table job manifests, baseline recipes, processed data, physical support files, checkpoints, or result files described in the paper. Before claiming table-level reproduction, restore and hash those exact items, and run the source snapshot tied to the target experiment. In particular, the repository contains multiple PEMS08 split variants; use the one tied to the target table rather than substituting the `strict70` profile.

## Historical scripts

Scripts in `src/` are retained for transparency. They can require Linux-specific facilities, multiple external project trees referenced through `PHRM_PROJECTS_ROOT`, and third-party code that is not vendored here. The dispatch helpers launch the included evidence and three-claim CLIs on one explicitly selected GPU; they do not create missing manifests or assets. Consult each script's argument parser and the data requirements before launching it.
