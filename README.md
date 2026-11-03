# PHRM ICDE Reproducibility — Algorithm Source Only

This archive contains selected Python source files for the PHRM/TQNet forecasting method and its support construction, residual-memory retrieval, adapter, and ablation logic. Source files remain grouped by their original project directory under `src/` so their provenance and relative imports are visible.

## Scope

This is a source-only code bundle. It intentionally contains no datasets, dataset caches, checkpoints, trained weights, generated predictions, experiment results, logs, plots, or machine-specific run configurations. The included training and evaluation code therefore needs the corresponding data, configuration, and runtime paths supplied separately before it can run.

The source roots and project-relative provenance are listed in `manifests/source_inventory.csv`. Only Python source was copied; files omitted by the source-only filter are not represented as available here. Hard-coded server project-root strings were replaced in the bundle with the `PHRM_PROJECTS_ROOT` environment variable; set it to the checkout root when using those source files.

## Code map

- `src/ICDE_TQNet_PhysicsResidualKV_All_20261015/`: generic TQNet and physical residual Key–Value implementation.
- `src/ICDE_PhysicsResidualKV_MultiDataset_20261014/`: multi-dataset residual-memory model and backbone.
- `src/ICDE_PHRM_PhysicalSupportValidation_20261020/`: physical graph/support construction, memory sampling, adapter, and validation logic.
- `src/ICDE_PHRM_FinalEvidence_20261017/`, `src/ICDE_PHRM_FinalSensitivity_20261017/`, and `src/ICDE_PHRM_ThreeClaims_GPU23_20261023/`: method variants and algorithm implementations retained from later isolated studies.
- `src/ICDE_Baselines_Ablations_20261016/`: selected TQNet and ablation logic only. Third-party baseline/vendor directories and benchmark outputs were excluded.
- `src/TQNet_PhysicsResidualKV_PEMS08_FINAL_20261014/` and `src/TQNet_PhysicsResidualKV_PEMS08_ICDE_20261014/`: PEMS08-specific model/training implementation.

## Important limits

The final manuscript source and its author metadata were not present in the audited project directories. This bundle does not establish that each copied file corresponds to the final submitted manuscript. No project-wide source license was found in the audited roots; the `LICENSE` file records that status and does not grant reuse rights. Review the original upstream licenses and obtain an explicit project license before public redistribution.
