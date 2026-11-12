# Main-method source map

Only the paper's main PHRM method and the paired TQNet baseline paths are included. Ablation, sensitivity, support-counterfactual, efficiency, and exploratory runs were filtered out of the transferred server snapshots.

| Source root | Dataset | Main training files |
| --- | --- | --- |
| `src/main_method/pems03/` | PEMS03 | `train_tqnet_base_generic.py`, `train_tqnet_physicskv_generic.py` |
| `src/main_method/pems04/` | PEMS04 | `train_tqnet_base_generic.py`, `train_tqnet_physicskv_generic.py` |
| `src/main_method/pems07/` | PEMS07 | `train_tqnet_base_generic.py`, `train_tqnet_physicskv_generic.py`, `coverage_graph.py` |
| `src/main_method/pems08/` | PEMS08 | `train_tqnet_base.py`, `train_tqnet_physicskv.py`, `pems08_protocol.py` |
| `src/main_method/solar/` | Solar | `train_tqnet_base_generic.py`, `train_tqnet_physicskv_generic.py` |
| `src/main_method/electricity/` | Electricity | `train_phase_balanced.py`, `training_common.py` |
| `src/main_method/weather/` | Weather | `train_tqnet_base_generic.py`, `train_phase_balanced.py`, `training_common.py` |

Run all configured cells through [`scripts/run_main.py`](../scripts/run_main.py). Each source root is isolated on `sys.path` so that dataset-specific copies of `models`, `protocol`, and graph code do not shadow one another. The root runner provides the common data paths and launch parameters.

The Electricity source snapshot consumes paired baseline checkpoints and does not include a baseline-training entry point. Their expected portable paths and hashes are in [`reference_artifact_hashes.csv`](../manifests/reference_artifact_hashes.csv). Other dataset runners train the paired TQNet baseline when a compatible local checkpoint is absent.
