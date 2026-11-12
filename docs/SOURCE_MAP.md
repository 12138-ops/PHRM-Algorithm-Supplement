# Main-method source map

Only the main PHRM method and its paired TQNet baseline code are included. Ablation, sensitivity, support-validation, evidence, and exploratory experiment code is omitted. Inputs and checkpoints are external; see [`DATA_AND_CHECKPOINTS.md`](DATA_AND_CHECKPOINTS.md).

| Directory | Scope | Entry scripts |
| --- | --- | --- |
| `src/tqnet_physics_residual_kv_all/` | Dataset-parameterized TQNet baseline and PHRM training | `train_tqnet_base_generic.py`, `train_tqnet_physicskv_generic.py` |
| `src/tqnet_physics_residual_kv_pems08_final/` | Paired PEMS08 reference workflow | `train_tqnet_base_70.py`, `train_tqnet_physicskv.py` |

The PEMS08 profile is orchestrated by [`scripts/run_pems08.py`](../scripts/run_pems08.py). The generic scripts accept dataset paths, graph paths, forecast horizon, cycle, seed, and split as explicit arguments. `scripts/check_package.py` checks Python syntax, local imports, prohibited path names, and the checksum manifest without requiring training dependencies.
