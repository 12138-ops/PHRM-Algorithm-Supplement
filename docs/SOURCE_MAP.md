# Source map

The research code is kept in nine isolated experiment snapshots. Run scripts from their snapshot directory because many modules use same-directory imports. Inputs and checkpoints are generally external; see [`DATA_AND_CHECKPOINTS.md`](DATA_AND_CHECKPOINTS.md).

| Directory | Scope | Useful entry scripts |
| --- | --- | --- |
| `src/baselines_ablations/` | Matched baselines, ablations, protocol preparation and checks | `prepare.py`, `protocol_check.py`, `run_job.py`, `smoke_pipeline.py` |
| `src/phrm_final_evidence/` | PHRM evidence runs and retrieval diagnostics | `evidence.py`, `largest_evidence.py`, `structure_controls.py` |
| `src/phrm_final_sensitivity/` | PEMS07/PEMS08 sensitivity runs | `PEMS07/train_validation_only.py`, `PEMS08/train_validation_only.py` |
| `src/physical_support_validation/` | Physical-support validation, memory budget and adapter fitting | `experiment.py`, `fit_support.py`, `evaluate_locked.py` |
| `src/phrm_three_claims/` | Three-claim ablations, evidence, retrieval cases and efficiency scripts | `three_claims.py`, `evidence.py`, `largest_evidence.py` |
| `src/physics_residual_kv_multidataset/` | Multi-dataset residual KV training | `train_kv_multidataset.py`, `gate_and_test_multidataset.py` |
| `src/tqnet_physics_residual_kv_all/` | Generic TQNet and physical residual-KV training | `train_tqnet_base_generic.py`, `train_tqnet_physicskv_generic.py` |
| `src/tqnet_physics_residual_kv_pems08_final/` | Paired PEMS08 TQNet and PHRM reference run | `train_tqnet_base_70.py`, `train_tqnet_physicskv.py` |
| `src/tqnet_physics_residual_kv_pems08/` | Earlier PEMS08 training variant | `train_tqnet_base_70.py`, `train_tqnet_physicskv.py` |

## Restored project-local modules

The prepared tree includes eight utility and dispatch modules required by imports in the supplied scripts: two `utils.timefeatures` modules, diagnostics support for physical-support validation, and GPU dispatch helpers for the evidence and three-claim queues. `scripts/check_package.py` checks Python syntax, local imports, prohibited path names, and the checksum manifest without requiring training dependencies.
