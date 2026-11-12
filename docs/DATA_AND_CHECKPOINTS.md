# Data, graph and checkpoint inputs

No raw datasets, graph files, pretrained checkpoints, generated results, or logs are included. Obtain each input from its legitimate source and follow its terms. Never put large data or checkpoint binaries into Git; keep a separate, access-controlled asset directory and record SHA256 values in the run manifest.

## PEMS08 strict70 profile

The top-level runner expects this layout under `--project-root`:

```text
dataset/PEMS08_data.npz
dataset/physical/PEMS08_physical_candidates.json
```

The NPZ must include `data` shaped `[time, 170, 1]`. The graph JSON must include `edges`, a list of groups of zero-based node indices in `[0, 169]`. The runner trains the paired base checkpoint itself, then passes that exact checkpoint to PHRM. No pre-existing checkpoint is needed for this profile.

## Dataset-parameterized main method

The generic main-method scripts under `src/tqnet_physics_residual_kv_all/` accept a dataset NPZ, a physical edge CSV, and a trained TQNet checkpoint. The NPZ must contain a `data` array shaped `[time, nodes, channels]`. Supply the dataset-specific node count, forecast horizon, cycle, seed, and chronological split explicitly. `train_tqnet_base_generic.py` creates the paired TQNet checkpoint; `train_tqnet_physicskv_generic.py` consumes that exact checkpoint and writes the PHRM result and adapter.

The source archive does not include complete table-specific manifests or all input graphs, checkpoints, and third-party baseline implementations. The package therefore provides the main-method code and a runnable PEMS08 reference profile, while full reproduction of every reported comparison requires restoring and hashing the matching inputs and baseline sources.
