# Data, graph and checkpoint inputs

No raw datasets, graph files, pretrained checkpoints, generated results, or logs are included. Obtain each input from its legitimate source and follow its terms. Never put large data or checkpoint binaries into Git; keep a separate, access-controlled asset directory and record SHA256 values in the run manifest.

## PEMS08 strict70 profile

The top-level runner expects this layout under `--project-root`:

```text
dataset/PEMS08_data.npz
dataset/physical/PEMS08_physical_candidates.json
```

The NPZ must include `data` shaped `[time, 170, 1]`. The graph JSON must include `edges`, a list of groups of zero-based node indices in `[0, 169]`. The runner trains the paired base checkpoint itself, then passes that exact checkpoint to PHRM. No pre-existing checkpoint is needed for this profile.

## Other archived source roots

The multi-dataset and later evidence snapshots read additional raw datasets, physical edge/metadata files, processed splits, checkpoints and result references from the directory layout selected by `PHRM_PROJECTS_ROOT` or script arguments. Some evidence scripts also use external baseline implementations. Those trees and their exact revisions were not present in the supplied archive. Check the `SPECS`, `settings.py`, CLI arguments, and `protocol.py` in the target source snapshot; paths from another snapshot are not interchangeable.

The paper names the official GTR implementation at commit `d941619` and RAFT at commit `bfe2320` for its comparisons. These third-party implementations are not included here. Restore the relevant upstream code at the stated revision and retain its license notices before running the corresponding comparisons.
