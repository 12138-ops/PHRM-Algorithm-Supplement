# Source provenance

## Code source

The dataset-specific source roots in `src/main_method/` were selected from the project source bundle transferred by the repository owner. The transfer audit recorded 428 source-bundle entries and reported matching SHA256 values for the transferred files before the seven main-method roots were selected. Ablation, sensitivity, diagnostic, and exploratory experiment paths were excluded.

The original 428-entry source-bundle inventory is not distributed here. The 428-file transfer comparison is therefore a recorded packaging check, not a comparison that a reader can independently repeat from this repository alone. The repository does provide a hash inventory for the included source files in [`../manifests/source_inventory.csv`](../manifests/source_inventory.csv), and a checksum list for the packaged repository snapshot in [`../manifests/SHA256SUMS.txt`](../manifests/SHA256SUMS.txt).

## Data and checkpoint source records

[`../manifests/reference_artifact_hashes.csv`](../manifests/reference_artifact_hashes.csv) records portable paths, byte sizes, dataset labels, checkpoint coordinates, and SHA256 values for the 14 raw-data/physical-support assets and 84 paired TQNet baseline checkpoints in the experiment record. The repository contains all 14 data/support assets and the 12 Electricity checkpoints. The remaining 72 historical checkpoints are not included.

The original server paths, account names, credentials, server logs, and environment records are deliberately absent. The hashes establish whether a local file matches the recorded artifact; by themselves they do not establish the legal license or ownership of that artifact.

## Paper attribution

The accompanying manuscript identifies Jiaman Ding, Junhui Jiang, Lianyin Jia, and Qingmeng Hu as its authors. [`../CITATION.cff`](../CITATION.cff) records that author list and the manuscript title for citation. No DOI or final publication record is asserted in this snapshot; update the citation file when an archival publication record is available.

## Repository snapshot integrity

`manifests/SHA256SUMS.txt` lists SHA256 values for the packaged files, excluding the checksum file itself. Rebuild the list after an intentional repository change and verify the current snapshot with:

```bash
python scripts/build_manifests.py
python scripts/verify_checksums.py
```

The generated `data/run_manifest.json` records code, input, and checkpoint hashes together with software and device details for an actual run. Preserve it with the per-cell results when reporting a reproduction.

## Evidence limits

Checksums help detect changes relative to a recorded manifest; a manifest stored beside the files is not an independent timestamp or third-party ownership attestation. Git commit history records the author identities and dates present in Git, but an unsigned commit does not independently prove legal authorship. For a persistent, citable snapshot, enable the repository's Zenodo integration and archive a GitHub Release. See [Zenodo's GitHub release archiving guide](https://help.zenodo.org/docs/github/archive-software/github-upload/). Add the resulting DOI to `CITATION.cff` when it exists.
