# Evidence and citation guide

This repository keeps evidence in forms that can be checked against the packaged files and the Git history. These records improve traceability; they do not prevent copying or replace an independent archival record.

## Verify this repository snapshot

From the repository root, run:

```bash
python scripts/verify_checksums.py
```

The command checks every path and SHA256 value in `manifests/SHA256SUMS.txt`. The list covers the packaged source, configuration, documentation, figures, raw data, physical supports, and included Electricity checkpoints; it excludes itself so it can be regenerated. The dedicated `manifests/source_inventory.csv` provides source-file sizes and hashes. Reference hashes for the 98 recorded data/support/checkpoint artifacts are in `manifests/reference_artifact_hashes.csv`.

Before training, run the input preflight as described in the README. For a completed reproduction, retain `data/run_manifest.json`, all per-cell `result.json` files, logs, and the reported environment details together. A run manifest is created by the experiment runner and is not included as if experiments had already been run.

## Cite the work

GitHub reads [`../CITATION.cff`](../CITATION.cff) to show a citation for the repository. The file lists the manuscript authors and does not invent a DOI. If the manuscript receives a DOI or final venue information, update the preferred citation before release.

[`AUTHORSHIP_RECORD.md`](AUTHORSHIP_RECORD.md) indexes the existing commits on the inspected main-branch snapshot that carry the repository owner's Git author field. The commit links point to the original GitHub records; the index does not change author metadata.

When a stable repository version is ready, create a GitHub Release from a reviewed commit and enable the repository in Zenodo. Zenodo can archive new GitHub releases and assign a DOI. Add the version DOI to `CITATION.cff` and the README after Zenodo confirms the record. See [Enable a repository](https://help.zenodo.org/docs/github/enable-repository/) and [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/).

## Keep attribution records accurate

Keep the existing Git history and its recorded author identities. Do not amend or recreate earlier commits just to change their displayed author. For future releases, use an annotated, signed tag when the maintainer's signing key is available, and preserve the release archive and DOI. Do not publish passwords, private server paths, or private environment logs as provenance evidence.

## Known limits

The checksum files are stored in the same repository as the files they describe, so they detect accidental or unrecorded changes but are not a third-party timestamp. The source transfer comparison against the original 428-entry inventory is described in [`PROVENANCE.md`](PROVENANCE.md); that original inventory is not part of this repository. The license notice in `LICENSE` does not itself grant a general license to reuse the bundled code or third-party assets.
