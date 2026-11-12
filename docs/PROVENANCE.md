# Source provenance

The selected dataset source roots were copied from the user-provided server source bundle. Its file inventory contained 428 entries, and each recorded source SHA256 matched the corresponding transferred file before filtering. The retained sources are organized under `src/main_method/` by dataset; server-side directory names, dates, usernames, absolute paths, logs, and environment snapshots were removed.

`manifests/reference_artifact_hashes.csv` keeps only portable artifact names, dataset labels, checkpoint coordinates, sizes, and SHA256 values. It omits the original remote paths. The source inventory and repository checksum list can be regenerated with `python scripts/build_manifests.py`.

The repository's `LICENSE` preserves the rights-status notice supplied with the source package. It does not grant redistribution rights for third-party sources or data. Confirm ownership and third-party terms before redistributing the repository.
