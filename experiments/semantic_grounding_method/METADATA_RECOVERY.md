# Missing gain_metadata.json.gz in attached preparation dataset

2026-10-27. User-run C/H stopped in header validation before model tests/training because `gain_metadata.json.gz` was not a file at the dataset root. Dataset processing may have expanded the gzip or changed its representation; no directory listing was supplied, so that cause is not confirmed.

The byte-exact original gzip already exists in the pinned project as `METADATA`, verified against `METADATA_SHA`. The compatibility helper accepts recovery only when both receipt metadata hashes match this pinned asset. It does not recompress a JSON file or guess targets. An existing corrupt metadata file or another missing payload is still a hard failure.

On recovery, create a fresh local cache view: link the HDF5 rather than copying it, copy the small unchanged original files and receipt, and restore the exact pinned gzip. Preserve the read-only dataset. Normal full payload/source/policy verification still runs before training. A provenance sidecar records the original dataset and recovery source.

C/H and T notebooks call the helper automatically. Their MODEL/runtime source manifest remains identical to the completed preparation; helper pinning is separate, so no repeat preparation is required. Head initialization, labels, losses, sample order, caps and frozen A are unchanged.

Local checks: four synthetic restoration/guard tests passed. Windows sandbox denied native hardlinks/symlinks; link calls were mocked on tiny fixtures, including cross-device fallback. Actual Kaggle link creation and model/GPU execution remain user-run checks.
