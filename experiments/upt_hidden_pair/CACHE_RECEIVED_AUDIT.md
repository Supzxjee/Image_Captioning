# Received UPT hidden-pair cache — 21 October 2026

**Artifact audit PASS; ready for user-run A/H caption pilots after mounting the complete cache.** No caption metric or semantic improvement has been measured.

Both workers completed5500images, totaling the locked train10000/development1000cohort with no duplicated COCO IDs. Runtime and completion digests agree on the frozen UPT model. Source hashes, cohort file and both JSONL file hashes match the receipt. All11000JSONL rows match exact planned rank order and report classifier hook/raw-logit parity plus native pair order checks. Selected endpoint IDs, labels, confidences>.2, finite positive-area boxes, native row identity, counts and locked sort order pass audit. The returned notebook log contains5data tests,3hook tests and1HDF5 test passing before extraction.

| Split | Images | With eligible selected pairs | No eligible selected pair |
|---|---:|---:|---:|
| Train |10000|6423 (64.23%)|3577|
| Development |1000|654 (65.4%)|346|

This coverage is proposal availability under the locked human→nonhuman and endpoint-confidence policy, not relation-truth accuracy. An empty image receives no fabricated interaction token. Full native UPT attention still precedes filtering.

The fixed32-image human caption-review cohort can now be determined without observing any caption outputs:16with eligible pairs and16without, excluding known phrase IDs. `CACHE_RECEIVED_AUDIT.json` retains these IDs/counts. Both training notebooks deterministically recreate the same selection and save its hash before training.

Preserve the whole `/kaggle/working/upt_hidden_pairs_11k` folder as a Kaggle dataset. Required payload: `features_rank0.h5`, `features_rank1.h5`, `pairs_rank0.jsonl`, `pairs_rank1.jsonl`, `cohort.json`, `cache_receipt.json`. Keep remaining provenance/logs too. The small review ZIP intentionally excludes HDF5 and cannot replace this dataset. A/H notebooks at commit`ebe77d7` are source-policy compatible with the received extraction; set `HIDDEN_CACHE` to the actual folder containing `cache_receipt.json`. Each arm uses2T4 and can run on a separate account once all private datasets are available to both accounts.

Limits: HDF5 bytes and original JPEGs were not supplied in this ZIP. File/feature hashes cannot independently be recomputed here, omitted native pairs cannot be re-ranked, and metadata is not proof of semantic correctness. Training performs the mounted-file hash, row ID, shape, padding, finiteness and extraction-source checks before fitting. No inference, notebook or training was executed locally during this artifact audit.
