# VSR relation-frame metadata inventory

28/10/2026. **METADATA_UPPER_BOUND_PASS_PENDING_GT_DETECTOR**.
This is a CPU-only data feasibility check, not a model experiment or a training authorization.

## Result

| Sequential filter | Records | Images | Within-statement one-to-one pair capacity |
|---|---:|---:|---:|
| Public VSR random-split corpus | 10,972 | 6,259 | 4,583 |
| At least three agreeing validators, no dissent | 8,657 | 5,344 | 3,010 |
| Sixteen spatial relations fixed by Astra | 4,802 | 3,387 | 1,813 |
| Exact template, two distinct COCO80 categories | 4,802 | 3,387 | 1,813 |
| Entire statement in A vocabulary | 4,781 | 3,370 | 1,807 |
| Only caption TRAIN/restval images | 4,399 | 3,103 | 1,584 |

No record was kept from caption val/test: 172/210 records respectively were excluded at the split filter. The full caption metadata SHA matches A's pinned source. Original VSR random splits share 1,400 image IDs across splits, so they must not be called image-disjoint. This inventory pools them for a separately declared derived experiment; it does not propose a score on the official VSR benchmark.

Split images, before pairing, with `sha256('vsr-frames-v1:'+str(coco_id)) mod 5`: bucket 0 held-out, others training. All statements of one image stay together. No model scores or caption test outputs are inspected.

| Derived split | Images | Records | Pair capacity | Pilot target |
|---|---:|---:|---:|---:|
| Train | 2,467 | 3,508 | 1,115 | 256 |
| Adapter-held-out | 636 | 891 | 116 | 64 |

Five relations meet both provisional per-relation counts (>=16 train and >=8 held-out):

| Relation | Train capacity | Held-out capacity |
|---|---:|---:|
| contains | 81 | 11 |
| on | 119 | 10 |
| on top of | 111 | 9 |
| touching | 292 | 43 |
| under | 123 | 10 |

These counts are **upper bounds**, before full instance uniqueness, detector class/IoU matches, capped candidate pools and any global image-reuse restriction. For each exact statement, capacity is `min(distinct positive images, distinct negative images)`, not their Cartesian product. Different statements can still reuse images within a split; pair counts are not independent-image counts. No conflicting labels for the same image/statement survive the current filters.

## Remaining prerequisite

The local COCO instance file covers only 5,400 VCOCO images. Of the 4,399 remaining VSR records, 4,242 have no image entry in that subset. Missing annotation is **unknown**, not a missing object or false assertion. Only 157 records currently have verified unique subject/object instances in the available subset. That partial coverage cannot determine the full-cohort GT/detector gate.

The metadata is promising enough to proceed to the complete annotation/box coverage check, but is **not** enough to start frame pretraining. Need full COCO instance counts and predicted-box matching, then the original fixed 256/64 and relation quotas. Feature/RGB availability and original-prompt generation for decoder probes also remain unresolved. No image download, detector/CLIP forward, model loading, training or model metric was performed.

## Reproduce

VSR source revision: `b27a0af0ee1462d2b6b92c8c83e869d9254a241a`; URLs, sizes and exact hashes are in `sources/source_receipt.json`. `fetch_sources.py` restores only those pinned JSONL files. Raw sources and intermediate candidates are ignored by Git.

Run `inventory.py --dataset <dataset_coco.json> --instances <official-instance-file.json>` after fetching sources. The current executed instance input was the local VCOCO subset, explicitly reported as partial. The script checks the caption hash, input hashes, duplicate COCO identities and image-disjoint partitioning; synthetic self-checks cover unanimity, dissent, repeated votes, multiword entity templates, same-category rejection and pair-count inflation. No neural library is imported.

This inventory does not validate originality, human-label correctness, grounding quality or caption improvement. Machine-readable detail is in `RESULT.json`.
