# Candidate TRAIN2000 feasibility passes the numeric screen

2026-10-25. Review SHA256 `0787edfd2a6c5b2b93c320513be0c1711b7a893181ec123329b72afccb8d4c61`.

Both T4 workers completed1,000 images each with unchanged exact A. No training took place. Saved source hashes, sample order, disjoint TRAIN/val IDs, raw-to-scored candidate identity, token eligibility, sealed support labels, all selected pairs and oracle aggregation pass the independent metadata audit. On-Kaggle synthetic sampling tests passed; greedy parity and frozen-A digests are reported by both workers. Model and actual stage tensors are not included in the small ZIP.

| Measure | Observed |
|---|---:|
| TRAIN images | 2,000 |
| Candidate captions | 10,000 |
| Valid candidates | 9,570 |
| Images with an eligible preference pair | 1,526 (76.3%) |
| Salient COCO classes represented in eligible-pair images | 67 |
| Images with a guarded candidate beating greedy | 773 (38.65%) |
| Greedy CIDEr under fixed TRAIN10k IDF | 1.330501 |
| Hypothetical guarded-oracle CIDEr | 1.509214 |
| Oracle improvement | +0.178713 |
| Greedy is the selected pair winner | 591 of1,526 pairs |
| Non-greedy is the selected pair winner | 935 of1,526 pairs |

The numeric gate required at least400 eligible pair images and guarded-oracle gain≥.05; both pass. The1,526 pairs must not all be described as corrections of greedy: many simply prefer greedy over a poorer sampled caption. Also,935 non-greedy pair winners do not imply935 improvements over greedy; the pair-selection rule and greedy-relative oracle safeguards differ.

Generation produced2 candidates without EOS and429 containing UNK before EOS; these conditions overlap in one case. All invalid candidates are retained in raw artifacts and excluded from pair construction. Mean unique caption strings/image is4.8945. No resampling or coefficient sweep was performed. Detector conflicts reported by preparation are0; this does not prove detector/annotation completeness or human correctness.

Preparation took650.28 seconds (~10m50s). The slower inference worker took93.33 seconds (~1m33s), excluding preparation, initialization and CPU scoring. This is not total notebook wall time or charged Kaggle GPU quota.

## Next decision

Proceed to the already-required blinded30-pair factual review, not automatically to training or a10k bank. The HTML is extracted locally as `review/review_30.html` (ignored, with a separate hidden key); original images load from COCO over HTTPS. The locked small-screen requirement is at least27/30 judgments favor the selected direction and no systematic parser/compound errors. Ties/unclear remain such, not counted as wins or overwritten. This small review cannot estimate population label accuracy.

After human checking, discuss whether to expand the bank and lock C/T, with U required for a later data-filter claim. C/U/T must share CE-winner captions and main reference streams; matched preference event/repeat counts do not by themselves remove all data-distribution differences. A learned raw-greedy gain is still unmeasured. The full10k bank has not been generated, so its≥2,000-image/≥15-class gate is not yet established from an extrapolation.

Novelty remains unresolved: Astra classifies this as an optimization baseline and a data-selection hypothesis, not a new visual–semantic alignment mechanism. Keep that distinction even if candidate headroom is positive.

Local work recomputed saved decisions/aggregates only. It did not execute a model, rerun CIDEr, reopen all original input assets or train anything. TRAIN oracle scores cannot be compared directly to val5000 metrics as a model improvement.
