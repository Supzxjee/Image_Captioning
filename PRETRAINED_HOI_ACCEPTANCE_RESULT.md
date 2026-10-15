# Frozen UPT: calibrated relation acceptance

This audit measures acceptance of individual instance-bound relations after choosing thresholds on separate images. It extends the previous target-ordering diagnostic; it does not measure caption quality.

**Outcome:** learned interaction scores provide substantially better observed ranking on this reused development pool: macro AP **0.805965 versus 0.488215** for CLIP on the same teacher-available cells. Absolute acceptance remains imperfect: **80 true / 16 false accepted**, precision **83.33%**, retaining **40.82%** of all 196 known positives. CLIP finds no calibration tail meeting the fixed 90%-precision/minimum-five rule, so its acceptance precision is undefined; this does not mean CLIP recognizes no relations.

The error pattern is specific: `hold` accepts **63 true / 5 false** (92.65%), while `sit_on` accepts **17 true / 11 false** (60.71%). Selecting only `hold` after seeing this would be a new, post-hoc rule requiring a separate evaluation. Four further development claims accepted by raw UPT have unknown labels.

Unchanged source thresholds accept **13 true / 0 false** among the 68 human-labeled claims, retaining 13/34 supported claims. Another eight accepted row/predicate cells lack human labels and are not established correct. The practical next experiment is caption-candidate reranking using this interaction evidence alongside the historical fixed-reranking baseline, with scoring choices selected on validation and frozen before test. The evidence supports that experiment; it does not establish a new caption result or a ready-to-use six-predicate prompt generator.

## Protocol fixed before inference

- Frozen official UPT-R50 HICO-DET checkpoint, SHA256 `12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c`. No local model fitting.
- 111 calibration images / 187 candidate pairs; 128 development images / 310 candidate pairs. Reuse the R106 image split. All 61 human-manifest images excluded from both.
- Per-predicate acceptance threshold: lowest observed calibration tail with at least 90% known-label precision and at least five known accepted claims. Both known outcomes required; otherwise abstain. No temperature fit. Calibration precision is empirical, not a guarantee.
- Class-consistent one-to-one maximum-total-IoU matching, IoU >=0.5. Relation labels are not used for matching. UPT additionally requires a matched pair and a trained HICO object/verb combination.
- Unknown labels stay unknown. Unavailable evidence abstains and remains in the denominator of full-pool positive recall. Never treat a missing detection as a negative relation.
- Each arm selects its own threshold on calibration only. The masked CLIP arm calibrates and evaluates on exactly the teacher-available cells.
- Six predicates: holding, riding, sitting_on, carrying, eating, talking_on_phone.

## Development acceptance

| Method | True accepted | False accepted | Known precision | Recall over all known positives | Available positives | Known AP, macro | Accepted unlabeled claims |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP, all source pairs | 0 | 0 | undefined (no accepted known claims) | 0.00% | 196/196 | 0.1729 | 0 |
| CLIP, UPT availability mask | 0 | 0 | undefined (no accepted known claims) | 0.00% | 170/196 | 0.4882 | 0 |
| UPT learned verb score | 80 | 16 | 83.33% | 40.82% | 170/196 | 0.8060 | 4 |
| UPT official native score | 63 | 13 | 82.89% | 32.14% | 170/196 | 0.7672 | 4 |

AP compares ranking without using an acceptance threshold. Compare UPT AP against **masked CLIP AP** for the same subset. CLIP-all AP is over a different set. Accepted unlabeled claims are not verified as either true or false and do not count in precision.

The candidate pool comes from the existing detector/source audit. Full-pool recall here is recall over its known positive pair/predicate cells, not recall of every relation in an image or a standard HICO benchmark score.

## Per-predicate development counts

| Predicate | Known positives | UPT-available positives | CLIP all true / false accepted | UPT raw true / false accepted | UPT native true / false accepted |
|---|---:|---:|---:|---:|---:|
| hold | 119 | 108 | 0 / 0 | 63 / 5 | 47 / 3 |
| ride | 24 | 24 | 0 / 0 | 0 / 0 | 0 / 0 |
| sit_on | 27 | 17 | 0 / 0 | 17 / 11 | 16 / 10 |
| carry | 18 | 14 | 0 / 0 | 0 / 0 | 0 / 0 |
| eat | 4 | 4 | 0 / 0 | 0 / 0 | 0 / 0 |
| talk_on | 4 | 3 | 0 / 0 | 0 / 0 | 0 / 0 |

## Unchanged threshold transfer to existing human target reviews

Exploratory transfer to 34 previously inspected contrasts / 22 images. Deduplicate repeated row/predicate appearances: 68 unique labeled claims. Thresholds are copied unchanged from source calibration; no human labels are used to fit them. All review images are disjoint from source calibration and development.

| Method | True accepted | False accepted | Known precision | Recall over all human-supported claims | Available supported claims | Both contrast endpoints accepted |
|---|---:|---:|---:|---:|---:|---:|
| CLIP, all source pairs | 0 | 0 | undefined (no accepted known claims) | 0.00% | 34/34 | 0 |
| CLIP, UPT availability mask | 0 | 0 | undefined (no accepted known claims) | 0.00% | 27/34 | 0 |
| UPT learned verb score | 13 | 0 | 100.00% | 38.24% | 27/34 | 0 |
| UPT official native score | 10 | 0 | 100.00% | 29.41% | 27/34 | 0 |

Correctly ordering the supported target above the alternate does not establish that either passes an absolute acceptance threshold. This transfer checks that distinction. Human contrasts are from a reused, single-reviewer development pool and do not constitute independent validation.

## Verification and limits

- 239 cached image predictions hash-verified and re-matched; all pair scores and availability masks replay exactly.
- 113,168 native output cells reproduce upstream vectorized sigmoid-times-prior scores bit-exactly, including pair and verb order.
- Calibration-only policies, decisions, development counts and tie-aware AP replay exactly. Sixteen scoped tests passed, including unknown/unavailable handling and existing review filtering.
- Source negatives mean the annotated action is absent for the actor. Unlinked role objects remain unknown; these labels are not a complete wrong-target benchmark.
- The source development pool has been inspected in earlier trials. HICO relation pretraining is different from V-COCO supervision; COCO detector pretraining may include images, and exact web-image overlap was not audited.
- The previous 80% human-contrast coverage gate remains failed. This is a broader diagnostic, not a retroactive pass.
- No caption inference, CIDEr/CHAIR improvement, statistical superiority, or novel architecture is established here. Historical strongest reported caption pipeline remains Q-Former + fixed reranking, CIDEr 1.221815.

## Reproduce on this machine

```powershell
$py = 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = 'C:/KhoaLuanTotNghiep/Code/.tmp/upt-deps;C:/KhoaLuanTotNghiep/Code/.tmp/patch-deps;C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
Set-Location 'C:/KhoaLuanTotNghiep/Code/github_repo'
& $py run_pretrained_hoi_acceptance.py --upt C:/KhoaLuanTotNghiep/Code/.tmp/upt-reference --pocket C:/KhoaLuanTotNghiep/Code/.tmp/pocket-reference --checkpoint C:/KhoaLuanTotNghiep/Code/.tmp/upt-r50-hicodet.pt --source C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip --partition C:/KhoaLuanTotNghiep/Code/output/pair_query_attention_v1/protocol.json --images C:/KhoaLuanTotNghiep/Code/output/pair_query_patches_v1/images --output C:/KhoaLuanTotNghiep/Code/output/pretrained_upt_acceptance_v1
& $py verify_hoi_acceptance.py --output C:/KhoaLuanTotNghiep/Code/output/pretrained_upt_acceptance_v1 --source C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip --metadata C:/KhoaLuanTotNghiep/Code/.tmp/upt-reference/hicodet/instances_train2015.json --package C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1 --review-output C:/KhoaLuanTotNghiep/Code/output/pretrained_upt_hico_target_audit_v1
```

Protocol SHA256: `b3a8c14f9850d272496ca31d6ed1d0ebd559f7e3cfc5d0b04492a87d8cb64408`. Frozen cache resumption verifies every reused file hash.
