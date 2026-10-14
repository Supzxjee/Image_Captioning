# Full-GT plus detector-view training — R76–R77

Adding detector views raises development AP slightly, but loses more confirmed relations on the reused detector transfer pool. The mixed head is **not promoted**. The previous full-GT seed42 verifier and frozen R50 proposer remain unchanged.

## Method and control

All1,776 original GT training pairs /826 images are retained. An additional1,141 matched training views are supervised using unchanged labels: GT crops/geometry for the control, detector crops/geometry for the mixed condition. This control holds matched-example weighting and training schedule fixed rather than comparing different-sized datasets.

Objective: `0.5 * full_GT_loss + 0.5 * matched_view_loss`. Each block has independent confirmed actor/predicate contrasts; there are no cross-view fake negative edges. Per-block loss remains known-label BCE plus .2 type-balanced ranking with margin1logit/temperature1. Unknown labels stay masked.

Both conditions use seed42, the same initialization,30 full-batch AdamW steps (lr.001,weight_decay.01,clip_norm1), the same candidate-specific frozen CLIP text, and the354,433-parameter union-free TripletVerifier. This experiment adds view supervision, not a new architecture or consistency loss.

Both heads fit thresholds on the **same1,141 matched training detector crops** with the existing >=90% empirical training precision / >=5 accepted examples rule. Thus GT-only control training does not use detector crops in its gradients, but does use them for train-only calibration. No dev/transfer/full100 labels are used for optimization or threshold fitting. Sigmoid outputs are not calibrated probabilities.

Source/configuration is locked in `audit/triplet_mixed_view_protocol.json`. Uploaded detector archive SHA256: `c3fcf5d65a462392891f4efa8adbd33a8d5359598ad0d0e80e4c91e863ee449f`.

## Common detector validation

Same310 detector-box dev pairs /128 images; zero training-image overlap. Dev includes only4 positive examples each for eating/talking_on. This is reused development data.

| Training condition | Macro AP on known labels | Correct actor contrasts |
|---|---:|---:|
| Full GT + matched GT control | 0.748412 | 149 /194 |
| Full GT + matched detector views | 0.761829 | 149 /194 |

AP increases0.013417; actor ordering does not improve.

## Frozen detector transfer

Same R55matched364pairs /286images,243known positives; zero training-image overlap. Frozen proposer and thresholds, no evaluation refit. Unknown labels excluded from precision. Conditional recall is against known positives in the matched pool, not all image relations.

| Method | Correct kept | Wrong kept | Unknown | Known precision | Conditional recall |
|---|---:|---:|---:|---:|---:|
| R50 proposer | 43 | 6 | 11 | 87.76% | 17.70% |
| Full GT + matched GT control | 33 | 2 | 8 | 94.29% | 13.58% |
| Full GT + matched detector views | 28 | 1 | 7 | 96.55% | 11.52% |
| Previous full-GT seed42 verifier (context) | 35 | 1 | 8 | 97.22% | 14.40% |

Relative to its paired control, mixed training removes one additional wrong claim but loses five additional correct claims. Relative to R50, it removes5 wrong claims and loses15 correct claims. Whole-verifier transfer macro AP also falls slightly,0.800881→0.796925, before applying the proposer gate. The loss of true claims is therefore not demonstrated to be exclusively a threshold issue.

The previous full-GT head has different matched-example weighting and calibration, so it is context rather than a controlled augmentation ablation. This run does not outperform it on retained positive/negative counts. These scores are not full detector proposal AP or caption metrics.

## Full100 regression check

Control selects12claims on8images; mixed selects11 on7images. The only selected-tuple difference is loss of `COCO_train2014_000000040423.jpg`: `det_23eab7c1d521 holding det_1f30b9e1e626` (umbrella), previously marked supported by assistant visual inspection. No new tuple is added.

Both retain uncertain cake/pizza eating, a merged motorcycle region, and uncertain hot-dog class. Both avoid the previously recorded contradicted tuples; the child-donut and other lost held-frisbee examples are still not restored. This cross-check reuses method-visible assistant assessments; it is not independent human annotation or exhaustive relation recall. Unknown/new tuples cannot be counted as correct from non-membership in the error list.

## Validation and decision

An identical-view synthetic fixture (32extra training /8dev pairs) produces **exactly identical model tensors and thresholds** for both new conditions and verifies preservation of all1,776 original GT rows. Real source/hash/label checks, finite-loss training, common-feature evaluation, transfer identity/head checks, and `git diff --check` passed. No synthetic score is reported as real model quality.

Do not replace the full-GT verifier or generate a full prompt cache from this head. Equal-weight detector augmentation is not supported as a solution to true-relation retention by this fixed-seed trial. Preserve the negative result without tuning epochs, loss weights, or thresholds against these inspected pools.

The next useful diagnosis is to distinguish true relations blocked by the frozen proposer from true relations rejected by the verifier, using already cached known-label scores. This can locate the remaining recall bottleneck before adding further model complexity. This result contains no new caption training or claim of scientific novelty.

## Reproduce and artifacts

Use the same source arguments as `run_triplet_verifier_cached.py` in the paired detector run, with `--train-condition gt_augmented_control` and `--train-condition mixed_detector`, both retaining `--detector-source`. Run `evaluate_triplet_transfer_cached.py` against each new head, then `summarize_triplet_detector_training.py --augmented` for common dev evaluation and review membership checks.

Control head SHA256: `53ca9308748ea27097eb14f2c069d0ab40a28f527601ecd13ae60f0af5fdb5ed`.

Mixed head SHA256: `1b99fff6d0b94e5fb0faf114f675218b9d433e531bdc5327ad455e896a832e47`.

Local output: `C:/KhoaLuanTotNghiep/Code/output/triplet_mixed_view_v1`. Compact review: `C:/KhoaLuanTotNghiep/Code/output/triplet_mixed_view_v1_review.zip`. Both heads, histories, prompt caches, candidate activations, transfer summaries, paired JSON, protocol and result document are included; frozen feature ZIPs are not duplicated.
