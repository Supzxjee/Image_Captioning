# Paired detector-box training diagnostic — R73–R75

Detector-box training helps relative to a GT control trained on the **same matched subset**, but does not justify replacing the previous full-GT verifier. This is a single seed42 development experiment, not a caption-quality result.

## Feature audit

Uploaded archive: `triplet_detector_training_features_v1_review.zip`.

SHA256: `c3fcf5d65a462392891f4efa8adbd33a8d5359598ad0d0e80e4c91e863ee449f`.

| Split | Reference pairs | Matched pairs | Coverage | Matched images | Positive labels retained |
|---|---:|---:|---:|---:|---:|
| Train | 1,776 | 1,141 | 64.25% | 616 | 755 / 1,089 |
| Dev | 489 | 310 | 63.39% | 128 | 196 / 298 |

Train misses: subject49, object557, both29. Dev misses: subject5, object167, both7. No cache issues reported; train/dev image overlap0. Frozen CLIP model/revision, feature order, finiteness, normalization, protocol/source hashes, unchanged labels, and original GT identity/boxes passed loader checks. Unknown labels remain masked; unmatched proposals are not labelled false. Class-aware IoU>=.5 matching transfers labels approximately and does not establish that every detector crop contains only one instance.

## Controlled training and common evaluation

Both heads use identical matched pair identities, text features, labels, seed42 initialization, 30epochs/lr.001, BCE plus .2 type-balanced actor/predicate ranking loss, and train-only threshold fitting. Both retain the frozen R50 proposer. GT control uses original GT crops/geometry; detector condition uses detector crops/geometry. The verifier excludes the union feature.

**The following dev comparison evaluates both trained heads on the same 310 detector-box pairs.** Comparing each head's native GT/dev and detector/dev numbers would confound training and evaluation boxes.

| Training boxes | Common detector-dev macro AP | Correct actor contrasts |
|---|---:|---:|
| GT control, matched subset | 0.707176 | 141 / 194 |
| Detector, matched subset | 0.736769 | 148 / 194 |

Gain: +0.029593 macro AP, +7 actor contrast wins. Only four positive dev examples each for eating and talking_on; these class APs are fragile. The actor metric enumerates all valid contrasts; the older matching summary helper caps some contrasts and reports190 rather than194.

## Frozen R55 detector transfer

Same364 matched pairs /286 images, 243 known positive labels, training-image overlap0. No training or threshold refit on this pool. Unknown labels are excluded from known-label precision.

| Method | Retained correct | Retained wrong | Unknown | Known precision | Conditional recall |
|---|---:|---:|---:|---:|---:|
| Frozen R50 proposer | 43 | 6 | 11 | 87.76% | 17.70% |
| Matched GT-control verifier | 29 | 3 | 7 | 90.63% | 11.93% |
| Matched detector-trained verifier | 30 | 2 | 8 | 93.75% | 12.35% |
| Previous full-GT verifier, seed42 (context only) | 35 | 1 | 8 | 97.22% | 14.40% |

Against its paired control, detector training keeps one additional correct claim and removes one additional wrong claim. Against R50, it still loses13 correct claims while removing4 wrong claims. The earlier full-GT head uses1,776 training pairs; its comparison is **not** a controlled box ablation. These reused matched-pair metrics are neither full detector proposal AP nor image-caption quality.

## Full100 regression cross-check

Both new heads select the same12 claims on8 images;92 images retain object-only prompts. Previously assistant-reviewed supported umbrella and held-frisbee claims are retained by both matched-subset heads. Neither restores the previously reviewed child-donut/held-frisbee examples removed by soft fusion. These are different instances from the retained held-frisbee example.

Neither selects any of the previously recorded contradicted tuples, including the background-spectator motorcycle tuple, but this is a limited regression cross-check of method-visible assistant assessments. It is **not** independent human annotation, complete selected-claim precision, or exhaustive recall. Ambiguous cake/pizza eating, merged motorcycle regions, and uncertain hot-dog object class remain. A right-officer/right-bike tuple outside the original reviewed union is also selected; it has no independent label here.

## Decision

Keep detector adaptation as a promising diagnostic branch; preserve the original full-GT seed42 head and thresholds. Do not deploy a full prompt cache or claim an improvement in captioning from this run.

The next justified training design is to retain all original GT supervision and add matched detector views, with a GT-only control using the same training schedule. This avoids discarding635 original training pairs. Multi-seed paired verification and a fresh independent relation audit are required before replacing the current verifier. No such mixed-view training has been performed in this result.

## Artifacts and reproducibility

GT-control head SHA256: `d878c1006d663ba2b22350f56bc1e08e0871bd4878712ffe3344b334024dff31`.

Detector head SHA256: `f6c2c81adb3f43f75521f2878b1ef646f8cc31016147b701d26f803b29fa90c7`.

Training: `run_triplet_verifier_cached.py --detector-source <uploaded ZIP> --train-condition gt_control|detector`, with the same V2/R63/full100 feature sources and frozen R50 full100 scorer used previously. Transfer: `evaluate_triplet_transfer_cached.py`. Common detector-dev evaluation and prior-assessment membership checks: `summarize_triplet_detector_training.py` (all paths explicit).

Local paired report: `C:/KhoaLuanTotNghiep/Code/output/triplet_detector_training_paired_v1/paired_detector_training_summary.json`.

Compact review ZIP: `C:/KhoaLuanTotNghiep/Code/output/triplet_detector_training_paired_v1_review.zip`, containing both heads, training histories/prompt caches, frozen transfer summaries, source coverage metadata, paired report, and result document. Frozen feature archives are not duplicated. Kaggle version number was not supplied.
