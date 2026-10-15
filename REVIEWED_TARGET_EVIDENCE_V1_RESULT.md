# Reviewed wrong-target check: the new verifier is not ready for decoder promotion

## What was done

Reused the human's existing50 completed reviews; no additional annotation or GPU run was requested. The original filter yields34 eligible contrasts across22 images:21 same-class and13 different-class target substitutions. It excludes10 source-positive conflicts,4 both-supported cases and2 unclear-localization cases;75 cases remain incomplete. Those excluded/unknown cases were not silently labeled negative.

An overlap check found15 eligible images in R98's fit partition and7 in calibration. Therefore the frozen R98 checkpoint is reported as familiar-image diagnostic only. It is not an unseen-image target evaluation, despite never training on the new human negative labels.

A fresh verifier was trained from the same seeded initialization and30-epoch AdamW schedule as R98, excluding ALL61 images in the125-case review manifest from both fit and calibration. It uses699 fit pairs/418 images and209 calibration pairs/137 images. Human labels are used for evaluation only; none are used for training, temperature fitting, thresholds or epoch selection. The CLIP control uses this same isolated calibration partition. Source archive/annotation manifest/checkpoint hashes and exact row order were verified.

These images are excluded from this particular refit, but are still part of an already inspected development pool, not a new independent benchmark. One self-reported human reviewer and conditional source-positive confirmation limit the evaluation.

## Primary target ranking result

Ranking asks whether the correct target scores above the explicitly contradicted alternate, holding image, subject and predicate fixed.

| Reviewed contrasts | Frozen CLIP | Fresh isolated verifier |
|---|---:|---:|
| All34 cases | 28/34 (82.35%) | 27/34 (79.41%) |
| Same-class targets,21 cases | 18/21 (85.71%) | 15/21 (71.43%) |
| Different-class targets,13 cases | 10/13 | 12/13 |
| Holding,10 cases | 10/10 | 10/10 |

The aggregate classifier/selection gain from R98 does not establish improved instance binding. This stricter check finds worse same-class target ordering. Holding's10/10 ranking also does NOT mean holding is solved: only2 of these10 correct holding claims pass the new verifier's positive threshold. Ranking two alternatives correctly and deciding that an action is actually supported are different questions.

## Ranking and selection must be reported separately

The isolated CLIP policy exports no supported claims among the68 unique reviewed endpoints. Its precision is undefined, not100%; it cannot serve as a useful selector here despite relatively good pair ranking.

The isolated verifier accepts11 true and5 false reviewed claims before the cap (68.75% known precision). After the same maximum-three-claims-per-image cap over the complete matched candidate pool,9 true and2 false reviewed claims remain (81.82% precision,26.47% recall of34 reviewed positives). In5 reviewed contrasts it initially supports both the correct and wrong target. More accepted relations is therefore not evidence of correct attachment.

The familiar R98 checkpoint has29/34 ranking wins and17/21 same-class wins; after its cap it retains5 true/0 false reviewed claims. Its original fit/calibration image overlap prevents treating those numbers as a clean comparison with the refit. Tiny selection counts must not be advertised as reliable100% precision.

## Where ordering was lost

Compared with CLIP, the isolated verifier fixes3 cases and regresses4. The4 regressions occur in just2 images, with riding/sitting_on evaluated for the same pair in each image; they are not4 independent examples.

| Development image | Predicate | CLIP correct-minus-wrong gap | Learned gap |
|---|---|---:|---:|
| 392793, two riders/horses | riding | +0.04934 | -0.75079 |
| 392793 | sitting_on | +0.07290 | -0.68950 |
| 402041, multiple riders/elephants | riding | +0.02758 | -0.03250 |
| 402041 | sitting_on | +0.00999 | -0.05739 |

Saved-head recomputation decomposes these margins into the normalized CLIP prior plus the learned correction. The learned correction reverses the initially positive margin. This locates the reversal mathematically; it does not prove which input feature or training factor caused it. The raw images were inspected for scene context, while relation truth and exact instance assignment remain grounded in the user's pre-existing box-specific reviews. No new assistant-generated training labels were added.

All4 regression cases have both alternatives rejected or abstained after calibration, so these particular reversals do not directly create exported false prompts. The remaining2 capped wrong-target claims are a separate acceptance failure. Do not equate rank regressions, exported prompt errors and caption errors.

## Decision and next implementation target

Do not promote this verifier into caption training or claim improved target grounding. Preserve both original R98 and the isolated checkpoint for reproducibility; keep the current caption baseline.

The next change should separate actor/action support from target-specific attachment. A high action score must not automatically license every nearby instance. Target supervision must compare confirmed true and false targets for the SAME subject/predicate, with unknown alternatives masked and multiple supported targets allowed. Existing human labels can support a small image-disjoint cross-validation pilot; they cannot become both training data and an independent test. The previous region/target-hinge trial already showed that fitting these training contrasts alone does not establish transfer improvement, so another training-fit score is not the acceptance criterion.

Before any decoder adaptation, require an explicit paired comparison against the retained CLIP prior on same-class target ordering AND true/false/unknown prompt acceptance. Preserve coverage reporting so abstention cannot manufacture an apparent success. Decoder gate weights remain untrained, and no CIDEr/CHAIR change is claimed.

## Reproduce locally

```powershell
python run_reviewed_target_evidence.py --source "C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip" --package "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1" --review "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1/human_batch1_review.json" --r98-directory "C:/KhoaLuanTotNghiep/Code/output/selective_evidence_routing_v1" --output "C:/KhoaLuanTotNghiep/Code/output/reviewed_target_evidence_v1"
```

17 relevant tests pass, including duplicate/identity rejection, unknown preservation, unique-endpoint counting, abstention vs ranking, wrong-target acceptance despite correct rank, image-level regression counts and existing evidence-router invariants. Metrics, calibration temperatures/policies and prior/correction margins were independently recomputed from saved heads without another training run.

Public aggregate: `audit/reviewed_target_evidence_v1_result.json`. Local protocol, checkpoint and case predictions: `C:/KhoaLuanTotNghiep/Code/output/reviewed_target_evidence_v1`. Raw human notes/images and case-level labels are not copied into the public aggregate.
