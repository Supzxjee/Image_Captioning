# Separate action support and instance attachment: R100–R101 development pilots

The factorized prototype is implemented and has completed image-group cross-validation on the existing human-reviewed contrasts. The first absolute-score version exports no supported claims. A subsequent relative-target version exports a very small useful subset, but neither learned target arm improves ranking over the frozen CLIP prior. No caption checkpoint is replaced.

## What changed

`captioning/factorized_evidence.py` separates two questions:

1. **Action support:** `ActionSupportHead` receives only the subject crop. Its decision is identical across candidate targets of the same source actor. Supervision is aggregated from original V-COCO actor labels, without treating an unlinked target as an absent action. Action fit uses671 actors on the R99 fit images; calibration uses198 actors on the R99 calibration images. All61 review-manifest images are excluded from both.
2. **Conditional target support:** a126-parameter head receives frozen subject/text, object/text, union/text similarities plus pair/geometry evidence. It adds a zero-initialized correction bounded by0.5 logits to a fixed CLIP prior. It is trained only on explicit reviewed supported/contradicted target endpoints. Unknowns are not negatives, and two supported targets do not compete through an exclusive softmax.

The learned control uses image-weighted target BCE. The matched second arm adds0.2 image-weighted logistic loss comparing confirmed true/false targets of the same image, subject and predicate. Architecture, annotations, initialization, schedule, calibration and candidate cap are otherwise identical. The prior-only reference receives no human training labels but uses the same held-out calibration protocol, so differences from it cannot isolate architecture from the benefit of labels.

Joint support requires BOTH the action and target branches to pass their own thresholds. The probability product is a cap-priority score, not a demonstrated calibrated joint probability. Action rejection can incorrectly discard a true relation; target ranking alone also cannot establish action truth.

## Image-group protocol

Use the34 eligible contrasts in22 images already provided by the user:21 same-class and13 different-class substitutions. No new reviews were requested. The10 source conflicts,4 both-supported cases,2 unclear-localization cases and75 incomplete cases do not become new negatives.

Five outer folds group by image, so riding/sitting_on or other cases from one image never cross fit/calibration/test within the same model. Each outer training pool has a further image-disjoint calibration subset. Across folds, fit/calibration/evaluation case counts are21/7/6,16/5/13,23/7/4,21/7/6 and18/11/5. Every reviewed case is evaluated out of fold exactly once; this is34 evaluations, not170 independent cases.

Action training:30 epochs, AdamW lr0.001. Target training:30 epochs, AdamW lr0.01. Weight decay0.01, gradient cap1, seed42. No best epoch/fold/seed selection. Each target calibration fold fits one shared temperature and positive/negative tails across predicates, requiring observed90% known-label precision and at least5 known claims in a tail. These small folds do not support a per-predicate calibration guarantee.

All images and labels remain previously inspected development data. Cross-validation here tests this head-fitting protocol; it does not create a new independent benchmark or erase previous method-development reuse.

## R100: absolute target scores

The action/target split by itself does not make absolute target scores transferable across image contexts. No positive tail satisfies the declared calibration rule in any target fold, in any of the three arms. Consequently all arms export zero joint supported claims. Precision is undefined, not100%.

| Out-of-fold ordering | Frozen CLIP | Conditional BCE | Conditional BCE + ranking |
|---|---:|---:|---:|
| Correct target above wrong target,34 cases | 28 | 28 | 28 |
| Same-class targets,21 cases | 18 | 17 | 17 |

The first run is retained rather than hidden or repaired by lowering thresholds after evaluation.

## R101: relative target evidence

After observing R100, a follow-up hypothesis changes the reference for target support. For each detector-derived subject and predicate, subtract the mean frozen CLIP prior across its unique candidate targets, THEN add the learned bounded correction:

`target_logit = CLIP_prior(target) - mean(CLIP_prior(candidate_targets)) + correction(target)`

Grouping uses filename, subject label and normalized detector box; target deduplication uses label/box. Annotation IDs and truth labels are not used to form the reference. The frozen reference has no classification gradient. This allows multiple targets to score positively, does not label other candidates false, and preserves prior within-group ordering. It is not a forced best-one selection. Results remain conditional on this existing matched-candidate pool; changing the candidate pool changes the reference, and single-candidate groups provide no relative prior advantage.

The image folds, schedules,90%/5-claim calibration rule and maximum3 claims/image remain unchanged. This is an adaptive development follow-up prompted by R100, not an independently preregistered confirmation. No third threshold/model sweep was performed.

| R101 out-of-fold result | Relative CLIP | Conditional BCE | Conditional BCE + ranking |
|---|---:|---:|---:|
| All target rankings correct | 28/34 | 28/34 | 28/34 |
| Same-class target rankings correct | 18/21 | 17/21 | 17/21 |
| Reviewed true / false target claims passing target-only rule | 1 /0 | 5 /0 | 5 /0 |
| Reviewed true / false after action AND target and cap | 0 /0 | 4 /0 | 4 /0 |
| Reviewed positive recall after joint selection | 0% | 11.76% | 11.76% |

The four retained reviewed claims are riding/sitting_on for two images, all from ONE outer fold. They are not four independent successes or evidence of reliable100% precision. The learned arms select five claims across the complete matched candidate pool: four human-supported and one not human-reviewed (its original source label is positive). That fifth claim is NOT counted as human-confirmed truth. Most candidate claims remain unevaluated by the human.

The action head supports15 of34 reviewed true actor/action claims, predicts contradiction for6 and abstains on13. This pool has only action-positive review examples, so action precision is not measured. The subject-only action crop is a limited prototype: the outcome does not prove that action/target factorization in general is ineffective.

## Interpretation and decision

- Relative evidence enables a small nonempty selected subset under this fixed rule. It does not improve target ordering, solve holding, establish reliable precision or establish caption gains.
- Added target ranking has no incremental ordering/selection gain over the same-label BCE control in this pilot. Do not claim that the contrastive term is a validated contribution.
- Shared global target calibration may suppress useful classes or be unstable. Small reviewed pools, limited frozen similarity features and the restrictive subject-only action head are separate possible limitations; these runs do not isolate a single cause.
- Keep the original CLIP prior and caption baseline. Both pilots remain development artifacts; the decoder evidence gate is still not caption-trained. No CIDEr/CHAIR or novelty claim follows.

A next evidence change should be tested against these fixed baselines, with actor-centric scene context and explicit target-local evidence rather than simply more epochs or a larger loss coefficient. The previous shared-union region pilot remains relevant prior negative evidence; this result is not a reason to repeat it unchanged. Further evaluation must report coverage and per-image concentration, not only precision on selected claims.

## Verification and artifacts

22 relevant tests passed, covering action independence from target/union crops, unknown handling, initial prior parity and bounded corrections, nonexclusive supported targets, nested image separation, conjunction/abstention semantics, relative-score shift invariance and detector-box deduplication. All saved action/target heads reload strictly. Their calibration temperatures/policies and out-of-fold logits, decisions and capped selections were recomputed exactly without retraining; metrics match the saved results.

Code: `captioning/factorized_evidence.py`, `run_factorized_evidence_cv.py`. Public aggregates: `audit/factorized_action_target_cv_v1_result.json`, `audit/factorized_action_target_relative_cv_v2_result.json`. Local models/protocols/private predictions remain in `C:/KhoaLuanTotNghiep/Code/output/factorized_action_target_cv_v1` and `C:/KhoaLuanTotNghiep/Code/output/factorized_action_target_relative_cv_v2`. No new dataset/model download or GPU run was needed.

Reproduce R101 on CPU:

```powershell
python run_factorized_evidence_cv.py --relative-targets --source "C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip" --package "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1" --review "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1/human_batch1_review.json" --r99-directory "C:/KhoaLuanTotNghiep/Code/output/reviewed_target_evidence_v1" --output "C:/KhoaLuanTotNghiep/Code/output/factorized_action_target_relative_cv_v2"
```

For R100 omit `--relative-targets` and use the separate `factorized_action_target_cv_v1` output. Protocol checks reject accidental changes in an existing output directory. Raw human identity/notes and per-image case labels are not copied into the public aggregates.
