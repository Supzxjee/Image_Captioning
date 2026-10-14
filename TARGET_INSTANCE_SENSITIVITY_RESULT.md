# Natural target substitutions and supervision audit — R85

The new heads respond to different target instances, so they do not simply ignore the target. However, the current known-label supervision contains **zero confirmed positive-versus-zero target-swap contrasts** with image,subject,predicate held fixed. This is a consequence of preserving unknown alternative roles, not evidence that every alternative target is false or that data alone causes the observed errors.

## What was checked

Reuse all1,321full100 detector candidate scores from preserved full-GT TripletVerifier, union-CLS control and ordered-region head. No training, threshold fitting, image extraction or pipeline change.

Group by same image, subject instance and object class; require at least two target instances. There are257groups and1,955pairwise target comparisons. All have target-box IoU<.8 (consistent with the earlier same-class NMS); this geometric rule does not independently establish clean instance identities.

This is a **natural substitution diagnostic**: target box, union crop and region/geometry evidence change together. It is not an intervention that fixes the subject's encoded features or every other input. Same object class keeps the candidate predicate-object text the same, but crop context still varies.

Cached candidate identity/order and source-head hashes are checked. Sigmoid outputs are activations, not calibrated probabilities; distributions/scales differ across models.

## Holding sensitivity on the same266active target comparisons

The common subset contains every low-overlap pair where at least one model accepts at least one target under its own frozen training threshold. All three models are summarized on the same266comparisons.

| Model | Mean absolute score difference | Median absolute score difference |
|---|---:|---:|
| Preserved crop TripletVerifier | 0.059622 | 0.021415 |
| Union-CLS control | 0.655112 | 0.756352 |
| Ordered regions | 0.347623 | 0.281610 |

New heads exhibit larger activation changes on this subset. These uncalibrated magnitudes do **not** prove better target correctness. The union control is at least as responsive in this description; larger sensitivity is not a demonstrated advantage of region content.

Within each model's own active low-overlap holding comparisons:

| Model | Comparisons with at least one accepted target | Both accepted | One accepted, one rejected |
|---|---:|---:|---:|
| Preserved crop TripletVerifier | 66 | 54 | 12 |
| Union-CLS control | 243 | 21 | 222 |
| Ordered regions | 68 | 13 | 55 |

These rows contain different active subsets and cannot be compared as accuracy rates. Multiple accepted targets may both be valid; no negative labels are invented from this statistic. Across108active group/predicate comparisons for union-versus-regions, the top target changes15times, without a correctness label.

## Example: three pizza targets in image303429

Same subject`det_d641ee12784d`; each row contains its holding activation for the supplied pizza target.

| Target instance | Old triplet | Union control | Regions |
|---|---:|---:|---:|
| `det_e4eeb360d577` | 0.879 | 0.964 | 0.944 |
| `det_5dcb40c0fd50` | 0.893 | 0.966 | 0.961 |
| `det_d0974408d52a` | 0.834 | 0.215 | 0.265 |

Both new heads separate the third target from the first two; the old head remains high for all three. This illustrates sensitivity, not which pizza is truly held. The earlier assistant assessment did not establish exhaustive target labels or one uniquely correct target.

## Training supervision audit

For every confirmed positive training anchor, enumerate alternate target instances with the same image,subject,predicate. Count their original labels without rewriting them.

| Training cohort | Alternate-target comparisons | Label-0 alternatives | Label-1 alternatives | Unknown alternatives |
|---|---:|---:|---:|---:|
| Full GT pool,1,776pairs | 287 | 0 | 0 | 287 |
| Matched detector train,1,141pairs | 125 | 0 | 0 | 125 |

Existing full-GT training contrasts:736actor and4,408predicate. Matched detector training:457actor and3,041predicate. There are no confirmed target-swap negatives in these cohorts. Unknown alternatives therefore do not receive BCE/ranking penalties for target mistakes.

The preserved policy is intentional: an annotated actor action with a missing or different role target is not automatically proof that another target is false. Changing the125unknown alternatives to negatives would manufacture supervision and is not authorized by this diagnostic. Known actor-absent rows still provide target-conditioned classification examples; the model is not completely unsupervised about target features.

## Decision and next work

Do not add another architecture solely because target sensitivity was suspected absent. Preserve all heads and acceptance rules.

The next data/method experiment should obtain **verified positive-versus-negative target-instance examples from training images**, retain ambiguous alternatives as unknown, and then introduce explicit target-swap ranking with a no-target-loss control. Verification must be separate from the inspected full100 evaluation examples; absence from an annotation list or low geometric overlap is not enough to certify a negative. This dataset/target loss has not been created or trained in this diagnostic.

This identifies a supervision gap and tests a hypothesis; it does not prove scientific novelty, causal attribution of all errors, or improved captioning. All score comparisons remain on reused development candidates.

## Artifacts

Reproduce with `audit_target_instance_sensitivity.py --region-directory <paired region output> --triplet-directory <preserved full-GT head> --output <new directory>`.

Local output: `C:/KhoaLuanTotNghiep/Code/output/target_instance_sensitivity_v1/target_sensitivity_summary.json`. Committed evidence: `audit/target_instance_sensitivity_result.json`. Source/prediction hashes, per-predicate score distributions, training-label counts and example target scores are included. No GPU or new model training was needed.
