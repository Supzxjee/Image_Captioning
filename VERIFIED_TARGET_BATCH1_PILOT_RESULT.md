# First human target-review batch and target-ranking pilot — R87–R88

50completed reviews produce34eligible target contrasts under the declared import rules. Both new heads fit these34training comparisons, but an explicit target-ranking loss does not improve detector transfer over the same-label control. Neither pilot head replaces the current pipeline.

## Review intake

The exported JSON contains51records:50with all five required fields and one untouched next case. Manifest/source checks pass. Review SHA256: `5ffdcedb7f9b72565f052b2e10860562c53ff9e1d4e386e6851e15291444507f`.

| Import outcome | Cases |
|---|---:|
| Eligible reference-supported / alternative-contradicted, clear boxes and note | 34 |
| Human assessment contradicts source-positive target | 10 |
| Both targets supported: no verified negative | 4 |
| Localization not clear | 2 |
| Incomplete/not yet reviewed, across full125case manifest | 75 |

Eligible34contrasts span22training images;21same-class and13cross-class alternatives. Predicates:riding11,holding10,sitting_on10,carrying2,eating1,talking_on0. They are not a representative balanced relation dataset.

The10source conflicts include holding horses/bicycles/motorcycles and other action/target cases. A disagreement is not proof that the reviewer is wrong or that the dataset is wrong. Earlier guidance on holding through reins/controls was too categorical; these semantics require review. Conflicts are excluded from new supervision and original labels are not silently overwritten. Their source positives remain in the base dataset, which is an unresolved label-quality limitation.

One human reviewer provided this batch; no second-reviewer agreement or independent adjudication is available. Raw reviewer identity/notes and mappings are retained locally rather than committed publicly.

## Paired pilot

Use the same1,141detector training pairs and real shared-union region features. Clone the training label tensor and convert **only the34eligible alternatives** fromunknown to0. Otherunknowns remain masked, and source files are unchanged. Both conditions receive exactly these same new BCE labels and the same original actor/predicate contrasts.

- Control (`classification_only`): existing BCE plus actor/predicate ranking. The name means no additional target-ranking term, not absence of the previous contrast losses.
- Target condition (`target_rank`): identical loss plus.2 mean hinge on the34verified target pairs,margin1logit, fixed image/subject/predicate and changed target instance.

Both use ordered-region heads with683,649parameters, identical seed42initialization,30AdamW steps,lr.001,weight_decay.01,clipnorm1 and train-only>=90%precision/>=5accepts thresholds. No dev-based epoch/weight/threshold selection. Only training images receive new annotations; earlier full100 assessments remain evaluation-only.

Two tests verify that only explicitly reviewedunknowns change in cloned tensors, original labels remain intact, unrelatedunknowns staymasked, and wrong-actor/known-label/duplicate contrasts are rejected. Real source/feature/manifest checks and finite-loss training pass.

## Training fit is not heldout target correctness

| Condition | Correct ordering on34reviewed train edges | Mean positive-minus-negative logit gap |
|---|---:|---:|
| Same-label control | 34 /34 | 5.251085 |
| Added target ranking | 34 /34 | 6.674597 |

The control already fits the34comparisons; increased margin does not prove better generalization. There is no independently labelled heldout target-swap set in this pilot.

## Existing detector development and transfer

Same310dev pairs /196known positives and364transfer pairs /243known positives. Counts are ungated, pre-cap, known-label precision excludesunknown. These pools are reused development data and omit unmatched detector proposals.

| Metric | Same-label control | Added target ranking |
|---|---:|---:|
| Dev macro AP | 0.873961 | 0.856816 |
| Dev correct /wrong /unknown | 152 /28 /8 | 156 /33 /8 |
| Dev actor contrasts | 182 /194 | 185 /194 |
| Transfer macro AP | 0.856115 | 0.838474 |
| Transfer correct /wrong /unknown | 190 /20 /25 | 189 /23 /29 |
| Transfer known precision | 90.48% | 89.15% |
| Transfer conditional recall | 78.19% | 77.78% |
| Transfer actor contrasts | 58 /60 | 58 /60 |

Explicit target ranking increases3dev actor wins but admits morefalse claims and does not improve transfer. Under the secondary unchanged R50 conjunction, both retain42correct /2wrong;unknown1(control) versus2(target condition). No pipeline promotion follows.

Full100 ungated output:128claims/73images(control) versus135/78(target condition). Both still select the previously reviewed false holding-skis-at-feet tuple. Most selected claims lack exhaustive independent labels; these counts are not full100accuracy.

## Decision and remaining work

The partial batch is usable and saved; further reviews can be imported cumulatively with the same manifest.75cases remain. Unknowns, both-supported pairs, bad boxes and source conflicts are not manufactured into target negatives.

Keep both heads as development artifacts. Resolve action semantics/source conflicts and expand reliable target labels before treating the pilot as a complete evaluation of target ranking. Do not tune new loss weights or thresholds to these inspected results. A labelled heldout target set and broader seed checks would be needed before claiming a reliable improvement; no caption training or novelty claim is made here.

## Artifacts

Local reviewed batch/filtered annotations: `C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1/human_batch1_review.json` and `human_batch1_filtered.json`. They are local data, not public commits.

Pilot: `C:/KhoaLuanTotNghiep/Code/output/verified_target_rank_pilot_v1`. Reproduce with `run_interaction_region_cached.py` using the prior source arguments plus`--target-review <returned human JSON> --annotation-package <original package> --output <new pilot directory>`.

Control head SHA256: `d8d499abb958c816b8a6f936d88274b13e3e9ae3703e6649e7ce14c2aadded3f`.

Target-ranking head SHA256: `2fc976ae2a7f70f19b21cf8672411c99f812baa40d460104a1d2e03edb0710ba`.

Committed aggregate evidence: `audit/verified_target_rank_pilot_result.json`. Head descriptive metadata was corrected after execution to identify the same-region control; model tensors were verified unchanged. Local compact review ZIP includes bothheads/results and excludes raw human annotations/source feature tensors. No new Kaggle/GPU run was needed.
