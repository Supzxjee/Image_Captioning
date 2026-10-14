# Recall bottleneck: proposer gate versus triplet verifier — R78–R79

The frozen R50 proposer gate is a major recall bottleneck on the matched detector pool. It rejects200 of243 known positive relations, including83 accepted by the preserved full-GT TripletVerifier. Removing that gate recovers correct relations but also exposes clear instance/action errors on the full100 candidate pool. The deployed gate and heads are unchanged.

## Frozen decomposition

Same R55 matched364pairs /286images, source SHA256 `a129ab178f02e147980c537b1a64f21f8cc8f16a2c2ad09e60049ddac0d36e07`. Existing scores and train-only thresholds are reused. No training, threshold refit, or pipeline promotion. Train-image overlap0 verified by the existing evaluator.

R50 is a CLIP union/text cosine prior with a learned bounded residual correction, not raw CLIP alone. Its hard threshold precedes the verifier. The preserved full-GT seed42 verifier has SHA256 `de8124a30555b39eabdf45cdf649d599788174361c94bd8036ebed714f61fab4`.

| Gate state | Known correct | Known wrong | Unknown |
|---|---:|---:|---:|
| Both accept: current conjunction | 35 | 1 | 8 |
| Proposer accepts, verifier rejects | 8 | 5 | 3 |
| Proposer rejects, verifier accepts | 83 | 11 | 30 |
| Both reject | 117 | 1,682 | 201 |

The states partition every candidate/relation cell; known-positive conservation is checked:35+8+83+117=243. Both-reject cases are not separately attributed to one stage. These counts precede the per-image prompt cap.

Proposer recall is43/243=17.70%; therefore the conjunction cannot exceed17.70% recall without changing the proposer gate. Current conjunction recall is14.40%. Verifier alone, as a diagnostic counterfactual, accepts118correct /12wrong /38unknown: known-label precision90.77%, conditional recall48.56%. The extra83correct come with11additional known wrong and30unknown claims. Unknown is not equivalent to false or true.

| Relation | Both accept | Proposer only | Verifier only | Both reject |
|---|---:|---:|---:|---:|
| holding | 5 | 4 | 62 | 71 |
| riding | 13 | 2 | 2 | 1 |
| sitting_on | 5 | 2 | 12 | 9 |
| carrying | 0 | 0 | 3 | 24 |
| eating | 12 | 0 | 2 | 9 |
| talking_on | 0 | 0 | 2 | 3 |

This table counts known positives only. Most verifier-accepted positives blocked by R50 are holding62/83. This locates a problem in these frozen acceptance rules, not proof that all underlying representation errors are solved.

The rejected mixed-view head shows the same proposer ceiling: both28correct, proposer-only15, verifier-only80, neither120. Verifier-alone108correct /11wrong /34unknown gives90.76% precision and44.44% conditional recall. It does not improve positive retention over the preserved full-GT head.

## Full100 counterfactual stress test

Source SHA256 `864bedb4925d5a3cc7eca50fa434d81761f84928dba9acab91d6c1badf8b8d4d`. All1,321 detector pairs /7,926 relation scores are already cached; no GT matching limits this candidate pool. Remove only R50's hard gate, reuse the full-GT verifier's scores and original thresholds, keep the same detector inventory and deterministic maximum3relations per image.

| Output | Current conjunction | Verifier-alone counterfactual |
|---|---:|---:|
| Selected relations | 11 | 91 |
| Images with relations | 7 | 44 |

Counterfactual has160 above-threshold claims before cap.81 selected tuples are newly admitted; one old selected tuple is displaced by cap competition. Removing a gate therefore does not necessarily preserve every previous selected tuple after capping. None of the six previously recorded contradicted tuples is selected, but those old reviews cover only a small part of the new output.

A purposive assistant spot-check of6claims on4images finds:

- Supported: child holding donut (`017356`,subject`det_d8db9964c24e`,object`det_ead7ab62f8e1`); foreground man holding frisbee (`020575`,subject`det_95f9611d26b5`,object`det_2afde1a7c352`). These are holding claims; recovering an object-related claim is not the same as recovering every previously reviewed predicate for that object.
- Contradicted: background swimmer riding the foreground man (`020575`,subject`det_1b70162949eb`,object`det_95f9611d26b5`); background spectator riding the skateboarder (`572807`,subject`det_a934cbf7e0f1`,object`det_4075a7f24702`).
- Contradicted: two men sitting on the cake table while standing/leaning around it (`219565`,subjects`det_f14de9bc480f` and`det_7ad2d9c5d1a6`,object`det_5db511eea5ef`).

The spot-check is method-visible assistant inspection, not independent human labels or random sampling. Do not estimate full100 precision from these6claims. Full100 is inspected development data; matched-pool precision cannot be extrapolated to all unmatched detector proposals. No manually reviewed labels are added to training.

## Decision and next implementation

Preserve the original pipeline. Neither a strict R50 veto nor unconditional verifier acceptance solves the observed retention/instance-error tradeoff.

The next development target is to use R50's continuous score as a **soft input** to a learned triplet acceptance score instead of letting it veto every candidate. Fit only on official known training labels, retain unknown masking and instance/predicate contrasts, and evaluate both true-relation retention and the newly exposed errors. This is a proposed next experiment, not implemented or shown superior by this diagnostic; no object whitelist or threshold adjustment from inspected examples is justified.

## Reproducibility and artifacts

`evaluate_triplet_transfer_cached.py --bottleneck-audit` adds exhaustive four-state partition checks, per-relation decomposition, and frozen per-pair scores. Use separate output paths for the full-GT and mixed heads; prior evaluation artifacts are preserved.

`audit_triplet_ungated_counterfactual.py` checks full100 source hash and candidate identity/order, then writes a separate counterfactual cache; it never overwrites the head's prompt cache.

Committed compact evidence: `audit/triplet_gate_bottleneck_result.json`.

Local outputs: `C:/KhoaLuanTotNghiep/Code/output/triplet_gate_bottleneck_v1`, including both gate decompositions/predictions, known-positive CSV, full100 counterfactual prompts and assistant spot-check. Review ZIP: `C:/KhoaLuanTotNghiep/Code/output/triplet_gate_bottleneck_v1_review.zip`. No new model weights or feature extraction were needed. No caption improvement or scientific novelty claim follows from these diagnostics.
