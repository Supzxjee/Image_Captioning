# Expanded actor verifier and train-only soft evidence fusion

The expanded data and soft fusion produce a useful **development signal**, not a validated full captioning improvement. Soft fusion removes the inspected wrong background-person/food links while retaining the foreground child's food links. On a reused detector-matched transfer cohort, known-label precision increases but recall decreases. No full prompt cache or caption retraining is launched from this result.

## Received data and verification

- Source: `actor_all_endpoints_features_v1_review.zip`, SHA256 `cd2b139265084045258a6ca251d9b4b8d8c8648a612ac1ab26615a3a23258136`.
- Train:1,013 endpoint pairs,322 images,736 exhaustive actor contrast edges. Dev:489 pairs,182 images,334 exhaustive actor edges.
- Official V-COCO train/val labels, GT boxes/classes, caption train/restval only. Train/dev image overlap and dev overlap with the1,388-image exclusion manifest are zero.
- Pair hashes, feature coverage/shape/finite values and frozen CLIP revision checks pass. The exclusion hash was verified against exact LF/CRLF forms of the committed manifest.
- Feature extraction code is pinned at7eb37b7. Kaggle version number was not supplied.

The earlier generic edge helper truncated negatives and reported316 dev edges. Exhaustive reporting is now explicit and tested. Main comparisons below use the same334 edges for all models; the legacy316-edge result is retained separately in the local JSON. The model weights were not changed to correct this reporting issue.

## Expanded actor head:30 fixed epochs

Same architecture/loss/optimizer as the prior nonlinear trial:272,454 parameters, union excluded, appearance+geometry, BCE and actor ranking, seed42,30 fixed epochs. Only confirmed actor endpoint labels supervise it. All six predicates now have enough endpoint support to enable the diagnostic veto, but eating still has only15 positive and18 negative train endpoints.

| Same334 dev actor edges | Correctly ordered |
| --- | ---: |
| Frozen CLIP union/text | 270/334 |
| Frozen R50 residual | 273/334 |
| Expanded actor head alone | 261/334 |
| Soft evidence fusion | 273/334 |

The actor head alone remains weaker than R50 on this exact cohort. Its hard train-fit veto leaves only one relation in the20-image detector audit: the ambiguous sitting-on-bed claim. It rejects both wrong background-person food links **and** the supported foreground-child eating-sandwich claim. This is not a successful acceptance policy.

## Soft evidence fusion implemented and trained locally

After observing hard-veto failure, freeze both heads and fit18 parameters: two nonnegative weights and an intercept for each predicate. Inputs are the residual similarity and actor logit, normalized by train-endpoint mean/std. Fit uses only confirmed train endpoints,100 fixed epochs,Adam lr0.03 and ridge0.01. All acceptance thresholds are fitted on train scores with the existing>=5accepts/>=90% empirical precision rule. No weights or thresholds are chosen from the child image or dev outcomes.

Selection still requires the original R50 proposer threshold, followed by the fused acceptance threshold and cap3. Multiple supported actors remain possible. Fused sigmoid activations are not calibrated probabilities. This is a sequential development trial, because the new dev set had already been inspected before designing soft fusion.

On the20-image detector audit, soft fusion selects four claims: child eating sandwich, child eating hot dog, person holding backpack, person sitting on bed. Both background-person food links are rejected. The conflicting hot-dog/sandwich labels, worn-backpack/holding error and ambiguous bed posture remain unresolved.

On confirmed dev actor endpoints among original proposer accepts:28positive/19negative become22positive/8negative. It removes11 annotated negatives but loses6 annotated positives. Endpoint-unassessed claims are not counted as false. Aggregate actor ordering remains273/334; equal win totals do not imply identical individual decisions.

## Frozen transfer to detector boxes

Reuse the prior R55 archive:364 class/IoU-matched detector pairs on286 images. Original residual scores reproduce exactly, archive/pair hashes match, and none of these images overlaps expanded verifier training. No training, threshold adjustment or feature extraction occurs in this evaluation.

| Conditional known-label selection measure | R50 proposer | Proposer + soft fusion |
| --- | ---: | ---: |
| Accepted positives | 43 | 36 |
| Accepted negatives | 6 | 3 |
| Accepted unknown links | 11 | 8 |
| Precision excluding unknowns | 87.76% | 92.31% |
| Recall among matched-pool positives | 17.70% | 14.81% |
| Actor edges correctly ordered | 48/60 | 52/60 |

The tradeoff is explicit:3 known false claims removed,7 known true claims lost. This small, reused, matched pool is not a fresh heldout benchmark, natural proposal distribution, full HOI detection AP, independent human precision or caption hallucination metric. The improvement in precision does not establish overall superiority. One seed was run; robustness remains untested.

## Decision

Retain the soft fusion as a candidate method for further development, keep all frozen baselines and failed hard-veto outputs. The next evaluation should measure precision **and** supported-relation coverage on a larger full detector candidate pool with disjoint images and independent labels. Cross-class physical-object ambiguity and holding-versus-wearing specificity still need their own treatment. Do not promote this to full caption training or claim scientific novelty from these results.

## Artifacts and reproduction

Local directories: `output/actor_all_endpoints_verifier_v1`, `output/actor_soft_fusion_v1`, `output/actor_soft_fusion_detector_transfer` under `C:/KhoaLuanTotNghiep/Code`. They contain heads, histories, dev predictions, candidate prompt caches, veto decisions, same-cohort baseline checks and transfer predictions. Compact review ZIPs exclude model binaries.

Actor head SHA256: `cdfaae489407b13d32eb065d63eaa101f008303f61484d2a5c81d9258f0e7295`. Fusion head SHA256: `5e7f838fe1fb585e3d41a090bef1617f61e868451225b7827ce5681b8b4f865d`. R50 head remains `86930afcbed9573a4a48a2a0b5a7d94507d52b67aaa63b73e5a5f7717907a438`.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_actor_compatibility_cached.py --train-source 'C:/Users/leduc/Downloads/actor_all_endpoints_features_v1_review.zip' --candidate-source 'C:/Users/leduc/Downloads/residual_detector_full_candidates_20_features_review.zip' --scored 'C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored' --output 'C:/KhoaLuanTotNghiep/Code/output/actor_all_endpoints_verifier_v1' --all-actor-edges
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_actor_soft_fusion_cached.py --train-source 'C:/Users/leduc/Downloads/actor_all_endpoints_features_v1_review.zip' --candidate-source 'C:/Users/leduc/Downloads/residual_detector_full_candidates_20_features_review.zip' --actor-directory 'C:/KhoaLuanTotNghiep/Code/output/actor_all_endpoints_verifier_v1' --residual-head 'C:/KhoaLuanTotNghiep/Code/output/clip_prior_residual_v1/residual_head.pt' --scored 'C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored' --output 'C:/KhoaLuanTotNghiep/Code/output/actor_soft_fusion_v1'
```

Five meaningful actor/supervision checks pass, including exhaustive metric accounting. Baseline score reproduction, artifact hashes, finite features and split-overlap checks also pass. No weekly report file was edited.
