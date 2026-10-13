# Instance evidence diagnostics and pair-only residual trial

The pair-only correction did not solve actor binding and is **not promoted**. All work ran locally on existing frozen features; no new Kaggle run, caption training, or GPU extraction was needed.

## Frozen R50 intervention

`audit_instance_evidence_cached.py` reproduces the validated R58 scores, then replaces only the subject feature with another detected person's feature from the same image and exact object. Object feature, union feature and text remain fixed. These synthetic feature combinations can be off-distribution; replacements are interventions, not labelled negatives.

There are 2,934 ordered interventions across the 600 detector pairs. Mean absolute score change across six predicates is 0.0008805. This includes many rejected/saturated scores and person–person candidates, so it must not be read as an accuracy metric or proof of global subject independence.

For the concrete wrong background-person/hot-dog pair, inserting the foreground child's subject feature changes eating by +0.005402. The original wrong-actor score is 0.400058, well above the frozen eating threshold 0.303956. The correct-child/hot-dog pair scores 0.417965, only 0.017908 higher. By contrast, the child's sandwich claim is more sensitive to subject replacement (-0.054875). Thus subject use depends on the candidate and text; it is not uniformly absent.

## Conflicting-region abstention ablation

A new exploratory rule flags different detector classes with box IoU >=0.9. It flags exactly one pair: sandwich/hot dog, IoU 0.993492. Excluding relations involving either region reduces selected claims from five to two and relation-bearing images from three to two.

This also removes the broadly supported child-eating-sandwich claim. The two remaining claims are holding-backpack (contradicted action) and sitting-on-bed (ambiguous posture). This is **not a successful precision fix**. The object inventory still contains both alternative labels. No ground-truth label is used to choose one, and no indiscriminate class-agnostic NMS is applied.

## Architectural trial actually trained

`PairOnlyResidualAlignment` retains the frozen union/text CLIP cosine as the base. Its zero-initialized bounded correction MLP sees only concatenated subject/object features, rather than subject/object/union. Scale remains 0.1. This is an architecture ablation, not a novelty claim.

Training uses the same V2 cached 1,024 training and 256 development pairs, seed42, 20 fixed epochs, learning rate0.001, train-only logit center, known-label BCE and the existing type-balanced predicate/actor ranking loss. Missing links remain unknown. The threshold selection rule is unchanged and thresholds are fitted on training scores only. The new model naturally has new train-fit thresholds; it is not the frozen R50 protocol.

| Development measure | Original union-conditioned residual | Pair-only correction |
| --- | ---: | ---: |
| Macro AP on known labels | 0.732156 | 0.720418 |
| Actor wins | 36/40 | 36/40 |
| Mean actor score gap | 0.034743 | 0.033606 |
| Predicate wins | 588/604 | 587/604 |

On the previously inspected 20-image detector pool, the new model selects seven claims in four images. It retains the wrong background-person/hot-dog claim (score0.361900, new threshold0.311541), worn-backpack/holding and the two competing food labels. It additionally selects two carrying-suitcase claims in image282934: the supplied object box is a foreground suitcase sitting on a seat, while the supplied people are farther back. Assistant inspection of the marked image finds no visible carrying link for either pair. More selected claims do not establish improvement.

The reused development sets have now informed design. No untouched heldout result, generalization gain, human-labelled precision, or caption benefit is claimed.

## Decision and next implementation target

Keep R50 frozen for reference, retain the pair-only model as a failed development trial, and do not generate a full prompt cache from either. The next substantive change must address the training objective and actor discrimination, rather than merely remove union input or increase a threshold.

The existing actor loss already uses confirmed actor negatives. A further experiment should separate predicate recognition from same-object actor compatibility, train the actor component explicitly on confirmed positive/negative actor competition, and report its own margin and acceptance behavior. Unknown actors cannot be assigned negative labels; multiple valid actors must remain possible. Additional confirmed same-object actor examples may be needed. A simple “one winner per object” rule would conceal this failure and can reject valid shared interactions.

## Reproduction and artifacts

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_residual_alignment_cached.py --source 'C:/Users/leduc/Downloads/instance_contrast_alignment_v2_diagnostic_1024_review.zip' --output 'C:/KhoaLuanTotNghiep/Code/output/pair_only_residual_development_v1' --pair-only
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' audit_instance_evidence_cached.py --source 'C:/Users/leduc/Downloads/residual_detector_full_candidates_20_features_review.zip' --scored 'C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored' --checkpoint 'C:/KhoaLuanTotNghiep/Code/output/clip_prior_residual_v1/residual_head.pt' --comparison-head 'C:/KhoaLuanTotNghiep/Code/output/pair_only_residual_development_v1/residual_head.pt' --output 'C:/KhoaLuanTotNghiep/Code/output/instance_evidence_development_audit_20'
```

Trial checkpoint SHA256: `422fa428f4d21ba05f57568475c0bd342fed3487355f97ff67e377e7e6d8626b`. Frozen source hashes are recorded in each JSON output. Four new checks passed: intervention grouping does not cross images/objects, conflicting-region handling does not merge distinct regions, initial pair-only score equals CLIP, and union changes cannot enter learned pair-only correction. Existing six selector/candidate tests also pass.
