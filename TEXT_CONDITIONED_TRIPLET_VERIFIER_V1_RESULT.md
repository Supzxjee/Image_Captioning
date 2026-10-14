# Text-conditioned ordered-triplet verification: local development result

A whole-triplet verifier was implemented and trained locally using existing caches. Candidate-specific predicate/object text improves a matched relation-bank ablation on the observed development sets. This is a promising **development candidate**, not a validated captioning improvement or novelty claim.

## Actual implementation and supervision

`TripletVerifier` encodes ordered subject/object CLIP features, their product/difference, and normalized box geometry. A separate projection encodes the frozen CLIP text embedding for each candidate predicate and object name. A shared interaction MLP scores the visual pair, text and their elementwise interaction. The union image feature does not enter this verifier; the frozen R50 proposer still supplies the first-stage candidate threshold.

Pool only official training partitions from the earlier V2 cache and expanded actor cache. Deduplicate by image/subject/object IDs and check labels, boxes and duplicate feature agreement. Result:1,776 distinct training pairs on826 images;1,089 positive,8,560 known-zero and1,007 unknown predicate labels. Duplicate feature max difference0. Train/development and train/full100 image overlaps are zero.

Zero labels mean the annotated actor action is absent, making that triplet false. They do **not** prove an object class can never support that predicate. Missing role links remain unknown and are excluded from BCE/contrast targets. No object whitelist, manual audit labels or invented wrong-object negatives are used for training.

Use all736 confirmed actor and4,408 predicate contrasts with the existing type-balanced contrast reduction. Seed42,30 fixed epochs,AdamW lr0.001/weight decay0.01, known-label BCE plus0.2 ranking, margin1 on logits. Train-only acceptance threshold rule remains >=5known accepts at >=90% empirical train precision. Sigmoid outputs are not calibrated probabilities.

## Matched text-conditioning ablation

The ablation has identical train/dev rows, model dimensions (354,433 parameters), seed, initialization, optimizer/loss, epoch count and threshold-fit rule. It replaces candidate-specific text with a shared six-entry bank computed from train-only mean text vectors, one per predicate. The bank can retain aggregate training-object information; this is removal of **candidate-specific** object text, not removal of all semantic knowledge. Both models still see the target object's visual crop.

| Same489-pair development set | Shared relation bank | Candidate-specific predicate/object text |
| --- | ---: | ---: |
| Macro AP, same known-label mask | 0.679835 | 0.760567 |
| Correct actor ordering, same334 edges | 268/334 | 271/334 |
| Selected claims on inspected full100 pool | 12 | 11 |

Frozen R50 orders273/334 edges correctly on this cohort. The new verifier does not beat it on aggregate actor ordering. Comparing against prior actor-only/soft-fusion architectures is not a clean mechanistic ablation because this trial also changes training data pooling, supervision and architecture. Only the relation-bank versus candidate-text comparison above isolates the candidate-text input under this fixed setup.

## Reused detector transfer audit

Evaluate frozen heads and train-fitted thresholds on the same R55 class/IoU-matched364pairs/286images, with zero overlap against either training cache. No fit occurs during evaluation.

| Conditional accepted claims | R50 proposer | Prior soft fusion | Shared relation bank | Candidate-specific triplet verifier |
| --- | ---: | ---: | ---: | ---: |
| Known positives | 43 | 36 | 28 | 35 |
| Known negatives | 6 | 3 | 2 | 1 |
| Unknown links | 11 | 8 | 6 | 8 |
| Precision excluding unknown links | 87.76% | 92.31% | 93.33% | 97.22% |
| Recall among matched-pool positives | 17.70% | 14.81% | 11.52% | 14.40% |

Candidate-specific text retains7 more known true claims and1 fewer false claim than the matched relation-bank ablation. Relative to the original proposer it still loses8 known true claims while removing5 false claims. The verifier's actor ordering on60 transfer edges is51/60 versus50/60 for relation-bank and52/60 for prior soft fusion. These small differences are not significance or robustness evidence.

This is a reused, conditional matched pool, not natural full-proposal prevalence, fresh heldout performance, independent human precision or full HOI detection AP. Unknown links are excluded from precision. Only seed42 was run.

## Full100 development findings

The candidate-text verifier selects11 claims on7 images;93images are object-only. The shared relation-bank ablation selects12claims on8images. This full100 set was inspected before developing these models, so it is now development evidence.

The candidate-text output no longer contains the six clear contradictions retained by prior soft fusion: eating-clock, riding-frisbee, holding-airborne-frisbee, sitting-on-teddy-bear, left-officer/right-bike binding, or holding-skis-at-feet. The relation-bank model also removes most of these and differs by retaining the airborne-frisbee holding error. Do not attribute all six removals specifically to candidate-text conditioning.

The two models still accept eating-cake in the reclining/sleeping-context image and ambiguous pizza/hot-dog targets. Candidate text does not resolve every visual action or object-class uncertainty. It does not recover the supported donut-bite and lowered-hand frisbee claims lost earlier, and additionally drops the supported umbrella-holding claim. A new right-officer/right-motorcycle claim is regionally supported in the previously viewed image, but merged motorcycle boxes still complicate exact physical-instance interpretation. Assistant visual observations are method-aware and are not human benchmark labels.

## Decision

Keep this as the next candidate verifier and preserve failed/previous methods. Do not yet generate a full relation prompt cache or start full caption training. The controlled ablation supports investigating text-conditioned verification, while low coverage and true-relation losses remain substantive problems.

Before a publication claim, examine stability across seeds and a newly fixed evaluation with independent claim/coverage annotations. A future method must preserve valid relations as well as remove incompatible triplets. Do not tune the threshold on the already inspected donut or frisbee image and present that as generalization.

## Artifacts and reproduction

Local outputs under `C:/KhoaLuanTotNghiep/Code/output`:

- `text_conditioned_triplet_verifier_v1`: head, history, thresholds, full100 predictions and prompts.
- `triplet_relation_only_ablation_v1`: matched ablation artifacts.
- `triplet_verifier_detector_transfer_v1` and `triplet_relation_only_detector_transfer_v1`: frozen transfer summaries.

Main head SHA256 `de8124a30555b39eabdf45cdf649d599788174361c94bd8036ebed714f61fab4`; ablation head SHA256 `ce5422b743157eaa4ac98eec7fb326071ad8544b200e928c8535c5150a38c020`. Source cache hashes are recorded in each run. No new Kaggle/GPU extraction was needed.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_triplet_verifier_cached.py --v2-source 'C:/Users/leduc/Downloads/instance_contrast_alignment_v2_diagnostic_1024_review.zip' --actor-source 'C:/Users/leduc/Downloads/actor_all_endpoints_features_v1_review.zip' --candidate-source 'C:/Users/leduc/Downloads/actor_soft_fusion_full100_features_review.zip' --scored 'C:/KhoaLuanTotNghiep/Code/output/actor_soft_fusion_full100_audit' --output 'C:/KhoaLuanTotNghiep/Code/output/text_conditioned_triplet_verifier_v1'
# Matched ablation: add --relation-only-text and write to a separate output directory.
```

Two meaningful verifier tests passed: text/candidate order affect the output while union features do not, and unknown-labelled scores cannot affect the supervised loss. Training/evaluation hash, feature agreement and image-overlap checks passed. No weekly report file was edited.
