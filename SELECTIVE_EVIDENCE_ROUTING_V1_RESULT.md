# First selective evidence-routing prototype: development result

Implemented and run locally on CPU from the existing matched-detector feature archive. Code: `c201b4f`. This trains an instance/predicate verifier and exports selected prompts plus exactly matching decoder evidence. The causal decoder adapter is implemented and functionally tested, but has not been caption-trained.

## Fixed protocol

- Seed42, 30 full-batch epochs, AdamW lr0.001, weight decay0.01, gradient norm cap1. No epoch or seed selected from evaluation.
- Source training pool: 1,141 pairs /616 images. Image-disjoint fit:863 pairs/462 images; calibration:278 pairs/154 images. Source development evaluation:310 pairs/128 images. No image overlap across these three partitions.
- Both the frozen union/text CLIP prior and learned verifier fit temperature/thresholds on the same calibration partition. Each positive/negative tail needs observed >=90% known-label precision and >=5 known selections. Unknown labels do not enter fitting or precision counts.
- The new verifier is trained on the existing valid binary V-COCO supervision, not caption word cooccurrence or pseudo-negative predicate swaps. This first run does not add the separate human target-review annotations.

## Before the three-claim-per-image cap

| Measure on known development labels | Frozen CLIP | Learned instance verifier |
|---|---:|---:|
| Accepted true relations | 10 | 30 |
| Accepted false relations | 7 | 7 |
| Accepted known-label precision | 58.82% | 81.08% |
| Recall of196 known positive relations | 5.10% | 15.31% |
| Selected claims with unknown truth | 1 | 2 |
| Known-label Brier score, lower is better | 0.19237 | 0.06732 |
| Correct actor ordering on194 confirmed contrasts | 157 | 170 |

The90% calibration target does not carry over to development evaluation. Actor contrasts hold the target fixed and change the actor; these results are NOT same-actor/wrong-target accuracy. The194 edges are correlated, not194 independent images.

## After the same maximum-three-claims cap

| Measure | Frozen CLIP | Learned instance verifier |
|---|---:|---:|
| Selected known true claims | 10 | 24 |
| Selected known false claims | 7 | 3 |
| Selected unknown claims | 1 | 2 |
| Known-label precision | 58.82% | 88.89% |
| Known positive recall | 5.10% | 12.24% |

The learned export has29 claims in12 of128 evaluation images. Sparse selection and abstention are material limitations: this is not full-dataset scene-graph coverage. Both conditions use the existing deterministic threshold-margin cap priority; probabilities are not asserted to be perfectly calibrated across predicates.

## Failure that remains important

Uncapped holding has8 true and6 false accepted known claims:57.14% precision, versus CLIP's10 true/7 false (58.82%). The aggregate gain comes chiefly from riding (11 true/0 false) and sitting_on (11 true/1 false). Carrying/eating/talking_on have no accepted supported tail under this calibration policy. The method has not solved holding, all actions, or target attachment. Known actor-action absence negatives do not supply every same-actor/wrong-target hard negative.

## Verification and interpretation

22 relevant tests passed, covering masked unknown gradients, nonexclusive predicates, localization abstention, exact90% threshold handling, ordered pair/union sensitivity, matched prompt/memory identity, gate bounds, frozen verifier features/scores, zero-initialization baseline parity, all-rejected fallback and causal decoding. Real feature ZIP and extracted-directory reads produced identical rows/tensors. The saved verifier reloads strictly; exported artifact hashes and split disjointness were checked.

These are reused development images and a single fixed seed. Thresholds are conditional on known labels and matched localization, not guarantees for unannotated detector proposals. No independent benchmark, new human labels, decoder adaptation, candidate reranking, CIDEr/CHAIR improvement or novelty confirmation is claimed. Preserve the existing caption checkpoint as baseline.

Next, measure wrong-target errors explicitly before promoting this selector. The paired caption trial should freeze the verifier, use the same verified slots and object-only baseline memory, and compare ordinary vs reliability-bound gating with equal training and candidate budgets. Keep prompt selection gains separate from fusion gains.

Local artifacts: `C:/KhoaLuanTotNghiep/Code/output/selective_evidence_routing_v1`. Verifier checkpoint is about0.8MB and decoder evidence about0.12MB; no large backbone is duplicated. The complete CPU notebook supports the existing ZIP or an explicitly supplied extracted feature root, without COCO input or GPU.

Public aggregate/provenance: `audit/selective_evidence_routing_v1_result.json`. Raw feature tensors, model weights, per-image predictions and images remain local.
