# Matched decoder adaptation: no demonstrated caption advantage

Both pilot arms completed two epochs on the same512 action-bearing train/restval images, with32 batches per epoch. Reported adaptation cohort has no overlap with the excluded100-image development pool. Both update exactly the same18 decoder cross-attention/norm2 tensors, totaling3,154,944parameters. Source checkpoint and prompt hashes agree with prior probes. Evaluation checkpoint hashes agree with training summaries; weights themselves are not present in the review ZIP.

Review ZIP SHA256: `e3de0d5aa31aa89f757d2ddb188c01539ea625f8c15666377d7efd729242b7b4`.

| Training caption CE | Legacy | Direct relation memory |
|---|---:|---:|
| Epoch1 | 1.594747 | 1.593035 |
| Epoch2 | 1.582251 | 1.582316 |

The final difference is .000065 and does not show a direct-memory loss advantage. These are training losses, not validation quality metrics.

| Recomputed12-image evaluation | Legacy | Direct |
|---|---:|---:|
| Top caption changed by off | 0/12 | 0/12 |
| Ordered candidate list changed by off | 0/12 | 3/12 |
| Candidate set changed by off | 0/12 | 3/12 |
| Top caption changed by target swap | 0/12 | 0/12 |
| Ordered candidate list changed by target swap | 0/12 | 0/12 |

Legacy-on versus direct-on: identical top captions on12/12; different candidate sets on3/12 (028517,089033,329841). Both off conditions also return the same ordered candidate wording lists. Earlier frozen-routing R93 had6/12 direct on/off list changes and1/12 swap change; after adaptation these are3/12 and0/12. Less sensitivity alone is neither better nor worse quality.

The previously observed third-candidate chair claim on329841 is absent from the returned direct target-swap beams; its third caption now says the child is sitting on a floor with a book. The other captions also changed during adaptation, so absence from these three beams does not establish learned rejection of incorrect relation evidence or a hallucination-rate reduction. No human adjudication of all claims or independent correctness metric has been added.

The memory norm deltas match the prior frozen-routing values, as expected with the encoder frozen. They include appended-slot content and must not be read as causal semantic contribution. Reported optimization changed both decoder checkpoint hashes; the ZIP does not permit independent tensor-by-tensor verification of what changed.

Training used the original heuristic V1 action cache, not the newest selector's training output. Caption CE alone was optimized. There is still no demonstration of improved prompt generation or fine-grained target correctness. Generic relation text remains in every evaluation condition, and the alternate targets have unknown correctness.

## Decision

Do not promote direct memory, increase epochs automatically, or move to full training on this evidence. Retain both adapted checkpoints and the baseline. Direct tokens influence some lower-ranked candidates, but this small CE adaptation has not changed the selected top caption or demonstrated target-specific use. No CIDEr/BLEU/CHAIR gain is established.

A subsequent experiment must identify and measure the unresolved target-grounding objective or caption-quality effect rather than optimize caption-change counts. Independent validation would be needed for a quality claim; these12 images are reused inspected development images and may have been seen by the source caption checkpoint before this adaptation. Only the pilot's512 images are disjoint from them.

## Evidence

Public aggregate: `audit/decoder_relation_memory_adaptation_512_result.json`. Local protocol/analysis and preserved source ZIP: `C:/KhoaLuanTotNghiep/Code/output/decoder_relation_memory_adaptation_512_result`. Checkpoint SHA256 legacy `a6d3888885306128271ea35aa83927627f46f1a83b0a02e69b732d9340f3d0a8`, direct `5373daaf8f2b0d546d9b03eab64b2e0171539f2b11330e270614c28e29f0503c`.

Counts were recomputed from records, checking source summaries, distinct image IDs, cohort separation, identical trainable parameter lists and matching evaluation/training checkpoint hashes. No new training or checkpoint publication was performed during analysis.
