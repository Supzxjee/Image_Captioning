# Candidate: visual evidence conditions noun-bank aggregation

02/11/2026. Brainstorm response to user asking for an additional pipeline stream that can improve CIDEr and hallucination metrics. **Not an approved implementation, established novelty or demonstrated improvement.** Do not change the current full10 XE run.

## Specific hypothesis and insertion point

Current CSA already uses image-dependent alignment/selection. However, `SemanticAggregation.forward(bank)` receives no image: the150 noun prototypes before stage1 are the same for all images at a given parameter state. Test whether image-conditioned evidence at **this earlier aggregation step** improves useful noun selection, without adding vocabulary-logit bias or a third decoder memory.

Proposed path: projected ROI features V0 → small object-evidence head → bounded evidence weights for mapped noun-bank entries → image-conditioned noun aggregation → unchanged object1/object2/relation/parallel decoder. Keep verb aggregation unchanged initially. This targets object grounding, not a claim to verify actions or subject-object relations.

Let existing word-by-prototype aggregation weights be A[j,k], with column sums1. For image i produce evidence weight r_i[j] bounded below by a positive epsilon. Define A_i[j,k]=A[j,k]r_i[j]/sum_l A[l,k]r_i[l]. Then To_i=A_i^T wb(noun_bank). Image-specific To_i enters existing stage1; selectedT1 still passes to stage2; Vo/Vr paths unchanged. Epsilon is not selected yet. Words outside the supervised mapping stay neutral r=1; no hard zeroing or claim that a missing detection means absent object. Initial integration must preserve the original forward and be checked for gradient flow before any experiment.

## Supervision / uncertainty still to resolve before code

Use training-only object annotations and mapped caption mentions to supervise a limited, auditable object vocabulary. These are training labels, never oracle inference inputs. Check available COCO-instance assets, mapping, coverage and ambiguous negatives first; current ROI-cache metadata alone is not exhaustive presence/absence supervision. Unknown/unmapped categories masked in evidence loss; do not convert absent captions or top-K detector omissions into negatives. On a new image the branch predicts evidence from visual features only.

Candidate loss: L=XE+lambda*masked_object_evidence_loss. Lambda, evidence labels, pooling, normalization and budget are **unlocked**. Auxiliary evidence accuracy is diagnostic, not caption success. If later using SCST/CIDEr, apply the same SCST schedule/budget to baseline and proposed arm; do not attribute an unmatched reward/schedule gain to this branch.

## Smallest useful comparison

After baseline full10 results, compare baseline continuation vs proposed continuation from the same checkpoint, TRAIN subset/order/update budget, validation IDs and decoding policy. Primary outputs CIDEr↑, CHAIRs/i↓, raw false captions/mentions, recall and caption length. Do not count shorter captions/omission as a dehallucination success. If a joint signal appears, add the auxiliary-head-only/no-feedback control to separate label-supervision effects from image-conditioned bank routing, plus a neutral/shuffled-evidence diagnostic. No new test tuning; final locked independent evaluation remains separate.

## Prior art / historical caution

- Existing G/S noun evidence into logits **failed** against A: `experiments/noun_evidence_proposal/GS_RESULT.md`. This proposal changes the insertion point and trainable system, but that difference is not proof it will work or be novel. Existing I/J and K/V D/E reader branches also failed quality gates; no automatic repeat as a renamed decoder branch.
- LightCap uses concept extraction and cross-modal feature modulation: https://arxiv.org/html/2212.08985#S3 . Evidence-conditioned semantic processing has prior art; exact novelty of the proposed aggregation operator is **unverified**. A larger nearest-method review is required before claiming originality, and the candidate may turn out to be an ordinary conditional-attention variant.
- Object grounding does not automatically establish lower free-generation hallucination: https://aclanthology.org/2024.emnlp-main.159/ . CHAIR primarily measures object hallucination, not general relation/attribute correctness.

Recommendation: discuss this concrete insertion point with the user; keep current CSA baseline unchanged while it runs. No notebook/model/extra GPU work authorized by this brainstorm alone.
