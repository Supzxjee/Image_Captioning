# Frozen E matcher diagnostic — 19 October 2026

**Decision: the probe does not justify treating the E matcher as reliable and blaming the decoder alone. Prioritize the matcher's training signal before full training or stronger decoder injection.** This is a small, reused diagnostic, not a caption evaluation or a proof of the cause of E's caption loss.

The submitted review passes the artifact audit: exact locked manifest and source hashes, 62 pairs/59 images, complete nonoverlapping two-T4 inference shards, consistent recorded module digests, correct checkpoint identity, and locally recomputed finite scores/accuracies/intervals. The run records no training and unchanged parameters. No new model inference was launched by this audit.

## Observed pair ranking

Percentages below are **pair accuracy**, with ties worth 0.5. Image-macro accuracy and image-cluster bootstrap intervals are separately reported in [PROBE_RESULT.json](PROBE_RESULT.json).

| Diagnostic task | E with boxes | E full image, no boxes | E zero visual features |
|---|---:|---:|---:|
| Wrong object instance | 8/12 = 66.7% | 50%, all ties | 50%, all ties |
| Wrong predicate | 8/12 = 66.7% | 10/12 = 83.3% | 6/12 = 50.0% |
| Wrong subject instance | 27/38 = 71.1% | 50%, all ties | 50%, all ties |

For the instance tasks the candidate text is identical. When boxes are removed the model receives identical inputs for both candidates, so 50% ties are **expected by construction**, not an independently strong baseline or proof that box-aware results establish semantics. Zero-image instance ties are also expected because this module pools constant image values.

For predicate ranking, positives are balanced holding/riding within bicycle (5/5) and motorcycle (1/1). This blocks a simple per-object-class verb prior from winning the set overall. E with boxes gets holding 3/6 and riding 5/6; the full-image condition gets 5/6 for each. Four boxed-predicate errors occur on COCO IDs 116512, 117371, 340406 and 562296. These are annotation-relative errors, not independently re-inspected visual judgments.

The full-image condition's higher predicate score suggests context outside the allowed union might matter. It does **not** establish this cause: removing masks and membership changes more than one factor, only twelve examples exist, and the result is exploratory. Do not switch the whole method to global attention on this evidence alone.

## Geometry countercheck

After reading the results, we added two simple descriptive controls, without training or tuning: choose greater subject–object box IoU, or choose smaller normalized center distance. These depend only on boxes, not image pixels, text embeddings or the learned model. The same 1e-7 tie tolerance is applied.

| Task | E boxed | Greater box IoU | Closer centers |
|---|---:|---:|---:|
| Wrong object instance | 8/12 | **12/12** | **12/12** |
| Wrong subject instance | 27/38 | **34/38** | **34/38** |
| Wrong predicate | 8/12 | 50%, all ties | 50%, all ties |

This is a limitation of the **probe as well as the matcher**. Many instance cases can be solved by geometry. A boxed score above 50% is therefore insufficient evidence of richer visual-semantic understanding; here it does not even exceed these simple geometric comparators. Conversely, perfect geometry on twelve selected cases does not justify replacing semantic modeling with box rules. These controls were added post hoc and do not provide a new fitted baseline or universal performance claim.

## Uncertainty

Primary image-macro results for E boxed:

- Object instances: 65.0%, 95% image bootstrap [40.0%, 90.0%], ten unique images.
- Predicates: 66.7%, interval [41.7%, 91.7%], twelve unique images.
- Subject instances: 70.3%, interval [54.1%, 83.8%], 37 unique images.

These marginal intervals do not quantify paired method differences or training-seed uncertainty. The tasks overlap on some images. All images are Karpathy train/restval, excluded from the latest E ten-thousand-image updates but previously used for caption pretraining; human examples have also been used in earlier diagnostics. GT/human boxes differ from the UPT boxes E learned with. Holding/riding negatives follow explicit human/V-COCO judgments; the V-COCO ontology is not exhaustive natural-language truth. No caption score or Karpathy test result is produced here.

## What this changes

Earlier E training achieved 96.85% positive-margin accuracy on its constructed training pseudo-pairs. The present 66.7–71.1% hard-pair results measure a different task, so they are not a direct generalization-gap estimate. Together they show why low auxiliary loss was not sufficient validation of the distinctions required downstream.

There is not enough evidence to conclude the matcher works well and only decoder use is broken. Keep the best caption checkpoint and stop scaling the current E variant. The next **narrow training hypothesis** is explicit same-image verified contrasts: preserve subject/object identity and swap the action when the alternate is labeled false; preserve the action and use a confirmed wrong participant. A pairwise margin loss can stay simple; change the evidence and negative construction rather than merely its coefficient. Fit on a disjoint source partition and reserve this diagnostic for reporting, not training or threshold choice. Include geometric controls from the outset and add spatially ambiguous cases where proximity cannot decide. Whether this change improves the matcher or caption is still untested.

## Reproduce the artifact audit

`python audit_probe_review.py --source semantic_evidence_matcher_probe_review.zip --output PROBE_RESULT.json`

The auditor reads JSON and ZIP bytes only, verifies submitted source without executing it, recomputes tie-aware summaries and bootstrap intervals from saved scores, and computes the geometry controls. It does not load the checkpoint or replay inference from the visual cache. Equal/unchanged tensor digests are worker-recorded evidence, not independently checked archived tensors.
