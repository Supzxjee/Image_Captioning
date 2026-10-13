# Frozen full-candidate100 audit: soft fusion is not ready for full prompt generation

The apparent precision gain on the reused GT-matched detector pool does **not** establish quality on the broader candidate pool. On100 disjoint images, the frozen method still accepts plainly incompatible action/object claims and loses visibly supported interactions. Do not expand this scorer into a full prompt cache or start caption fusion training from these outputs.

## Verified run

- Source: `actor_soft_fusion_full100_features_review.zip`; SHA256 `864bedb4925d5a3cc7eca50fa434d81761f84928dba9acab91d6c1badf8b8d4d`.
-100 images;1,321 ordered person→other-object pairs;7,926 predicate scores. Excluded image overlap0 with the locked1,711-image manifest.
- Head hashes, protocol/input hashes, seeded image selection, caption splits, detector-cache hash/settings, candidate reconstruction and feature checks passed. No training, weight update or threshold refit.
- Frozen residual selects27 claims; soft fusion selects19 claims on11 images;89 images are object-only. Lower claim count is not recall or an improvement metric.
-17 selected claims are shared;10 baseline-selected claims disappear and2 appear after fusion filtering/cap ordering. The2 promoted claims were already eligible pre-cap baseline proposals, not newly generated relations.

## Assistant visual inspection

All29 claims in the union of both methods' selected outputs were inspected across18 full/marked images. Method membership was visible to the assistant. This is **not independent human annotation**, benchmark precision/recall or a blinded evaluation. Human fields in the prepared CSVs remain blank, and the other82 images' supported-relation coverage was not exhaustively annotated.

The10 removed claims were assessed descriptively as4 contradicted,2 supported and4 unclear. The2 promoted claims have merged/ambiguous actor/object boxes. These counts are audit findings, not ground-truth performance estimates.

### Clear errors retained by soft fusion

| Image suffix | Accepted claim | Visible issue |
| --- | --- | --- |
| 157350 | person sitting_on teddy bear | Person lies on bed next to toy. |
| 361397 | person eating clock | Target is wristwatch on person's wrist. |
| 418569 | person holding frisbee | Target disc is airborne above another person; supplied actor is not holding it. |
| 418569 | person riding frisbee | Supplied actor is running in shallow water. |
| 455667 | left officer riding right motorcycle | Actor is mounted on different bike; multiple nearby instances remain confused. |
| 130024 | person holding skis | Supplied skis are at feet; poles are held instead. |

Additional selected claims remain uncertain: exact hot-dog label during a visible bite; cake/eating on a reclining person with eyes closed; multiple pizza targets while a person smiles holding a slice; merged stunt-rider and motorcycle detections. Static context alone does not confirm a specific actor–action–target link. The uncertain claims are not silently counted as false.

### Supported interactions lost

-017356: child visibly bites donut held at mouth. Soft fusion rejects the claim.
-020575: person visibly holds frisbee in lowered hand. Soft fusion rejects the claim.

### Useful removals do occur

It rejects a far-edge spectator linked to motorcycle riding/sitting, a background spectator linked to a police bike, and a collage subject linked to pizza in another panel. These successes do not remove the retained failures or true-relation losses.

## Interpretation and next research target

The actor verifier helps some instances but does not establish the entire triplet. The method needs to distinguish **predicate–object compatibility**, the physical object instance, and actor/contact evidence. An actor-contrast objective alone is insufficient to certify claims such as riding-frisbee or eating-clock.

Do not tune thresholds or add a whitelist from these100 images and call it a heldout improvement. The audit has now been inspected and is development evidence for future changes. A learned predicate/object semantic verifier and better instance evidence require justified supervision; incomplete V-COCO role links cannot automatically become negative labels. Object-class and same-instance counterfactuals must be independently confirmed or use an explicit unknown/abstain policy.

The next implementation should test complete-triplet validity before prompt acceptance, rather than another global acceptance threshold. Retain both methods and this failure audit. For quantitative precision, independently annotate blind selected claims; for recall, enumerate supported instance-level relations in all images within the retained detector inventory and six-predicate scope. Do not infer recall from object-only outputs or absent links.

## Artifacts

Local output: `C:/KhoaLuanTotNghiep/Code/output/actor_soft_fusion_full100_audit`.

- Full and marked JPEGs, paired visual-review HTML, all candidate scores.
- Baseline/fusion prompt caches, paired-review CSV, blind claim CSV and separate method membership.
- Blank image-level coverage review template.
- `assistant_visual_review.json` contains29 method-aware assessments separately from human annotation fields.
- Compact review ZIP excludes images; full scored-review ZIP includes them. Original input and frozen predictions are preserved.

Scoring used core20439e4/workflowb84da52. Kaggle version number was not supplied. No weekly report document or caption metric was changed.
