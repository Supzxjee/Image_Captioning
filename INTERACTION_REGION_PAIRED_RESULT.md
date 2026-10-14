# Shared-union region evidence versus union-CLS control — R82–R84

The uploaded real region features pass validation, and both matched heads have been trained locally. **Region pooling does not outperform the repeated union-CLS control on primary matched transfer acceptance.** Both heads improve the earlier inspected person/table cases but still produce unsupported relations. No new production head is promoted.

## Feature audit and controlled training

Uploaded archive SHA256: `6b055345bebfc6c9e82fb8ee81e09b0a4894bf04c6f5c3b889eee3a7e5b7aaee`.

All3,136pairs are present:1,141train /310dev /364transfer /1,321full100. Reported shard hashes match the uploaded bytes. Local validation reproduces the extraction report, including finite normalized6x768descriptors, zero empty regions, ROI/crop geometry, pair metadata/protocol hashes, exact ordering and zero training/evaluation image overlap. Model revision is checked through the pinned extraction manifest. Feature extraction is complete; the notebook version number was not supplied.

Both heads have683,649parameters, identical seed42 initialization, the same1,141known-label detector training pairs, geometry, validity/support mass, candidate-specific cached CLIP text and schedule. Train30full-batch AdamW steps,lr.001,weight_decay.01,clipnorm1. Loss is known-label BCE plus.2 type-balanced confirmed actor/predicate ranking,margin1logit/temperature1. Unknown labels stay masked. Train-only thresholds use the existing >=90% empirical precision / >=5known accepts rule; no dev threshold fitting or checkpoint selection.

The region model uses ordered subject/object/intersection/bridge/outside/union-CLS content. The control repeats union CLS in each valid slot while retaining the same masks, mass and geometry. Thus the controlled difference is local region content rather than capacity or input dimensionality.

Primary acceptance has no hard R50 gate. A separate frozen R50 conjunction is reported as a secondary diagnostic; neither replaces the current pipeline. All cohorts are reused development data, not fresh heldout evaluation.

## Primary common detector validation

Same310pairs /128images,196known positive labels. Eating/talking_on each have only4positive dev examples.

| Method | Macro AP | Correct actor contrasts | Correct /wrong /unknown accepts | Known precision | Conditional recall |
|---|---:|---:|---:|---:|---:|
| Union CLS control | 0.886535 | 180 /194 | 157 /35 /7 | 81.77% | 80.10% |
| Ordered regions | 0.869123 | 182 /194 | 153 /28 /8 | 84.53% | 78.06% |

Regions reduce known wrong acceptance by7 and improve2actor contrasts, but lower AP and correct acceptance. The90%training target does not guarantee90%validation precision.

## Primary R55 matched detector transfer

Same364pairs /286images,243known positives. These pre-cap counts cover matched detector pairs only; unmatched proposals are absent and unknown claims are excluded from known-label precision.

| Method | Correct kept | Wrong kept | Unknown | Known precision | Conditional recall | Actor contrasts |
|---|---:|---:|---:|---:|---:|---:|
| Union CLS control | 199 | 22 | 33 | 90.05% | 81.89% | 58 /60 |
| Ordered regions | 189 | 22 | 25 | 89.57% | 77.78% | 58 /60 |

Region transfer macro AP is0.855691 versus control0.851513, a small ranking gain that does **not** improve acceptance at frozen train thresholds. Regions lose10correct accepts with the same22wrong accepts. No superiority claim follows.

These new heads use different training rows, hidden features and capacity from the earlier512D crop TripletVerifier. Their stronger counts than older runs are context, not proof that region content caused the improvement: the controlled union-CLS head is already strong.

## Secondary conjunction with frozen R50

Same original R50 head/thresholds, no refitting.

| Method | Correct | Wrong | Unknown | Known precision | Conditional recall |
|---|---:|---:|---:|---:|---:|
| Union CLS AND R50 | 41 | 3 | 2 | 93.18% | 16.87% |
| Regions AND R50 | 42 | 2 | 1 | 95.45% | 17.28% |

This modest+1correct/-1wrong result is secondary, single-seed and on reused data. R50 still imposes its43/243known-positive ceiling. It does not establish a solution to the recall bottleneck.

## Full100 candidate checks

Same100images /1,321detector pairs, cap3, detector inventory and instance identifiers unchanged.

| Method | Ungated selected relations | Images with relations | With R50 gate |
|---|---:|---:|---:|
| Union CLS control | 169 | 82 | 21 |
| Ordered regions | 128 | 73 | 19 |

Both ungated heads retain the supported child-holding-donut and man-holding-frisbee tuples and exclude the four recently reviewed person-riding-person/standing-men-sitting-on-table errors. Both still select the previously contradicted holding-skis-at-feet tuple (`130024`).

After R50 gating and cap competition, the union control also selects the old eating-clock tuple (`361397`); the region head does not. Gating can change which candidates occupy the cap, so absence from ungated selected output does not establish rejection before capping.

A new purposive assistant spot-check of6region-selected claims on4images finds:

- `018250`: holding skateboard is contradicted by hands at the head, away from the board resting against the ground.
- `172901`: holding laptop is contradicted by the visible hand supporting the bird rather than grasping the laptop.
- `028517`: three cup/wine-glass holding tuples remain unclear because actors/hands overlap and detector classes alias the same drinking context.
- `046940`: holding tie is unclear; the tie is worn, no grip is visible and its lower part extends outside the frame.

These are method-visible assistant assessments, not independent human labels or random sampling. Do not estimate full100 precision from them, or count unreviewed claims as correct. No audit labels enter training.

## Decision

Keep both new heads as development artifacts and retain the current production pipeline. The region prototype has not demonstrated better whole-triplet acceptance than its matched union control. Coarse geometric ROI pooling is not a contact detector: contextualized14x14patches mix scene information, and subject/object rectangles can include distractors. Empty support is explicitly masked rather than treated as proof of absent interaction.

Before adding another acceptance-rule change, the next useful research step is to test whether the score actually responds to changing the supplied target instance while keeping the scene/subject fixed. Existing cached full100 candidates can support a label-free sensitivity diagnostic; independent labels would still be required to interpret it as correctness. No such new intervention is reported here.

## Artifacts and reproducibility

`run_interaction_region_cached.py` validates the uploaded ZIP against its report and old paired text caches, then trains both conditions without modifying prior heads/caches. The tuple/list conversion in comparing JSON reports is representation normalization only; no numerical validation tolerance was relaxed.

Union-control head SHA256: `699890b712497563d9d28deea157011f7e4afa39d46686ccd4ac92176da8e71a`.

Region head SHA256: `32484612d27752050c2a2693cff5b328d0ea6b2befaaa0c0fb9cc1fdbd270b3b`.

Committed compact evidence: `audit/interaction_region_paired_result.json`. Local output: `C:/KhoaLuanTotNghiep/Code/output/interaction_region_paired_v1`. Compact review ZIP: `C:/KhoaLuanTotNghiep/Code/output/interaction_region_paired_v1_review.zip`, containing bothheads/histories/metrics/prompt caches/predictions, paired summary, extraction validation report, assistant spot-check and result document. Frozen source feature tensors are not duplicated. Local training used CPU; no further Kaggle GPU run was needed. No caption improvement or scientific novelty is claimed.
