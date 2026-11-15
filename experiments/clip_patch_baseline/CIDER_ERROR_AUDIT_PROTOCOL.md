# VALID caption error audit

Use the completed SCST1500 v94 CPU recovery ZIP. Join scores, selected ranks, image metadata and all five beams by COCO ID and verify the selected captions against the beams. Rank by saved final CIDEr with COCO ID as the tie-breaker. Sample 30 images uniformly from the lowest decile and 10 from the remaining 4,500 without replacement using seed 1500. Shuffle the review order. The comparison group is drawn from outside the low-score group.

Review the final caption against the image first, without scores, group, references or reranker details. Then review the alternatives in shuffled order and identify any clearly better caption. Keep uncertainty explicit. Low CIDEr is a sampling criterion; correctness comes from image review. The design is descriptive and cannot estimate population error prevalence or establish metric gains.

No new decoding, training or TEST. The outcome will guide the choice between studying errors in the candidate generator and studying errors in the selector.

## Returned review and findings

The user returned a 40-case export on 2026-11-15 (export SHA-256 `b11c0db2aeba1ca4cf46cfe4f544f83ea22727206b7bd6d708eee93044395d58`). All three fields on all 40 cases match the saved assisted labels. The returned export records no edits and does not record which cases were independently checked.

| Primary judgment | Bottom decile sample | Comparison sample |
| --- | ---: | ---: |
| Correct and adequate | 9 | 4 |
| Missing salient content | 9 | 1 |
| Incorrect or added object | 6 | 0 |
| Incorrect action or relation | 2 | 2 |
| Attribute or count | 1 | 1 |
| Language | 3 | 1 |
| Unclear | 0 | 1 |

Among the 21 low-score cases with an identified error, 8 had a clearly better saved alternative and 13 did not. In the comparison group the corresponding counts were 3 and 2 among 5 identified errors. The alternative can be an improvement while still incomplete. The groups were sampled by score and the comparison sample is small.

Suggested next check: for the 15 low-score cases with missing content or wrong objects, trace the visually verified object through detector output, prompt/ROI linkage, and all five beams. Use this to distinguish missing upstream evidence from available evidence the generator fails to verbalize. No model run or modification in this analysis.

## Fifteen-case evidence trace

Reuse the 15 low-score cases with a primary missing-content or wrong-object judgment. Inspect the exact v94 saved captions and reranker detection labels. A class label is evidence of a detection at image level, with instance identity and fitted boxes still to verify. Join the exact v175 prompt-link rows by VALID COCO ID. Absence of a linked slot does not imply absence of a visual ROI. No inference or training.

### Prompt–ROI join complete

[Extracted v175 rows](CIDER_ERROR_VALID15_PROMPT_LINKS.json); returned export SHA-256 `3d9f227b28123970db43c756850da5fbbd8ee0328cdb2581895c3032ba3bf75d`. The declared source digest matches the v175 pin. All 15 VALID IDs, object ROI bounds and relation endpoints verified. Fourteen images have no retained relation. The only two retained phrases are on image 29358 and refer to a donut, whose identity remains uncertain by inspection.

- 239894 and 487438: teddy bear and bed appear in saved reranker detection labels respectively, but the matching prompt classes are absent. The export contains retained links, so it cannot distinguish missing UPT proposals from filtered links.
- 436349: bus and truck both have slots; the review favors bus, while all five beams use truck.
- 82174: two bird slots and two chair slots are present; none of the five beams expresses both pairs. Slot counts alone do not prove correct instance or relation use.
- 72024 and 468858: boat and couch slots are present. A saved alternative mentions the omitted object, while the selected caption leaves it out.

These are selected error cases. The metadata localizes failure modes without measuring causal prompt effects. Proposed next check: inspect raw UPT proposals and YOLO ROI rows on the teddy bear and bed cases, then test adding one verified object–ROI link at a time on the same frozen checkpoint if a usable ROI exists. Keep other inputs fixed and use the saved original beam/caption baseline. No new inference or training performed.

### Two-case source check approved

Read the exact detector and UPT prompt caches that fed v175. Export the source boxes and class evidence for COCO 239894 (teddy bear) and 487438 (bed), reconstructing detector ROI order with the historical confidence sort, clipping and cap. Compare the count and all retained v175 object ROI boxes. The frozen UPT cache exposes previously selected proposals, so an absent class can be attributed to absence in that cache, with raw UPT model output still unknown. No model execution or prompt injection in this phase.

### Two-case source check complete

[Export](CIDER_ERROR_TWO_SOURCE_BOXES_RESULT.json). Exact pinned source IDs and the two retained v175 records match. In both cases ROI 1 is a usable detector box by visual overlay. For COCO 239894, teddy bear confidence is 0.9260 and the cached UPT object list contains only person. For 487438, bed confidence is 0.8526 and the cached UPT list has suitcase and backpacks. The target class is absent from both cached UPT lists, so there is no same-class UPT–YOLO IoU to compute. Raw UPT output before cache selection is unknown. Proposed frozen two-image check: append only the verified detector-derived noun slot and compare the five original versus modified candidates. No inference has run for that intervention.
