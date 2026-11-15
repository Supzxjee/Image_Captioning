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
