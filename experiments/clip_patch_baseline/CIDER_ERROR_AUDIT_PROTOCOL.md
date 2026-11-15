# VALID caption error audit

Use the completed SCST1500 v94 CPU recovery ZIP. Join scores, selected ranks, image metadata and all five beams by COCO ID and verify the selected captions against the beams. Rank by saved final CIDEr with COCO ID as the tie-breaker. Sample 30 images uniformly from the lowest decile and 10 from the remaining 4,500 without replacement using seed 1500. Shuffle the review order. The comparison group is drawn from outside the low-score group.

Review the final caption against the image first, without scores, group, references or reranker details. Then review the alternatives in shuffled order and identify any clearly better caption. Keep uncertainty explicit. Low CIDEr is a sampling criterion; correctness comes from image review. The design is descriptive and cannot estimate population error prevalence or establish metric gains.

No new decoding, training or TEST. The outcome will guide the choice between studying errors in the candidate generator and studying errors in the selector.
