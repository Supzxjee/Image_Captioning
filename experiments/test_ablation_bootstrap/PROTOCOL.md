# Paired bootstrap of the completed TEST ablation

Decision recorded 24/11/2026, before resampling. The user approved the two comparisons proposed in chat. Aggregate TEST results were already known: this is a retrospective uncertainty analysis, not a preregistered confirmatory study.

## Question and fixed comparisons

Does the observed advantage of the fixed full-prompt checkpoint persist under resampling of TEST images?

1. Full prompt minus zero-text M3: prompt text contribution, with adapter/ROI/geometry retained.
2. Full prompt minus M2-only: contribution of the entire M3 addition, not text alone; parameter counts differ.

Use only CE10 seed42 top1 beam5/max20 captions, before SCST/reranking. No checkpoint/policy selection using these intervals. Full checkpoint SHA b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef; zero-text v193 checkpoint, TEST v198; M2-only v245 checkpoint, TEST v248. Reuse saved JSON and SHA-seal the exact source files. Check identical 5,000 unique image IDs, references, caption-to-CHAIR linkage and audited aggregate scores/counts before statistics.

## Estimator and resampling

CPU NumPy aggregation of saved scores/counts only: no Torch, model, metric scorer, Java, tokenization, Kaggle execution or new captions.

- Draw 5,000 image indices with replacement, using the same draw for all three arms.
- 100,000 replicates, NumPy PCG64 seed 20261124; batch size 64 only bounds memory.
- CIDEr: mean saved per-image CIDEr times 100. Reference document frequencies remain fixed to the original TEST corpus. This is conditional fixed-reference-IDF uncertainty, not CIDEr recomputation with newly estimated document frequencies.
- CHAIRs: mean of the per-caption any-hallucinated-object indicator times 100.
- CHAIRi: sum hallucinated mentions divided by sum generated object mentions, times 100, recomputed in each replicate. Never average per-caption ratios. Reject a zero denominator rather than silently impute it.
- Report full minus comparator. Positive CIDEr and negative CHAIR differences favor full prompt. CHAIR differences are percentage points.
- Report percentile two-sided 95% pointwise intervals. Also report percentile 99.1666667% intervals (alpha .05/6) for Bonferroni control over the pre-fixed six comparison/metric endpoints, subject to bootstrap coverage approximation. Do not label unadjusted intervals as simultaneous.
- No p-values. Report paired CHAIRs transitions as scorer flags, not human-confirmed repaired errors.

## Checks and limits

Synthetic checks cover pairing despite row order, duplicate/missing ID rejection, zero-denominator rejection, ratio-of-totals, identical-arm zero differences, deterministic resampling, and shared draws. Reconstruct all point estimates and counts before resampling.

Intervals are conditional on three trained checkpoints, fixed captions/scorers/reference IDF, and treating images as sampling units. They do not measure training-seed variance, prove causality across all architectural changes, establish no quality degradation, or validate relation/action/attribute hallucination. TEST has prior development exposure. This does not restore an untouched held-out test set. A confidence interval crossing zero is inconclusive, not proof of no effect. No new training or Word edits are authorized by this analysis.

Save compact paired per-image scores/counts plus provenance for reproducibility, code, checks, result JSON/report; update the existing results sheet and repository.
