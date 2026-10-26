# Stale prompt preparation — result, 2026-10-26

Preparation completed for **2,000 images / 58 target classes**. No optimizer step,
training pilot or new checkpoint. Cache is 2.58 GB and must be preserved in full.

## Frozen A diagnostic (128 fixed TRAIN images)

| Quantity | Result |
|---|---:|
| Mean factual prompt boost, b1 | 0.748723 |
| Mean occluded prompt boost, b0 | 1.647003 |
| Mean interaction, b1−b0 | −0.898280 |
| Interaction image bootstrap 95% CI | [−1.122926, −0.689220] |
| Negative interaction | 112/128 |
| b0 above locked 0.1 margin | 113/128 (88.28%) |

Scores are log probabilities of the locked one-token alias set at the same
reference prefix. CI uses 10,000 image resamples, seed42; this restricted TRAIN
sample is a diagnostic, not held-out caption efficacy.

| Input | Mean absolute alias probability |
|---|---:|
| Original image, original prompt | 0.609655 |
| Original image, neutral prompt | 0.411095 |
| Occluded image, original prompt | 0.477996 |
| Occluded image, neutral prompt | 0.218779 |

The effect is not confined to negligible absolute probabilities. It does not establish
that the residual prediction is wrong: masks preserve silhouettes/context and COCO
may omit instances. Removing words with `object` is an additional distribution shift.
No free-generated caption or hallucination endpoint has been evaluated here.

## Integrity and practical cost

11 present payload hashes, 48 panel file hashes and 10 runtime source hashes verified.
Three HDF5 payloads are intentionally absent from the review ZIP; their receipts
cannot replace direct hash verification when reused for a pilot.
Metadata + cache: 1,724.86 seconds (28.75 minutes), within the locked 30-minute budget.
Two T4 workers cached 1,000 images each.

50 forward/backward batches: C 6.84s, T 6.28s; peak allocated GPU memory approximately
0.924/0.928 GB. Projected arm times 386.68/363.59s are extrapolations only: optimizer
latency and actual full training/scoring wall time were not measured.
Cache parity checks passed the configured elementwise tolerance; they are not exact.
Maximum recorded text error 0.007744; original-image panel error 0.031189.

## Assistant visual review and next decision

Inspected all 16 panel images. Person82225/554374/275432 and cow213961 retain clear
silhouettes; surfboard568417 retains shape/context; suitcase10232 retains handle.
Car166391 appears to retain tiny cars outside annotated masks; this is a visual
suspicion, not a rerun of COCO annotation coverage. Not human-verified labels.

**Conclusion:** enough observable signal to discuss a bounded recipe experiment,
not evidence of effective alignment or caption improvement. Sent compact audited
evidence to Astra; no C/T pilot started. Independent Argon response not received.

ZIP SHA256: `c0fc32e3cebc336600e539b34a39250b0547d09e082e1a985829efc2c80ec565`.
