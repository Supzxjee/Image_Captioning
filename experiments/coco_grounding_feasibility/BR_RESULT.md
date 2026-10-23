# Matched B/R/A result — 2026-10-23

**Both arms completed; neither a locked caption gain nor a bbox-position benefit is established.**
Source/protocol/history/metric audit passed for both1250-update T4×2 runs.
B/R have identical source bundles, staged feature SHA, actual image order and span schedule.
Protocol differs only in arm, output/stage path and preparation time.
Both use the same17,904 spans/person4,476 over10,000 images. No test or reranking.

| System | CIDEr raw ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Hallucinated captions |
|---|---:|---:|---:|---:|---:|
| Original | 1.189066 | .058 | .035452 | .456809 | 58/1000 |
| A continuation | 1.205513 | .052 | .032475 | .458039 | 52/1000 |
| B all-spatial-path loss | 1.203607 | .050 | .031098 | .459576 | 50/1000 |
| R bbox-path loss | 1.212063 | .050 | .031385 | .455579 | 50/1000 |

| Paired comparison | Mean CIDEr difference | Image bootstrap95% CI |
|---|---:|---:|
| B−A | −.001905 | [−.023282,+.019980] |
| R−A | +.006550 | [−.015988,+.029683] |
| R−B | +.008455 | [−.011348,+.028581] |

20,000 paired resamples,seed20261023, same1000development IDs.
All intervals cross zero. R−A also misses the fixed+.01 minimum.
CHAIR/Recall safeguards pass for R relative to A, but do not override the failed CIDEr gates.
R has a higher mean CIDEr than B while B has slightly better CHAIRi/Recall;
this pilot does not establish a statistically clear superiority of either auxiliary loss.

Both B and R have50hallucinated captions, but43are shared and7differ in each direction.
Relative to A, either arm clears15A-only cases and introduces13new cases; net−2.
Do not interpret a reduction from52to50 as solving all original errors.
All1250rank0 preclip norms exceed1 in both arms; A also clips at every update.

**Recommendation:** retain A as the established caption baseline. Do not promote R,
scale the same recipe, or ask for more blind labels solely on these results.
Report this pilot to Astra and choose a diagnostic or alternative mechanism that can
distinguish useful visual evidence from increasing attention mass.
These results do not prove all visual-semantic alignment approaches are ineffective.

Limits: reused development1000, one seed; image bootstrap does not measure training-seed
variance. Actual B/R image order matches; actual historical A order was not byte-logged.
Weak noun/class attention routing is not verified causal grounding or relation truth.
Audit recalculates stored data without local model execution; checkpoint SHA values are
reported, not independently verified from checkpoint bytes.
See `BR_RESULT.json` and `audit_BR_result.py` for assertions and calculations.
