# R pilot result — 2026-10-23

**R completed successfully; small mean gain over A, but locked numerical gate failed. B pending.**
User-run T4×2,1250updates,train10k,17,904 noun spans/person4,476; both ranks
passed corrected preflight and ended at identical model hashes. No inference parameters added.
Source hashes, main-loop equivalence apart from diagnostic logging, runtime/optimizer,
five-reference exposure counts, loss decomposition and metric reaggregation passed.
Original predictions and development IDs match historical A exactly.

| System | CIDEr (raw) | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Hallucinated captions |
|---|---:|---:|---:|---:|---:|
| Original warmstart | 1.189066 | .058 | .035452 | .456809 | 58/1000 |
| A continuation | 1.205513 | .052 | .032475 | .458039 | 52/1000 |
| R bbox routing | 1.212063 | .050 | .031385 | .455579 | 50/1000 |

R−A = **+.006550**, below fixed+.01 gate. Paired image bootstrap95% CI
**[−.015988,+.029683]**, crossing zero (20,000resamples,seed20261023).
R−original =+.022997, exploratory95% CI[−.000196,+.046716], also crossing zero.
The CHAIRs/i and Recall≥A−.01 safeguards passed; they do not rescue failed CIDEr gates.
193images gained CIDEr,201lost,606tied relative to A.

CHAIR transitions:15A-only hallucinated captions cleared,13newly hallucinated under R,
37hallucinated under both. The net improvement of two captions is not fifteen net fixes.
All1250rank0 gradient norms exceeded clip1 in both A and R; do not change clipping for one arm.

**Next:** obtain B with the same locked recipe. R−B is required to attribute any
benefit to bbox position rather than generic image-path routing. No promotion,
additional blind annotation or full-scale run on the strength of R alone.

Limits: reused development1000, one seed; historical A image order was not byte-logged.
Weak noun/class labels and attention mass do not prove causal grounding or relation truth.
Audit uses cached receipts/arithmetic, with no local model execution or independent GPU rerun.
Checkpoint SHA is reported in the archive; checkpoint bytes were not included and not verified locally.
See `R_RESULT.json` and `audit_R_result.py` for the reproducible calculations.
