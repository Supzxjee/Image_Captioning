# S completed: the noun-evidence head degrades matched caption quality

G has subsequently been supplied and audited; see the final [matched G/S report](GS_RESULT.md).
The single-arm analysis below records the earlier S-only decision.

2026-10-23. **Audit PASS; S/A performance gate FAIL. Keep A.**
S completed 2,500 updates/two train10k epochs on T4×2. Both preflights passed, both
ranks report identical final head tensors, and the frozen A base digest is unchanged.
This is a completed negative experiment, not a notebook execution failure.

## Matched raw-greedy reused val5000

| Measure | Frozen A | S | S minus A |
|---|---:|---:|---:|
| CIDEr (0–10 convention) | 1.182248 | 1.171573 | −0.010675 |
| CHAIRs ↓ | 5.82% | 7.12% | +1.30 percentage points |
| CHAIRi ↓ | 3.7827% | 4.4895% | +0.7068 percentage points |
| Object Recall ↑ | 46.6035% | 47.5640% | +0.9606 percentage points |
| Captions flagged by CHAIR | 291 | 356 | +65 |
| Hallucinated object mentions | 307 | 379 | +72 |
| All object mentions | 8,116 | 8,442 | +326 |
| Recalled unique image/object classes | 7,423 | 7,576 | +153 |
| Mean caption words | 10.1456 | 10.1180 | −0.0276 |

S−A paired image-bootstrap 95% CI: **[−0.018133, −0.003335]**,20,000 resamples,
seed20261023. The interval is below zero on this reused validation set and seed;
it does not quantify training-seed uncertainty or establish a population-wide failure.

The locked minimum CIDEr gain +.01 and lower CI >0 both fail; CHAIRs/i safeguards
also fail. Recall passes its safeguard but cannot compensate for these failures.
The two inference controls were correctly skipped after S/A failed. There is no
reason to promote S, start a full run, add seeds or tune coefficients to rescue this
configuration using the same validation results.

A is evaluated here on the **same 5,000 images and full-val5000 CIDEr IDF** as S.
Its captions on all old dev1000 images exactly reproduce saved historical A.
The historical CIDEr1.205513 used dev1000 IDF; comparing that number directly with
the current val5000 score would mix evaluation sets/IDF.

## What changed

1,195/5,000 captions changed. CHAIR flags99 previously unflagged captions and clears34,
with257 flagged in both conditions. These are automatic annotation-based labels;
no new human image review has been performed.

The head increases both object recall and unsupported object mentions. Consequently,
the current correction is not selective enough to improve the captioner. This does
**not** identify whether presence loss, region loss, prior subtraction, SCST or the
prefix-independent correction is responsible.

Training diagnostics: mean region loss in the first/last100 updates is2.82866/0.56072;
last100 mean absolute post-clip correction is1.07143, with20.8078% of TRAIN image/class
scores at the ±2 bound (all80classes). These minibatch averages are not matched held-out
accuracy measurements. Reduced region loss is not evidence of useful grounding.
They motivate inspecting bias/calibration only if a subsequent decision justifies it;
they do not justify changing the frozen recipe after seeing this result.

## Audit and reproducibility

Model-free audit independently checked the cached archive contents against local
trusted sources and sealed labels:

* Exact source hashes and pinned protocol, label and object-target hashes match.
* Both rank receipts identify the correct A base, zero trainable base parameters,
  108,752 head parameters, initial/final head equality and off-parity after training.
* Every epoch contains each of10,000 TRAIN images exactly once globally.
  Every step's region/observed counts match the corresponding TRAIN rows.
* Total region exposures11,204 and observed-label exposures1,597,744 are correct.
* Reconstructing the mean DDP loss from CE/SCST/global auxiliaries differs by at most
  2.21e−7. Uneven local region counts correctly cause different local losses.
* Prediction IDs, rank merges, CIDEr per-image means, paired CI and CHAIR
  numerators/denominators reproduce the saved summary.

Review ZIP SHA256:
`d75aa187011d104fd329f777768d9b9df1a91c6b207c6ef4f2799702141dcaf8`.
Data is cached once locally; the audit never executes source from the ZIP or loads a model.
`S_RESULT.json` stores full values/hashes; `audit_result.py S` reproduces the audit
from the ignored local review cache. Head/checkpoint bytes are not in the ZIP, so their
reported tensor/file digests cannot be independently recomputed from this attachment.

## Next decision

**Keep A and stop promotion of S.** If G has already run/is running, collect its result
to separate generic evidence-head effects from bbox-supervision effects. G is not yet
supplied, so no S/G or spatial-supervision claim is possible. It is not necessary to
wait for G to conclude that this S endpoint fails the agreed A comparison.

Any new method should follow a joint G/S diagnosis and a new preregistered experiment,
not extra branches selected to improve this already-inspected validation result.
