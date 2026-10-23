# G/S completed: neither improves frozen A

2026-10-23. **Both run audits and the matched comparison PASS. Both caption gates FAIL.
Keep A and stop this fixed noun-evidence configuration.**

G and S each completed2,500updates/two train10k epochs with108,752trainable head
parameters and the entire A captioner frozen. G supervises region events with global
patch pooling; S uses fractional bbox-weighted pooling. Inference is identical in form:
both use global pooled class evidence to add a bounded bias to eligible noun logits.

## Same reused val5000, same raw greedy evaluation

| Measure | A | G: global loss | S: bbox loss |
|---|---:|---:|---:|
| CIDEr ↑ | **1.182248** | 1.172104 | 1.171573 |
| CHAIRs ↓ | **5.82%** | 7.52% | 7.12% |
| CHAIRi ↓ | **3.7827%** | 4.6593% | 4.4895% |
| Object Recall ↑ | 46.6035% | **47.9658%** | 47.5640% |
| CHAIR-flagged captions ↓ | **291** | 376 | 356 |
| Hallucinated object mentions ↓ | **307** | 398 | 379 |
| All object mentions | 8,116 | 8,542 | 8,442 |
| Recalled unique image/class pairs | 7,423 | 7,640 | 7,576 |
| Mean caption words | 10.1456 | 10.1010 | 10.1180 |

| Paired CIDEr difference | Mean | 95% image-bootstrap CI |
|---|---:|---:|
| G−A | −0.010144 | [−0.018156, −0.002271] |
| S−A | −0.010675 | [−0.018133, −0.003335] |
| S−G | −0.000531 | [−0.003947, +0.002888] |

20,000 resamples; seed20261023. Both A comparisons have intervals entirely below
zero on this reused validation set. S−G includes zero and fails the locked positive-CI
gate. This is not an equivalence test: absence of a detected S/G benefit does not
prove the methods are equivalent.

The +.01 gain requirement and no-CHAIR-increase safeguards fail for both endpoints.
Higher Recall does not compensate for worse CIDEr and hallucination. S has20 fewer
CHAIR-flagged captions than G, but65 more than A; this alone is not successful
caption alignment. TRAIN-mean/shuffle controls were correctly skipped after S/A failed.

Historical A CIDEr1.205513 used dev1000 IDF. Here A is evaluated on the same val5000
and IDF as G/S; all old dev1000 A captions reproduce exactly. The changed A metric
therefore must not be interpreted as baseline weight drift.

## The comparison is matched

Identical across arms: runtime source hashes, locked hyperparameters, label receipt,
reward targets, group mapping, **staged tensor bytes**, initial head state, frozen A
state, both rank runtime manifests and optimizer options. Every rank's actual image
order and auxiliary event counts match at each step. The protocol differs only by
arm, local label/output/stage paths and preparation duration.

Each single-arm audit verifies two complete TRAIN epochs,5,602unique regions/
11,204region exposures,1,597,744observed-label exposures, DDP mean-loss reconstruction,
5,000unique prediction IDs, saved CIDEr means/paired CI and CHAIR count reaggregation.
Both A prediction sets, per-image scores and CHAIR outputs are identical.

G/S generated captions differ on only253/5,000images (5.06%). CHAIR flags348captions
in both,28only in G and8only in S. Relative to A, G introduces121flags and clears36;
S introduces99 and clears34. These are automatic annotation-based labels, not a new
human inspection of the underlying images.

## What this does and does not establish

**Evidence:** both variants mention more objects, recall more annotated objects, create
more unsupported mentions and lower CIDEr. The negative result is not exclusive to
bbox pooling. This entire fixed recipe fails the practical caption-gain objective.

**Not established:** which shared component causes the failure. Possible explanations
include the prefix-independent noun bias, the prior reference, weak labels or competing
losses. The current experiments do not isolate them. Do not turn one of these
possibilities into a proven diagnosis or infer that all visual-semantic alignment fails.

Last100TRAIN updates: G/S mean absolute post-clip correction1.3942/1.0714 and saturation
40.33%/20.81% across all80image/class scores. Region-loss minibatch averages decrease in
both arms. These are useful diagnostic observations, not evidence of held-out recognition
accuracy or causal grounding, and not grounds for a post-hoc coefficient sweep.

## Decision and record

Keep A. Do not automatically launch a full run, additional seed, new gate or hyperparameter
sweep for this configuration. The next method/diagnostic should be chosen after reviewing
this matched negative result, with a distinct hypothesis and a new decision criterion.
A compact audited summary has been sent to Astra; it need not reread the ZIP files.
No new experiment has been authorized by a result alone or started here.

Full reproducible numbers and checks: [GS_RESULT.json](GS_RESULT.json),
[G_RESULT.json](G_RESULT.json), [S_RESULT.json](S_RESULT.json),
[audit_result.py](audit_result.py), [compare_results.py](compare_results.py).
Raw review contents are cached locally once and ignored by Git. The audit does not
execute ZIP source or load/train a model.

Limits: one training seed; historically reused validation, no test; image-bootstrap
uncertainty is not seed uncertainty. Checkpoint/head bytes are absent from the review
ZIPs, so reported tensor hashes cannot be independently recomputed from the attachments.
