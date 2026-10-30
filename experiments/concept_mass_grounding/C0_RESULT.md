# C0 ablation result — 30/10/2026

C0 completed500updates from A with KL=.1,keep=0,negative=0. Both-rank preflight passed; actual weighted keep/negative gradients in all46logged rank snapshots are zero. Metadata-only audit274assertionsPASS. No neural execution locally.

| Model | CIDEr ↑ | CHAIRs ↓ | Hallucinated / all object mentions | Recall ↑ | Words |
|---|---:|---:|---:|---:|---:|
| A | 1.205513 | 52/1000 | 53/1632 | 0.458039 | 10.213 |
| C0 | 1.230741 | 52/1000 | 53/1646 | 0.460498 | 10.044 |
| C | 1.225678 | 51/1000 | 52/1632 | 0.458654 | 9.987 |
| G | 1.229535 | 52/1000 | 53/1633 | 0.457731 | 9.968 |

## What the ablation establishes

CIDEr gain over A is retained without positive preservation. C0 has the highest CIDEr point estimate of these endpoints, but C0 versus C/G differences are uncertain. Do not attribute the C−A gain specifically to preservation, and do not claim that preservation is generally harmful or equivalent to no preservation. C0 combines SCST continuation and rollout KL; their separate contributions are still not isolated.

C0 has52hallucinated captions and53hallucinated mentions, exactly the A totals. CHAIRi fell only because object mentions grew1632→1646. This is not evidence of fewer hallucinations. C has51captions/52mentions: a difference of one does not establish a dependable factuality benefit.

## Paired comparisons

- C0-A: CIDEr +0.025229,95%CI [0.006252924286127629, 0.043991879749774504]; CHAIRs +0.0000,95%CI [-0.008, 0.008].
- C-C0: CIDEr -0.005063,95%CI [-0.014019676794208376, 0.0034381572256716583]; CHAIRs -0.0010,95%CI [-0.003, 0.0].
- G-C0: CIDEr -0.001206,95%CI [-0.008234714656566064, 0.005984909957625966]; CHAIRs +0.0000,95%CI [0.0, 0.0].

Bootstrap:10000paired image resamples, seed20261029, sortedCOCO IDs. Small interval differences from notebook summaries reflect resample ordering/MonteCarlo variation; point deltas agree. These are conditional on one seed/reuseddev1000 and a post-pilot ablation, not seed robustness or confirmatory test results. G/C0 have identical hallucinated-caption flags on these1000images; their zero conditional CHAIRs interval does not establish population equivalence.

## Mechanism and run validity

Same4000uniqueTRAIN exposure IDs/order, metadata, object targets, policy, initial/frozen digests, tensor-content and prepared payload hashes as C. A predictions/per-image scores identical; preflight passed on both GPUs and frozen tensors unchanged. All archived source texts match runtimecorebb84be8 and declared hashes. Only the five predeclared implementation files differ from C; model and diagnostic code unchanged. Config, summary and history expose effective coefficients despite the shared policy template retaining keep_weight=.1.

Neutral-prefix70class separation: A.101457008,C0.106160565,C.106576544,G.106297197. C slightly exceeds C0 here, but this diagnostic does not establish caption-level benefit and is not causal grounding evidence. Weak-negative macro mass A.0001295722,C0.0001297802,C.0001310964,G.0001313372. Weak-negative annotations are not verified object absence.

## Next decision

Preserve C0 as a quality/control candidate and preserve all existing checkpoints. Current C/G objectives have no demonstrated distinct endpoint benefit; no validated new alignment contribution or dual-objective success. Sent compact verified facts to Astra for a concrete next decision. Do not auto-expand training, sweep coefficients, launch seed2 or use test as tuning data. A larger independent evaluation can test a locked choice, but does not on its own establish preservation attribution.

C0 checkpointSHA256: `5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d`. Keep `/kaggle/working/concept_mass_C0_v2/checkpoint.bin`; review ZIP contains no checkpoint/HDF5 payload to independently load. No such binary was loaded locally.
