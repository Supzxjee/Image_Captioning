# UPT alignment pilot B: region memory has not beaten A

Artifact audit: PASS. Matched A/B artifact comparison: PASS. These are single-seed screening results on the reused Karpathy val1000, using raw greedy decoding, not final test results.

| Model / inference condition | CIDEr ↑ | BLEU-4 ↑ | CHAIRs ↓ | CHAIRi ↓ | Object recall ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original checkpoint | 1.189066 | 0.350305 | 5.8% | 3.5452% | 45.6809% |
| A: continued-training control | 1.205513 | 0.351719 | 5.2% | 3.2475% | 45.8039% |
| B: UPT region memory on | 1.199067 | 0.353150 | 5.3% | 3.3129% | 45.7731% |
| B: same trained checkpoint, region memory off | 1.199913 | 0.352685 | 5.5% | 3.4167% | 45.9576% |

B is +0.010001 CIDEr versus its starting checkpoint, but -0.006446 versus the matched A control. Fixed-IDF paired bootstrap 95% intervals from saved per-image scores: B minus A [-0.023243, +0.010462], B on minus off [-0.006101, +0.004314]. Both include zero. B has not demonstrated an overall advantage; neither interval establishes inferiority. A's total CIDEr improvement cannot be credited to the new region branch in B.

## What the probes show

Disabling B's new region memory changes 66/1000 captions, reduces hallucinated captions from 55 to 53 when enabled, but does not improve overall CIDEr or recall. Swapping region values changes 19/1000 captions, with mean reference NLL delta +0.00001267. Text swapping changes none, as expected for B, which has no new UPT text input. The legacy prompt input remains present.

These probes suggest a weak observable effect under this short training budget. They do not prove the regions are unused or that UPT is wrong. The branch contains 1,187,840 trainable parameters and starts fresh, while the base is pretrained; matched updates do not imply matched convergence.

Among the 269 images with UPT-selected relations, CIDEr is 1.353118 for B versus 1.322335 for A (+0.030784). On the 722 object-only images it is 1.147047 versus 1.167390 (-0.020343). These are exploratory subgroup caption metrics using the full-val IDF, not direct measurements of relation correctness. Nine no-object images are too few for a strong conclusion.

## Validation and limitations

Both ranks completed 1250 updates / 10,000 total training images. Source checkpoint, initial base weights, inputs, cohort, training configuration (except arm/output paths), runtime manifests, original predictions and references match A. Saved source hashes match published training code. Loss/reward arithmetic, finite histories, equal logged cross-rank gradient norms, all scored-condition aggregates and sensitivity probes pass artifact checks. Recorded GPU smoke checks pass.

No checkpoint or staged features are included in this review ZIP. No inference replay or independent caption-metric retokenization was performed locally. CHAIR parser outputs were reaggregated; it measures object hallucination, not predicate truth. Bootstrap intervals are conditional on saved per-image scores and do not measure training-seed variation. No test images were used for these metrics.

Keep `/kaggle/working/upt_alignment_pilot_B/checkpoint.pth` separately; recorded SHA256: `634bbea1c185f2736fb672524ebfb86cca0f4ebc7f599455a1ed2ec427241d5d`.

## Decision

Do not promote B or spend a full-training budget on B alone yet. Complete D and compare it with both A and B. If D is promising, C will help separate text fusion from conditioned gating; C versus D still changes routing and parameter count. If D is also insensitive to text/regions, inspect the conditioning path and gradient scale before increasing training.

Evidence: [B artifact audit](B_result.json), [matched A/B comparison](AB_comparison.json). Auditors read artifacts only and do not launch notebooks.
