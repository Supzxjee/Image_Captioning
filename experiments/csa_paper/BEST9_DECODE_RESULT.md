# CSA epoch9: greedy versus saved beam5 — COMPLETE

User-run review on03/11/2026. ZIP SHA256 `10c75a46248b29435b5b55e667ac91fca3d3f504b8e79aa21df5a42b92668cd8`; runtime source `4c680ae60faa1073ddebc2af9b02e9f23bb3f9e6`. Same frozen best9 SHA f09e11c3…, unchanged model source139edbb1…, FP32/eval/max20; no training or TEST. Original16-image beam replay passed; all archived beam headline metrics/counts replayed. Detailed [audit JSON](BEST9_DECODE_RESULT.json); small receipts in `best9_decode_review_text/`.

## Full5,000 validation

| Measure | Saved beam5 | Greedy |
|---|---:|---:|
| CIDEr ↑ | **1.002933** | .943869 |
| BLEU4 ↑ | **.335528** | .290154 |
| ROUGE-L ↑ | **.535968** | .529880 |
| CHAIRs ↓ | **5.660%** | 8.040% |
| CHAIRi ↓ | **4.095%** | 5.262% |
| Object recall ↑ | 40.149% | **42.780%** |
| Hallucinated captions ↓ | **283** | 402 |
| False object mentions ↓ | **303** | 502 |
| Recovered class/image pairs ↑ | 6,395 | **6,814** |
| Mean words | 8.263 | 10.197 |
| Captions of at most2 words | 409 | 77 |
| Empty captions | 0 | 0 |
| EOS within20tokens | 4,925 | 4,558 |

Greedy changes3,653/5,000 captions. CIDEr decreases.059064; CHAIRs increases2.38 percentage points/119captions; false mentions increase199. Greedy repairs62 beam hallucinations but introduces181 elsewhere. More object coverage and fewer very short captions do not translate into better overall captions. Lower EOS rate means more outputs reach the20token cap; no max-length change was evaluated.

## Exact common4,000 development cohort

| Measure | Beam5 | Greedy |
|---|---:|---:|
| CIDEr ↑ | **1.017741** | .955679 |
| CHAIRs ↓ | **5.800%** | 8.050% |
| CHAIRi ↓ | **4.160%** | 5.199% |
| Recall ↑ | 40.316% | **42.998%** |
| Hallucinated captions ↓ | **232** | 322 |

Same direction as full5000. These are different cohorts/reference IDF; do not mix their CIDEr values. Reference, vocabulary, beam caption bytes and IDs were checked against the original139 text package.

## Interpretation and next step

Retain beam5 as current decoding baseline. This experiment rejects the simple practical hypothesis that replacing beam5 with greedy improves this checkpoint on CIDEr and object hallucination. It does not identify the root cause of short captions, prove the model/decoder implementation fully correct, or evaluate other decoding policies. CHAIR covers object hallucination, not arbitrary relations/attributes. One checkpoint and reused validation set, no independent test or multi-seed claim; this is not a new alignment contribution.

Recommended next discussion: resume the exact latest epoch10 `checkpoint.bin` with optimizer/state to complete15 XE epochs, retain beam5 and per-epoch validation, keep original best9 as candidate. Do not resume training from best9 merely because it has the best metric. More epochs are not guaranteed to improve quality; no15epoch trainer/run or RL/new-module budget was started by this review. After stable baseline, compare a separately approved alignment module against equal-training controls.

Checks:63 metadata/count/text checks and11 source Git hashes PASS, including5000 caption/ID coverage,16sample beam replays, token-to-caption consistency, source text identity, CHAIR raw aggregation, stored CIDEr vector means and old beam replay. No local Torch/model/official scorer execution; stored counts were audited. Independent workers report165.384 and166.353 seconds on T4, excluding setup, staging and CPU metrics; not billable GPU quota estimates. Binary checkpoint and ROI arrays were not supplied in review, their full bytes were validated by user-run code rather than locally re-opened.
