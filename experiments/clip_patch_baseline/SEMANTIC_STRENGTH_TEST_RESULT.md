# Strength-gated bank + prompt — TEST5000

Reviewed 10/11/2026. Complete fixed 5,000-image MS COCO Karpathy TEST; same references, patch-cache seals, vocabulary, PTB and evaluator assets as both comparison runs. Evaluation source: `ducanh2403/kltnnnnnff/14`, user confirmed. Weights: `ducanh2403/kltnnnnnff/12`, VALID-selected best epoch 8, SHA `456681d45b8b9ce8e3d34f3452c5f6b7bebdef21fd2ad3e6f344790ccbe4ccf9`.

| Metric | Plain patch baseline199 | Bank + prompt | Bank + prompt + strength gate |
|---|---:|---:|---:|
| BLEU_1 | 77.1623 | 76.9530 | 77.2950 |
| BLEU_4 | 37.1675 | 36.6553 | 37.1050 |
| METEOR | 28.7868 | 28.6547 | 28.5181 |
| ROUGE_L | 57.5850 | 57.3237 | 57.3146 |
| CIDEr | 120.5145 | 120.4520 | 120.9105 |
| SPICE | 21.6475 | 21.6008 | 21.5770 |
| CHAIRs | 5.1600 | 4.1600 | 3.9400 |
| CHAIRi | 3.5958 | 2.9299 | 2.8307 |
| Recall | 43.7254 | 43.8651 | 43.3761 |
| NLL | 2.1218 | 2.1235 | 2.1172 |

All metrics except NLL shown ×100; CHAIRs/i are percentages (lower is better). All models trained from scratch with the same 10-epoch XE schedule and selected on VALID; selected checkpoints are plain epoch10, bank+prompt epoch9, strength epoch8. No SCST or reranking in these TEST runs.

## Interpretation

- Versus plain baseline: CIDEr +0.3960 points; CHAIRs 5.16%→3.94% (258→197 flagged captions), CHAIRi 3.5958%→2.8307%. Relative CHAIRs reduction 23.64%, absolute reduction 1.22 percentage points. 142 previous flags removed and 81 introduced; a removed flag is not proof that the whole caption is now correct.
- Versus prior bank+prompt: CIDEr +0.4585 points, CHAIRs 4.16%→3.94% (208→197), CHAIRi 2.9299%→2.8307%. 84 flags removed, 73 introduced. This small incremental gain is not established as robust.
- Conditional paired bootstrap (2,000 image resamples, seed42, fixed reference IDF): versus plain, CIDEr delta CI95 [-0.9979,+1.9193] points and CHAIRs [-1.80,-0.64] pp. Versus bank+prompt, CIDEr [-0.9977,+1.8458] and CHAIRs [-0.72,+0.26] pp. CIDEr intervals include zero; gate-specific CHAIRs interval also includes zero.
- METEOR/ROUGE-L/SPICE and recall are slightly lower than plain baseline; BLEU-4 is slightly lower too. Recall 43.7254%→43.3761%; mean caption length 9.3906→9.3170 words, no empty or ≤2-word captions. The result supports object-hallucination reduction relative to plain baseline, not improvement on every semantic metric.
- Reported teacher-forced TEST NLL 2.121834→2.117182, distinct from the training objective and not a direct grounding-accuracy measure.

## Mechanism and limits

The model retains bank-region and image-specific prompt-region alignment and adds a learned scalar per patch: `M = V + sigmoid(W[V;delta]+b) * delta`. The older branch gate remains; it mixes object/relation semantic residuals. The new gate controls their combined strength. +1,025 parameters versus prior bank+prompt. These results compare complete systems and do not isolate prompt text, banks, or ROI masks.

Single training seed and prior project TEST exposure; conditional intervals do not account for model selection, multiple comparisons or seed variation, and are not equivalence/noninferiority tests. Do not tune this gate on TEST or claim pixel-level grounding correctness from CHAIR alone. Prefer matched VALID work for any further method change.

## Audit and provenance

5074 cached-text, source-pin, seal, caption/token, count and arithmetic checks passed. ZIP SHA `4945ce495dbb9cf777b370de64aa1e152baf399f4c44a27f7b754338720bbdca`. Inference source `29fa3bd7b0353941eb40038249a13b4e832c36f1` (28 pins), strict known checkpoint SHA; both user-run GPU beam-path checks passed. Source archive read once and cached outside Git. No local model/checkpoint load, Torch, official scorer rerun or Kaggle API call. CIDEr/SPICE saved per-image means and CHAIR counts replayed; other scorer values checked against saved phase/result provenance.

Staging 538.53s; longest inference worker 125.18s. These exclude setup/CPU scoring and are not GPU billing measurements.

Recommendation: retain this as a useful hallucination-reduction configuration. Do not launch another long train or label the small CIDEr increase significant on this evidence alone. No new experiment was executed or authorized by this result review.
