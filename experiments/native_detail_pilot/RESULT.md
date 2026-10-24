# Shared H/L native-detail pilot: completed, investment gate failed

2026-10-24. Review SHA256: `7f36d9f4a6a28e7b331fe74f9b2425896611a06405e26a51f30fe97a3b69ee4b`.

| Raw greedy, reused val5000 | CIDEr (0–~3 scale) ↑ | CHAIRs ↓ | CHAIRi ↓ | Object recall ↑ | Flagged captions ↓ |
|---|---:|---:|---:|---:|---:|
| Frozen A | 1.182248 | 5.82% | 3.7827% | 46.6035% | 291 |
| Shared head + L (degraded detail) | 1.172661 | 5.62% | 3.6671% | 46.4905% | 281 |
| Same shared head + H (native detail) | 1.172088 | 5.52% | 3.6049% | 46.4905% | 276 |

H−A CIDEr = −0.010160; paired image-bootstrap 95% CI [−0.017489, −0.002744]. H−L = −0.000574; CI [−0.003721, +0.002551]. Both prelocked CIDEr improvement conditions fail. CHAIR/recall safeguards pass, but the joint investment gate is false. H and L produce identical caption strings on 4,679/5,000 images (93.58%). This similarity does not itself prove the adapter ignores detail; the matched H−L quality result provides no demonstrated benefit from native detail under this recipe.

H removes CHAIR flags on 47 images previously flagged under A, but introduces flags on 32 previously unflagged images: net reduction 15. CHAIR checks object mentions against its annotation-derived vocabulary, not relation correctness or human-confirmed grounding. Lower CHAIR alone does not establish a better captioner or a new alignment mechanism.

## Integrity and audit

Both ranks completed 5,000 updates, four epochs, 40,000 total image exposures. Each TRAIN image received H twice and L twice; complete unique 10k coverage each epoch and train/val COCO-ID disjointness independently verified from histories. The 262,656-parameter adapter is the only trainable component. Frozen A tensor digest and endpoint head digest match both ranks. Preflight records zero-init A memory/logit/greedy recovery, global-token DDP CE normalization, adapter gradients, and discarded/restored profiling updates as PASS. The submitted protocol and all archived source byte hashes verify.

Local audit reproduces all metric means, CHAIR aggregations and both exact 20,000-resample bootstrap intervals from saved predictions/per-image scores. It does not run model inference, training, or reinstall metric dependencies. The actual saved head and HDF5 payloads are absent from the small review; their on-Kaggle hash checks are supported by receipts, not independently repeated locally. The fixed original A CIDEr exactly matches the prior G/S comparison.

## Decision

Retain A as the stronger caption-quality baseline. Do not scale this unchanged adapter/CE recipe or launch a seed sweep: its prelocked gate failed. Report the negative information-source result and modest CHAIR tradeoff to Astra before choosing another GPU experiment. Possible causes include the training objective, bottleneck, pooling/fixed crops and mixed H/L learning; this run does not distinguish them and does not show that all detail-based or visual–semantic alignment methods fail.

One seed, historically reused validation; image-bootstrap confidence intervals do not measure training-seed uncertainty. No new test evaluation, reranking, checkpoint selection or automatic next run.
