# Semantic evidence pilot A/E — 19 October 2026

**Decision: keep A as the matched control; do not promote E or scale this version to full training.** E's alignment objective learns on training pseudo-pairs, but its visual memory has no demonstrated caption-quality benefit on val1000.

## Matched result

Both submitted runs pass the artifact audit: identical train10k/val1000 cohort, input/source hashes, source checkpoint, initial base state, seed42, two T4 GPUs, 1250 updates and decoding/evaluation settings. E adds its evidence module and 0.1 alignment loss. This compares the whole E package, not individual module/loss contributions. All 1000 A captions exactly reproduce the previous UPT pilot A submission.

| Condition | CIDEr ↑ | BLEU4 ↑ | METEOR ↑ | CHAIRs ↓ | CHAIRi ↓ | Object recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Source checkpoint, before continuation | 1.18907 | .35031 | .28537 | 58/1000 | 3.545% | 45.681% |
| A: continuation control | **1.20551** | .35172 | **.28727** | **52/1000** | 3.248% | 45.804% |
| E: semantic evidence enabled | 1.19682 | **.35218** | .28486 | 56/1000 | 3.452% | **46.326%** |
| E: same trained checkpoint, added evidence masked | 1.19942 | .35197 | .28485 | **52/1000** | **3.214%** | 46.265% |

E versus A: CIDEr -0.00870; recall +0.523 percentage points (17 more matched unique object mentions out of the same 3253 reference object instances counted per image), but four more captions with CHAIR hallucinations. E is above its starting checkpoint; A improves more. This is not an improvement over the matched control.

Paired fixed-IDF bootstrap, 5000 resamples of saved per-image scores:

- E minus A: -0.00870, 95% interval [-0.02744, +0.01026].
- E enabled minus E masked: -0.00261, interval [-0.00848, +0.00333].

Both intervals cross zero: there is no clear superiority, nor proof E is reliably worse across training seeds. Masking at inference tests immediate memory use; it does not undo E's training effects on base weights.

## What actually learned

Both ranks completed finite histories with consistent global gradient norms. Query, role, key, value, position and normalization parameters each have positive recorded gradient norms at all 2500 rank-steps. Gradient norms come from the combined objective, not isolated alignment gradients.

| Training window | Mean alignment loss | Pseudo-pairs scoring above both swaps |
|---|---:|---:|
| Updates 1–100 | .16567 | 55.21% |
| Updates 1001–1250 | .01747 | 96.85% |

The percentage measures a positive margin over the two constructed negatives, not achievement of the full .2 margin. These are different training batches and training pseudo-labels, **not held-out grounding accuracy**. All subject labels are `person`, so exact-phrase exclusion leaves no subject-role contrastive negatives. Wrong-image/different-phrase negatives can often be separated by object appearance; success does not establish discrimination of `holding` versus `riding`, or which same-class instance participates.

## Does caption generation use the added evidence?

Against E enabled, on the same 1000 images:

| Intervention | Changed captions | Mean reference NLL change |
|---|---:|---:|
| Mask evidence | 60 | +.00024447 |
| Swap evidence text | 8 | -.00000319 |
| Swap evidence regions | 52 | +.00016745 |

These are sensitivity probes, not verified counterfactual labels. Text swapping has weak observed output impact; this does not prove zero text use or zero attention changes. The exact swap policy is defined in the submitted source and remains a limitation.

The 269-image `object_relation` subgroup has E CIDEr 1.34600 versus A 1.32233 (+.02366), but E with evidence masked reaches 1.34651. Therefore this subgroup result cannot be credited to using the added relation evidence at inference. The 722-image `object_only` group drops from 1.16739 to 1.14639. Subgroup analysis is exploratory, not a selection rule or relation-correctness metric; the nine no-object images are too few for a reliable conclusion.

## Interpretation and next decision

**Evidence:** the new branch trains, yet has not improved the matched caption outcome. **Hypothesis:** separating broad pseudo-pairs is easier than learning the fine visual-language distinctions the decoder needs. Training loss alone cannot establish this cause; missing held-out grounding diagnostics and noisy UPT pseudo-labels prevent that attribution.

Keep the best existing caption checkpoint. Before another training run, test whether the frozen E matcher can distinguish same-object/different-action or wrong-instance alternatives on a small verified held-out set. This requires the E checkpoint and features; the review ZIP does not contain them, so this report does not claim to have run that evaluation. If it cannot distinguish them, revise the supervision/negative construction first. If it can but the caption ignores it, revise how the decoder uses the evidence. Do not add multiple training arms or launch full training merely because the auxiliary loss decreased.

## Reproduction and limits

Run `audit_review.py` separately on A and E with the pinned dataset JSON; then `compare_ae_review.py --a A.zip --b E.zip --output AE_comparison.json` and `audit_training_diagnostics.py --source E.zip --output training_diagnostics.json`. Audit outputs: [A](A_result.json), [E](E_result.json), [paired comparison](AE_comparison.json), [training diagnostics](training_diagnostics.json).

Audits validate ZIP CRC, submitted source/protocol/target/cohort hashes, true Karpathy split and all five references from the pinned local dataset, complete finite rank histories, loss/reward arithmetic, recorded GPU preflights, rank outputs, saved per-image CIDEr aggregation and CHAIR parser-output aggregation. Archived code is not executed. No model training or notebook run is launched.

Limitations: no checkpoint tensors or staged visual features in the review ZIPs, so no checkpoint/inference replay; metrics are checked from saved outputs without independently rerunning their tokenizers; one seed, repeatedly inspected val1000, raw greedy. Bootstrap does not quantify training-seed uncertainty. Historical best test5000 beam/reranked metrics are a different evaluation and must not be directly compared with this table. No test split was used in these two runs.
