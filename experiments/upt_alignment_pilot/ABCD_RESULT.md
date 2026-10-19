# Completed UPT alignment pilots A/B/C/D

All four artifact audits and pairwise comparisons PASS. No new branch has demonstrated an overall gain over continued-training control A in this single-seed screening experiment. Keep A as the pilot control; do not promote C or D, and do not start full training of the current gated design merely because D is close to A.

## Same val1000 and raw greedy policy

| Arm | CIDEr ↑ | BLEU-4 ↑ | CHAIRs ↓ | CHAIRi ↓ | Object recall ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1.205513 | 0.351719 | 5.2% (52/1000) | 3.2475% | 45.8039% |
| B | 1.199067 | 0.353150 | 5.3% (53/1000) | 3.3129% | 45.7731% |
| C | 1.192828 | 0.351616 | 5.4% (54/1000) | 3.3846% | 45.5887% |
| D | 1.202747 | 0.350297 | 5.7% (57/1000) | 3.5496% | 45.7424% |

Original checkpoint: CIDEr 1.189066, CHAIRs 5.8%, CHAIRi 3.5452%, object recall 45.6809%. Improvements versus that checkpoint must also be compared with A, which receives the same continued training.

A: no new UPT memory. B: visual region memory. C: region memory plus frozen CLIP text via additive fusion. D: text-conditioned scalar gates over region values. All arms retain the legacy prompt input; A is not literally a prompt-free architecture.

## Paired comparisons

Saved per-image CIDEr, 5000 paired bootstrap replicates, seed 20261019, fixed full-val IDF:

| Difference | CIDEr delta | 95% interval |
| --- | ---: | --- |
| C - A | -0.012685 | [-0.030735, +0.005049] |
| D - A | -0.002766 | [-0.022477, +0.015199] |
| C - B | -0.006239 | [-0.023079, +0.010825] |
| D - B | +0.003680 | [-0.015212, +0.022671] |
| D - C | +0.009919 | [-0.009871, +0.030226] |

All intervals include zero. Neither superiority nor equivalence is established. These are exploratory comparisons on reused validation data, with no correction for multiple comparisons and no training-seed uncertainty estimate.

## Intervention evidence

| Arm | On CIDEr | Off CIDEr | Captions changed by off | By text swap | By region swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 1.199067 | 1.199913 | 66 | 0 (expected) | 19 |
| C | 1.192828 | 1.191955 | 72 | 12 | 7 |
| D | 1.202747 | 1.205174 | 65 | 0 | 7 |

C on-minus-off: +0.000873, 95% interval [-0.008263, +0.009747]. D on-minus-off: -0.002427, interval [-0.011783, +0.006679]. Off uses the SAME trained checkpoint with extra memory masked, not an independently trained control. D off has 56 hallucinated captions versus 57 with memory on; C off has 51 versus 54 on. Turning the branch on does not show a useful overall gain under this budget.

D's text swap changes no decoded caption, but its reference NLL is not exactly invariant (mean delta -0.000000951). Region swap changes 7 captions, reference NLL mean delta +0.000008355. These are sensitivity probes, not verified false relations or calibrated grounding tests. Unchanged greedy captions do not prove complete absence of text use.

The 269 UPT-relation images have CIDEr A 1.322335, B 1.353118, C 1.348492, D 1.311989. D does not retain B's exploratory subgroup improvement. CIDEr on this subset does not measure predicate or instance correctness. The no-object subset has only 9 images.

## Is D disconnected?

The archived code matches the published model. Its text projection feeds the gate MLP; the gate is applied AFTER LayerNorm. Logged D branch gradient norm is nonzero on all 1250 steps (median 0.027873), although this aggregates the whole branch and does not prove each text/gate parameter receives useful gradients.

D's per-image valid-slot mean gates range 0.2771–0.3715, median 0.3034 (initial sigmoid value was 0.5). Text swapping changes those image-mean gates by mean absolute 0.00660, maximum 0.04602. Thus the gate computation responds to text, but that response has very weak observable influence on decoded captions. These are image-level averages, not individual-slot ranges or calibrated confidence values.

Interpretation: the model may learn to attenuate a redundant or unhelpful branch while relying on the strong existing memory. This is a hypothesis, not a demonstrated cause. Other possibilities include limited training, noisy pseudo-relations, feature pooling limitations and insufficient conditional supervision. Training only optimizes CE plus caption/object-reward SCST; it has no explicit pairwise alignment objective. We should not call the learned scalar gate a verified semantic alignment mechanism.

## Validation and checkpoint preservation

C and D each completed 1250 synchronized updates on 10,000 train images, 2xT4. Initial base digest, source inputs, expected hashes, stage/cohort, references, original predictions and runtime manifests match A/B; protocol differs only by arm/output/stage paths. Source files, loss/reward arithmetic, finite histories, rank gradient norms, metric aggregates, CHAIR reaggregation and all saved sensitivity summaries pass artifact checks. Train loops: C 509.76s, D 499.88s, excluding setup and evaluations.

Keep checkpoints separately:

- C: `/kaggle/working/upt_alignment_pilot_C/checkpoint.pth`, recorded SHA256 `f7a18ddb8d52e094cc7689ec754f15da7edcbea9131fccdfe2ca51647e1b0a09`.
- D: `/kaggle/working/upt_alignment_pilot_D/checkpoint.pth`, recorded SHA256 `5c8a758efe906731c2313aec905493b77750af0425968f9cadf4e46b88466f58`.

No checkpoints or staged HDF5 are supplied in the review ZIPs. No local model replay or independent caption-metric retokenization was performed. Bootstrap uses saved fixed-IDF scores; CHAIR parser outputs are reaggregated, not retokenized. No test images were used; these are reused-development results, not final paper evidence.

## Recommended next experiment, not launched

First run a small conditioning diagnosis rather than full training: log gradients separately for region/text/gate submodules and gate distributions by role; evaluate matched and deliberately mismatched text/regions with attention to actual hard negatives. Test whether a briefly frozen base lets the new branch learn a useful conditional signal, with an equal-budget control. Freezing alone is not a contribution or guaranteed improvement.

If the text path remains weak, design an explicit matched-versus-mismatched region/text learning objective before another full run. Such a loss needs reliable positive pairs and carefully chosen negatives; UPT predictions are pseudo-labels, not ground truth. This is a proposal to validate, not an implemented or successful method.

Evidence: [C audit](C_result.json), [D audit](D_result.json), [A/C](AC_comparison.json), [A/D](AD_comparison.json), [B/C](BC_comparison.json), [B/D](BD_comparison.json), [C/D](CD_comparison.json), [branch diagnostics](branch_diagnostics.json). `compare_ab_review.py` now accepts any matched pair without changing training code. No notebook was launched by the auditor.
