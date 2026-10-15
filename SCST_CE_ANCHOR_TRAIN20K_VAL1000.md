# SCST with a CE anchor: matched train20k / validation1000

Status: PREPARED_NOT_RUN. User launches Kaggle manually. No new performance claim.

The completed CIDEr-only experiment raised raw CIDEr1.218230→1.230603 but CHAIRs3.6%→6.9%. After fixed reranking, CIDEr1.275819→1.276707 was essentially unchanged, with CHAIRs3.8%→5.4%. This experiment isolates whether adding reference-caption CE stabilizes sequence-level training. It does not implement object/relation rewards or claim a new scientific contribution.

## Two arms from the original epoch10 checkpoint

| GPU | Arm | Loss |
|---|---|---|
| 0 | scst | .05 × SCST |
| 1 | scst_ce | .05 × SCST + 1 × CE |

The SCST coefficient is the same in both arms. The sole objective difference between them is the CE term. The .05 scale is one provisional setting, not an optimal coefficient: observed unweighted SCST pre-clip gradient norms were roughly35–64 versus CE2.6–2.7, motivating a smaller RL contribution. Norms alone do not establish gradient compatibility. This control is a **scaled** SCST run, not a promised exact reproduction of the previous coefficient1 checkpoint; clipping and AdamW can make scaling affect updates. No weight sweep or held-out coefficient tuning.

- Same deterministic20,000 train/restval and1,000 validation images as the previous experiment. Same source checkpoint, source hashes, prompt and visual caches, all trainable modules, shuffle order, seed42, batch8, one pass,2,500 updates, AdamW5e-6, FP32, dropout disabled. No additional inputs.
- Reward remains fixed-training-IDF CIDEr-D using Treebank normalization. Sample/greedy settings unchanged. CE uses all five training reference captions.
- Accumulate RL and CE gradients from separate forward/backward graphs, then clip the **combined** gradient once and perform **one** optimizer step. This reduces peak graph memory compared with keeping both loss graphs alive. Log unweighted SCST loss, unweighted CE loss, total weighted loss, reward and combined pre-clip gradient norm.
- Both arms restart from the original epoch10. Do not continue the hallucination-regressed SCST checkpoint. No test images or new relation/hallucination reward.

## Fixed evaluation and gate

Evaluate original, scaled SCST control and CE-anchored SCST after training on the same1,000 validation images. Beam5, raw rank0 plus existing .3 decoder/.4 CLIP/.3 OCC reranking; same official PTB caption metrics and CHAIR. No UPT, new reranker or weight changes. Keep original checkpoint as the reference even if the scaled SCST control regresses.

Primary expansion gate fixed before running: anchored raw CIDEr exceeds both controls; raw CHAIRs/i no worse than either; raw recall within1 percentage point of the better control. Additionally, anchored fixed-rerank CIDEr and CHAIRs/i must be no worse than original fixed-rerank, with recall within1 percentage point. Report all outcomes, including failures; passing this exploratory point-estimate gate still does not replace uncertainty analysis or a subsequent full-validation check.

Do not compare subset scores directly to full-val5000/test5000. This is one seed on previously inspected validation; the CE anchor may fail or reduce CIDEr. No claim of less hallucination or better relation grounding is made before results.

## Run manually

Use SCST_CE_ANCHOR_TRAIN20K_VAL1000.ipynb in **QFormer UPT Val5000 20261015**, replacing the old launcher cell. GPU T4x2, Internet on, same five inputs. Explicit paths remain editable. The new launcher uses a fresh output directory and keeps prior results intact. Preflight runs the original11 SCST tests and three new anchor tests; errors retain the actual failed-test output.

Return `/kaggle/working/qformer_scst_ce_anchor_train20k_val1000_review.zip`. Keep separately:

- `/kaggle/working/qformer_scst_ce_anchor_train20k_val1000/scst/checkpoint.pth`
- `/kaggle/working/qformer_scst_ce_anchor_train20k_val1000/scst_ce/checkpoint.pth`

The review ZIP excludes model weights. Do not auto-submit or execute this notebook on the user's behalf.
