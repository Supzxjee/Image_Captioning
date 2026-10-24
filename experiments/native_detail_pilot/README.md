# Shared native-detail H/L pilot — exact frozen A + one adapter

**Prepared for user-run Kaggle only. No local model inference/training or automatic notebook launch.**

Use `NATIVE_DETAIL_SHARED_HL_2GPU.ipynb`, **T4 ×2**, Internet ON, **Save & Run All**. One notebook trains one shared adapter; H and L are not separate models. The user reported30 remaining GPU hours. An operational8-hour wall budget, with a measured preflight estimate and1.5× training/decoding safety factor, limits this run; wall estimates are not a measurement of Kaggle's charged quota.

Required inputs (paths editable in the first cell):

1. User-supplied H/L cache: `/kaggle/input/datasets/ducanh2403/native-detail-hl-15k-datase/native_detail_HL_15k_dataset`.
2. Exact A: `ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`.
3. Legacy prompt tokens: `ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt`.
4. Original global CLIP visual cache: `ducanh2403/visual-cache`.
5. MSCOCO dataset JSON: `vuthetam/mscoco-2014/dataset_coco.json`.

No new UPT/detection/VG input or original-image encoding is needed. Metric dependencies and pinned CHAIR are prefetched/checked before staging/training. No Kaggle API token is needed.

## Locked mechanism and loss

The adapter receives original A queries32×512, cached quadrant tokens196×512 and their normalized global xy coordinates. LayerNorm is non-affine. Bias-free q512→128,k/v514→128,o128→512 matrices implement4-head attention: exactly**262,656** trainable parameters. Output projection starts at zero.

A short-lived forward hook replaces `Q_A` by `Q_A + adapter(Q_A,T,xy)` at the output of A's QFormer. The original encoder then recomputes its existing legacy prompt attention/gate/norm from that updated Q, and the unchanged decoder receives the original number of memory slots. The hook is removed after each memory construction; no rewritten baseline grounding code or cached grounded-prompt shortcut is used.

All A weights are frozen and its dropout remains in eval mode. **Autograd stays enabled through frozen legacy grounding and decoder** so CE gradients reach the adapter. H/L use identical weights/positions/token counts and receive no condition flag. The condition label is retained only for data selection and exposure auditing.

Full-caption CE only: five natural TRAIN references, BOS excluded as a target, PAD ignored, EOS included. If a30-token training sequence truncates before EOS, the last token is explicitly replaced by EOS. DDP loss scaling is `local_token_loss_sum × world_size / global_valid_token_count`; averaging rank-local means would be wrong with unequal caption lengths. No new SCST, object/presence/bbox/contrastive loss or reranker. Prior SCST knowledge stays in frozen A.

Four epochs,5,000 global updates, global batch8(local4), seed42, AdamW1e-4, weight decay.01, gradient clip1. Hash-sorted COCO IDs assign exactly5,000 H and5,000 L images per epoch; alternate that assignment by epoch. Every TRAIN image receives H twice and L twice, independent of caption content. DistributedSampler(seed42,epoch) supplies image order. Actual per-rank IDs, conditions and valid-token counts are logged.

## Input and resource preflight

The consuming notebook rehashes the30 actual HDF5 files and33 metadata files against the audited cache inventory, not just its completion flag. Exact A, vocabulary/tensor identity, COCO JSON, legacy token file, cache split/order and CHAIR cache pins are checked.

Legacy tokens are loaded once in a preparation subprocess. Only the locked15k original A visual/prompt rows are staged for both ranks: raw old visual FP16 remains byte-exact (cast toFP32 inside A as before), legacy tokens remainFP32, binary masks are stored asuint8. No extra quantization or new feature encoding. Grouped source-file/row access and persistent HDF5 handles avoid the earlier per-image metadata-lookup bottleneck. No duplicate H/L working copy is made.

Before5,000 updates, real-data checks require:

- Zero adapter output recovers original A memory, teacher logits and greedy tokens on the same batch.
- Global-token normalization works across two uneven rank counts.
- CE gradients traverse frozen downstream modules to all four adapter projections; A tensors remain unchanged.
- Initial adapter identities match both ranks.

Three warmup +25 measured optimizer steps on TRAIN estimate step cost. These28 updates and their optimizer state are **discarded**; adapter state/digest and seed are restored. TRAIN-only A/L/H greedy samples at the actual evaluation batch size estimate decoding cost; no validation quality is inspected to make a training choice. Model/optimizer/seed start fresh for the actual4-epoch pilot. If the conservative estimate exceeds the operational wall budget, stop before the main run and return the profile. No incomplete run is interpreted as a negative result for the method.

The measured estimate cannot guarantee full duration: warm/cold I/O, image/prompt distribution and final caption lengths can differ. The training/decoding watchdog preserves a partial head if budget is exhausted; a partial head is not a completed pilot and is not an optimizer-resume checkpoint. Preparation and official metric CPU time also matter. The full8-hour window is not a claim about the number of GPU hours Kaggle charges.

## Endpoint evaluation

Raw greedy max length30 on historically **reused val5000**, for:

| Condition | Memory |
|---|---|
| A | Exact original A, adapter bypassed |
| L | Shared endpoint adapter, degraded-detail tiles |
| H | Same endpoint adapter, native-detail tiles |

The inference loader does not encode or pass validation reference captions. References are accessed by the official evaluator after prediction, for full-val PTB/CIDEr and CHAIR. Same IDs, tokenizer, IDF and decoding policy for all three conditions. Report CIDEr, CHAIRs/i, recall, caption length and object mentions. Paired image bootstrap20,000 resamples, fixed seed20261023.

Prelocked investment gate: H−A≥.01 and H−L≥.005 CIDEr, both paired95% lower bounds>0; H CHAIRs/i no higher than A; recall at least A−.005. Verify final head identities across ranks, frozen A digest, all5,000 updates and exact40,000 image exposures(H20k/L20k;Htwice/Ltwice per image) before metrics are accepted. No endpoint selection, test, sweep or automatic scale-up.

Only `run_summary.json: status=PILOT_COMPLETE` after all prediction/metric checks means completion. ZIP creation alone is not success. Return `/kaggle/working/native_detail_shared_HL_pilot_review.zip`; keep `head.bin` **and original A checkpoint**. The small head uses legacy serialization to avoid Kaggle treating it as a ZIP; it is not a standalone captioner. Source/protocol/profiles/history/predictions/per-image metrics are in the review, model weights are excluded.

## Interpretation and validation limits

This follows Astra's selected information-source experiment. Positive H>A and H>L supports this native-detail processing path; it does not establish word/box or relation grounding, nor novelty. More views, attention and zero initialization are inherited components. Online inference H requires four crop encodes in addition to A's global encoding; cached decoding time omits that deployment cost. Fixed quadrants can cut objects, pooling can lose detail, and mixed H/L training can favor shared information. One seed and reused validation limit conclusions in either direction.

Local verification: five model-free checks pass (schedule balance/order invariance, global-token arithmetic counterexample, rejected wrong cache, source syntax and notebook/source hashes). Three Torch/HDF5 interface checks are deferred to Kaggle and run before expensive staging; actual A/GPU identity, gradient, timing, training and metrics remain unexecuted until the user runs the notebook. No accuracy gain or remaining-quota guarantee is claimed.
