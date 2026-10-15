# QFormer CE versus SCST: completed train20k / validation1000

**Decision: expansion gate failed. Keep the original checkpoint and fixed reranker as the current reference. Do not expand CIDEr-only SCST or run test on this checkpoint yet.**

The user completed both arms from the same original QFormer epoch10, 32-query/2-layer checkpoint. Each saw the same 20,000 training images for 2,500 updates, batch8, AdamW5e-6, FP32 and dropout disabled. CE used five references; SCST used one sampled caption and its greedy baseline. Training budgets match images/updates, not FLOPs or caption counts. Both took about16 minutes in this run. Code core dabaca2; manual launcher d858fd5. No test evaluation.

## Same 1,000 validation images

| Generator and policy | BLEU4 | METEOR | CIDEr | CHAIRs ↓ | CHAIRi ↓ | Object recall |
|---|---:|---:|---:|---:|---:|---:|
| Original, rank0 | .365078 | .284004 | 1.218230 | 3.60% | 2.4523% | 44.0113% |
| CE continuation, rank0 | .366503 | .286617 | 1.220075 | 3.70% | 2.5728% | 44.0428% |
| SCST, rank0 | .372824 | .280445 | 1.230603 | 6.90% | 4.5249% | 45.1745% |
| Original, fixed rerank | .384152 | .294604 | 1.275819 | 3.80% | 2.5828% | 44.9544% |
| CE continuation, fixed rerank | .370457 | .293075 | 1.252264 | 3.70% | 2.5016% | 44.8915% |
| SCST, fixed rerank | .377294 | .288718 | 1.276707 | 5.40% | 3.3457% | 47.5322% |

Fixed means the existing decoder/CLIP/OCC weights .3/.4/.3, with no UPT or weight tuning. These are subset-specific results, not directly comparable to previous full-val5000 or test5000 numbers.

SCST rank0 increases CIDEr by .012372 over original and .010528 over CE. However hallucinated captions increase from36 to69:7 original errors removed and40 introduced. CHAIRi worsens as well. Mean caption length decreases from9.261 to8.760 words, so the increase is not explained simply by longer captions mentioning more things.

After the same fixed rerank, SCST increases CIDEr only .000888 over original, while hallucinated captions rise38→54:16 errors removed,32 introduced. Recall rises by2.578 percentage points, but BLEU4 and METEOR decrease. A .024444 CIDEr gain against CE-fixed alone is misleading because CE-fixed itself is worse than original-fixed.

## Uncertainty and training diagnostics

Paired 5,000-resample bootstrap intervals, keeping the validation-corpus IDF fixed:

- SCST rank0 minus original rank0 CIDEr: +.012372,95% interval[-.023134,.048134].
- SCST rank0 minus CE rank0: +.010528,interval[-.024390,.045593].
- SCST fixed minus original fixed: +.000888,interval[-.033180,.035272].

These intervals include zero. They describe caption/image variability for these fixed checkpoint outputs, not variation across training seeds or fresh datasets. Validation was inspected in prior work. This does not establish a reliable quality improvement.

All2,500 history entries are finite. SCST sampled training reward averages .7056 in the first250 batches versus .8936 in the last250; greedy reward averages1.3212 versus1.2615 on different image batches. This is not a held-out learning curve. Negative SCST loss is valid for negative sampled-minus-greedy advantage; it is not comparable numerically to CE loss. The stored gradient norm is measured before clipping. No training objective or sample masking was changed during this audit.

## Artifact verification

- Source ZIP SHA256: `2ceb81898b8650c39151b15e2e54dc20b20977c51806c029bdd36735b1b90732`,4,966,591 bytes.
- Protocol SHA256: `95acb277e6f9b0ff6e5d753c57d87995e4298d062e51182963a23f358ca3875a`.
- Source checkpoint metadata SHA256 matches the previously verified original checkpoint. Both training summaries reference the same source and protocol.
- Cohort IDs are disjoint20,000/1,000; training histories have exactly2,500 ordered updates and20,000 images seen. Full source dataset is absent locally, so split provenance is not independently reconstructed from the original dataset JSON.
- Three candidate pools each have1,000 ordered records; two500-image shard journals merge exactly. CLIP attachment preserves candidates and raw decoder scores; rank0 predictions match first beams; fixed predictions belong to their pools.
- CIDEr, BLEU1–4 and ROUGE-L recomputed using official PTB and matching to1e-10 for all six policies. METEOR cross-checked between saved reports only.
- CHAIR caption identities verified and corpus ratios reaggregated from supplied per-image annotations. This is not a fresh external-ground-truth CHAIR evaluation; fixed-rerank weights were not replayed without the full detector cache.
- Actual model weights are excluded from the ZIP; recorded checkpoint hashes are not independently rehashed or loaded here. Both checkpoints should be kept separately in Kaggle output.

See `audit/scst_comparison_train20k_val1000_result.json` and reproducible reader `audit_scst_comparison.py`.

## Proposed next decision, not implemented

The measured problem is now CIDEr optimization without sufficient object-consistency control. A useful next comparison would start again from the original checkpoint and test a CE-anchored SCST objective against an otherwise matched version with image-supported object penalties. Any negative object supervision must use reliable **training-only** annotation evidence; a missing detector prediction or missing caption mention alone is insufficient. First establish that stability and object consistency improve without erasing caption quality; then test the proposed instance/relation alignment term against the strong control. An object penalty or SCST alone is not a novelty claim, and CHAIR does not establish relation correctness. Do not stack several new terms into one arm and attribute any gain to relation alignment.
