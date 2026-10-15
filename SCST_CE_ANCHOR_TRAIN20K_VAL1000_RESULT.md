# CE-anchored SCST: completed train20k / validation1000

**CE substantially reduced the matched SCST control's object-hallucination regression. A reliable improvement over the original checkpoint is not yet established. The prespecified expansion gate failed; no test or full-training promotion.**

Both arms restarted from the original QFormer epoch10 checkpoint. GPU0 optimized .05SCST; GPU1 optimized .05SCST+CE. The training cohort, input identities, batch order policy, seed42, batch8,2,500 updates, AdamW5e-6, FP32, dropout disabled, reward and evaluation policies were fixed. No object/relation reward or UPT was added. Core2e20864/manual launcherc0128b3. Source weights themselves are absent from the review ZIP.

## Same1,000 validation images

| Generator/policy | BLEU4 | METEOR | CIDEr | CHAIRs ↓ | CHAIRi ↓ | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original rank0 | .365078 | .284004 | 1.218230 | 3.60% | 2.4523% | 44.0113% |
| .05SCST rank0 | .370120 | .275823 | 1.219472 | 7.70% | 5.0095% | 45.8661% |
| .05SCST+CE rank0 | .368413 | .286795 | 1.235314 | 3.70% | 2.4949% | 44.2314% |
| Original fixed | .384152 | .294604 | 1.275819 | 3.80% | 2.5828% | 44.9544% |
| .05SCST fixed | .381143 | .285439 | 1.287170 | 6.30% | 4.0248% | 47.1864% |
| .05SCST+CE fixed | .378600 | .296355 | 1.282620 | 3.80% | 2.5033% | 44.9859% |

Fixed is the unchanged decoder/CLIP/OCC .3/.4/.3 policy. These subset scores must not be compared directly with historical full-val5000/test5000 scores. The control coefficient .05 differs from R118's coefficient1; the current matched pair isolates adding CE, not whether RL scaling alone exactly reproduces R118.

## What CE achieved, and its limits

- Versus matched SCST rank0, CE anchoring reduces hallucinated captions77→37, removes46 errors and introduces6. CIDEr rises by .015842. Recall falls45.8661%→44.2314%, so improved consistency also comes with reduced object coverage relative to that control.
- Versus original rank0, anchored CIDEr rises by .017084, with hallucinated captions36→37 (8 errors removed,9 introduced). Recall rises only .2201 percentage points; no hallucination reduction relative to original is demonstrated.
- After fixed reranking, anchored CIDEr rises by .006802 over original. Both have38 hallucinated captions:13 removed and13 introduced. Object hallucination mentions decrease39→38; correct unique objects increase1430→1431, out of3,181 reference objects. Mean caption length9.633→9.885 words. This is nearly unchanged object coverage, not a demonstrated major grounding gain.
- The matched SCST-fixed control has higher CIDEr1.287170 but63 hallucinated captions. Anchoring trades some of that CIDEr for fewer hallucinations; do not select the control solely for its highest point-estimate CIDEr.
- BLEU4 after fixed reranking is lower than original (.378600 versus .384152), while METEOR is slightly higher. The outcome is not an across-metric improvement.

## Uncertainty and the unchanged gate

Paired5,000-resample bootstrap of per-image CIDEr, with the validation IDF held fixed:

| Comparison | CIDEr delta | Conditional95% interval |
|---|---:|---:|
| Anchored raw minus original raw | +.017084 | [-.010869,.044274] |
| Anchored raw minus matched SCST raw | +.015842 | [-.017432,.050627] |
| Anchored fixed minus original fixed | +.006802 | [-.020986,.035232] |
| Anchored fixed minus matched SCST fixed | -.004550 | [-.039305,.030503] |

All intervals contain zero. These are fixed-checkpoint, one-seed intervals over this previously inspected validation subset, not confidence intervals across training seeds or an untouched confirmatory dataset.

The raw CIDEr condition passes. Raw CHAIRs/i no-worse-than-both fails because anchored CHAIRs3.7% and CHAIRi2.4949% exceed original3.6% and2.4523%. The raw recall condition also fails:44.2314% is below the better control45.8661% minus1 percentage point. The additional fixed-rerank preservation condition passes. Do not relax those thresholds after seeing the result or mark the entire experiment a success; distinguish the useful CE stabilization observation from baseline superiority.

## Verification

Source ZIP SHA256 `b9652996cfeeda924a6fbdcaf2febfb5859a20cc950c4ec835803bf74ffb8505`,5,172,775 bytes. Protocol SHA256 `779c5be39b82e3c0ff2b48cf2bc5be0961d4c6b14e3071ba9a575b2628d44936`.

- Train20k/validation1k IDs disjoint and exactly match R118's cohorts. Input hash metadata and original raw/fixed captions match R118 exactly.
- Both histories contain2,500 finite updates through20,000 images. Recorded weights match the protocol. Every weighted loss matches .05SCST plus the CE term where applicable within floating-point tolerance. This checks logs and the executed source protocol; actual training gradients cannot be replayed without weights and features.
- Three1,000-record candidate pools have exactly matching2x500 shard journals. CLIP attachment preserves all candidates and raw decoder scores. Predictions and CHAIR caption IDs match their cohorts.
- Official PTB CIDEr, BLEU1–4 and ROUGE-L recomputed and matched for all six policies. METEOR cross-checked between artifacts only. CHAIR ratios reaggregated from supplied object annotations, not independently retokenized against external ground truth. Fixed-rerank membership checked, but detector inputs are absent locally, so the full selection formula was not replayed.
- Training summaries report about561.7s for SCST and617.8s for anchored SCST in this run; these are not transferable runtime guarantees. Checkpoints were not included in ZIP or independently loaded.

Reproduce with `audit_scst_comparison.py --source <anchor-review.zip> --prior-review <previous-ce-scst-review.zip> --output <audit.json>`. Structured results: `audit/scst_ce_anchor_train20k_val1000_result.json`.

## Next proposed research step, not implemented

Keep the original checkpoint as the reference and preserve the anchored checkpoint as a development candidate. The CE term is a useful stability component for a subsequent matched experiment adding a reliable training-only image/object consistency signal. It is not a replacement for semantic alignment, proof of relation grounding, or scientific novelty. No new coefficient sweep, full training or test is justified automatically by this point-estimate gain. A subsequent alignment arm must retain an otherwise identical anchored control and avoid labeling absent detector outputs or absent reference-caption mentions as definite contradictions.
