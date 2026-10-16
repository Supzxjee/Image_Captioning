# Object Count v2 weight0.5: near-original fixed CIDEr, fewer hallucinated captions

Status: user-run comparison complete; local artifact audit PASS. The predeclared expansion gate remains FAIL: raw hallucination exceeds original, and fixed CIDEr remains slightly below original. No test run, automatic notebook submission or further coefficient search was performed.

## Same validation1000 results

| Model | Raw CIDEr | Raw CHAIRs | Fixed CIDEr | Fixed CHAIRs | Fixed CHAIRi | Fixed recall |
|---|---:|---:|---:|---:|---:|---:|
| Original QFormer | 1.218230 | 3.6% | 1.275819 | 3.8% | 2.5828% | 44.9544% |
| Matched SCST+CE | 1.230521 | 3.8% | 1.260458 | 3.8% | 2.6264% | 45.2373% |
| Count v2 weight1.0 (R123) | 1.230817 | 3.3% | 1.264003 | 3.2% | 2.1164% | 45.1430% |
| Count v2 weight0.5 (R124) | 1.233177 | 3.7% | 1.274298 | 3.3% | 2.2487% | 45.0802% |

The lower coefficient recovers fixed CIDEr by+.010295 versus weight1.0, with one additional hallucinated caption32→33 and two additional hallucinated mentions32→34. Total recognized object mentions remain1512. Raw hallucinated captions increase33→37, so the lower coefficient has a real tradeoff in this run.

Versus original fixed selection, weight0.5 has5 fewer hallucinated captions38→33 (13 removed,8 introduced), and5 fewer hallucinated mentions39→34. CIDEr is−.001520 lower, about0.12% relative. This small observed difference is not proof of equivalence/noninferiority. Versus matched SCST+CE fixed selection, CIDEr increases+.013841 and hallucinated captions38→33 (10 removed,5 introduced); correct unique objects decrease1439→1434, so recall is slightly lower than the matched control, but higher than original1430.

Raw versus original has one more hallucinated caption36→37. Do not describe the method as reducing hallucination across all decoding settings.

## Paired CIDEr uncertainty

5000 paired image bootstrap resamples, validation IDF held fixed:

| Comparison | Delta | Conditional95% interval |
|---|---:|---|
| Weight0.5 raw − original raw | +.014947 | [−.011391, +.041103] |
| Weight0.5 raw − matched control raw | +.002657 | [−.014189, +.019330] |
| Weight0.5 fixed − original fixed | −.001520 | [−.029106, +.025894] |
| Weight0.5 fixed − matched control fixed | +.013841 | [−.005696, +.033740] |
| Weight0.5 raw − weight1.0 raw | +.002360 | [−.017862, +.022598] |
| Weight0.5 fixed − weight1.0 fixed | +.010295 | [−.009750, +.030847] |

All intervals include zero. These are conditional on one checkpoint pair and this reused validation pool, not multi-seed estimates or evidence of statistically established CIDEr superiority. The5-caption reduction is descriptive; no significance claim is made.

## Audit evidence

- Pinned training core remains7a31327. Protocols differ from R123 only in object coefficient1.0→0.5 and output/target paths; training target bytes are identical. All2500 control history records match R123 exactly excluding elapsed time; raw/fixed control predictions are identical. Runtime and initial-model identities also match.
- Control first32 common trace fields match the prior strict same-GPU replay. This is trace evidence, not independent replay of all trained tensors. Checkpoint file hashes differ between runs because metadata/protocol paths differ; this does not itself show parameter divergence.
- Both arms2500 finite updates; CE+.05SCST and reward arithmetic checks pass.20k target rows,19885 eligible and115 neutral; no held-out reward rows.
- Official PTB/CIDEr/BLEU/ROUGE recomputed for all6 policies, plus CIDEr for prior weight1.0 treatment. METEOR cross-checked only. CHAIR reaggregated from supplied per-caption annotations. Cohort/input/original-caption matches and candidate journals verified.
- Source ZIP5860937bytes; SHA256 `8a8b5005e0a27f7d0a35067634840b2c1c3e595d6e4ac9379bd6af73e5d510c9`.
- Reported training seconds:1241.16 control,1038.96 treatment, excluding setup/evaluation.
- Recorded treatment checkpoint SHA256 `fc7dad03be986161fcd0669c1116503db2c82a5024ed87fc9e00460b962c0b06`. Preserve `scst_ce_object/checkpoint.pth` in `/kaggle/working/qformer_scst_ce_object_count_v2_w05_train20k_val1000`; weights absent from review ZIP.
- Original source data/detections and trained tensors absent; no independent end-to-end ground-truth/weight replay. COCO label support is not physical-absence proof. No relation-grounding contribution established.

## Decision

Keep both weight1.0 (lower hallucination) and weight0.5 (closer fixed CIDEr) checkpoints as exploratory candidates. Stop coefficient tuning on this same1000-image pool for now. Before any full-training expansion or test-based claim, validate stability with another training seed using locked weight0.5 and the same original/control comparison. If broader validation is later used, separate the already-inspected1000 images from the remaining4000 and report both; do not present tuning-set selection as independent confirmation. No next notebook was prepared or launched during this review.

Google Sheet synchronization is pending: the Google Drive/Sheets connector is unavailable in this session. Existing R124 and metric cells have not been changed; the local audit JSON contains the exact results for a future update.
