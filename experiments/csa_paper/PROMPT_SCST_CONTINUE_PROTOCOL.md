# PROMPT60: thêm tối đa một epoch SCST + CE — 08/11/2026

User approved continuation; current GPU budget reported13h for running account. PREPARING, not run. No report/TEST changes.

Source `ducanh3/notebook7bc83c4c76/60`, file `csa_prompt_scst_PROMPT_full1_v2/checkpoint.bin`, SHA `be2759b4a6f57af9bf8baeb13ee097bae2d8703d265b0791d9d39bb9c3dcbc63`. This is selected XE9 from the10-epoch study plus one full SCST/CE epoch, not XE10 weights. Restore model AND Adam state and source per-rank RNG. New epoch order seed43, deterministic hash pool42; rotate CE reference by one from each image's exact previous-epoch reference. No adapter reset or new architecture.

Keep same FP32/dropoutOFF/global8/twoT4, Adam lr5e-6/betas(.9,.99)/eps1e-8, CE+.05SCST TRAIN-only CIDEr.113287 TRAIN images once,14161 added updates; last4/3. Existing ROI134/prompt175/COCO/detector inputs unchanged. No object reward.

Original replay before any training must match source60 fullVALID rank0 CIDEr and exact caption/token files. Actual replay/gradient/DDP gates run by user. Validate original/added7081/added14161 on VALID5000, beam5/max20. Preserve source/original, MID and END. Same fixed .3D+.4CLIP+.3O, no new weight search. Choose max after-rerank CIDEr among checkpoints with CHAIRs/i<=original and recall>=original, ties earlier; original fallback. Require strict CIDEr improvement for promotion. Report all checkpoints and raw/reranked results. This isolates continuation within PROMPT, not a causal prompt ablation.

Four-hour parent STOP cap (not predicted runtime or billed quota); reserve score/rerank time, measured throughput stop, atomic checkpoints every500 updates and on cap. No auto restart or epoch extension. User runs notebook on Ngân account for own-source60 download; alternative local input accepts only exact SHA. Separate optional same-recipe resume is supported; no model/notebook/scorer execution by author. Local checks limited to metadata/NumPy/static/mock.

Question: can further training improve CIDEr while retaining current hallucination/recall? Improvement is not guaranteed; do not relax guards or choose TEST checkpoints after seeing results.

## Completion — Ngân69

Completed one added epoch. END fixed-rerank CIDEr improves +1.2623points and CHAIR flags151→148, but recall44.5944→44.3119%; MID also loses recall. Original60 is sole eligible checkpoint under the unchanged guard. Preserve MID/END as tradeoff candidates; no automatic extra epoch/TEST/guard relaxation. [Result](PROMPT60_CONTINUE_RESULT.md).
