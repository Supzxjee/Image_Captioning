# Same-GPU SCST+CE replay32: PASS

User-run Kaggle review independently rechecked locally. Both fresh-process runs completed32 updates on TeslaT4 with PyTorch2.11.0+cu128, CUDA12.8, deterministic algorithms enabled and math attention. No notebook was submitted by the assistant.

## Evidence

- Same original QFormer epoch10, pinned prompt/dataset hash identities, train20k/validation1k cohort protocol; unique and disjoint image/coco IDs.
- Objective unchanged: `CE + 0.05 * SCST(CIDEr)`, batch8, seed42, AdamW5e-6, FP32, dropout off. Reward IDF uses20k train images; only256 images receive updates per repeat.
- All32 steps match exactly for batch order, visual/prompt/caption/mask hashes, CUDA RNG, sampled/greedy token hashes, losses, gradient norms and post-step model hashes. Only elapsed time differs.
- Initial model identity: `99e61183c595be06eac112d6fc6ec0bfaafea5f809608a740f547e343a0bb3ee`.
- Final model identity: `6aa32fddc8b89229296965548391e4e1161326600b74dd007260c0035c4e1b0f`.
- Final optimizer identity: `b8fabd42a0c02ccd443567ab5843db4d01a32d4d1eb48ff92d0e88fc37cc3e48`.
- Training traces finish after35.295s and30.864s respectively; these exclude input/bootstrap setup and are not full notebook runtimes.
- Local audit recomputed comparison; worker protocol equivalence, history/log equality, objective arithmetic, RNG continuity and step32/final-state consistency all pass.

Source ZIP: `qformer_scst_ce_same_gpu_replay_32_review.zip`,758100 bytes, SHA256 `3fbda979023232693f0b50236ae0127c7e3ee8182e47a4c5c87f71436bebba70`.

Reproduce artifact audit:

```powershell
python audit_scst_replay.py --source "C:/Users/leduc/Downloads/qformer_scst_ce_same_gpu_replay_32_review.zip" --output audit/scst_ce_same_gpu_replay_32_result.json
```

## Interpretation and next experiment

Strict same-device reproducibility is demonstrated for these32 updates in this runtime. This does not establish the exact cause of earlier divergent controls, equality across devices/releases, or reproducibility of all2500 updates. Tensor hashes are reported by the GPU workers; weights and original datasets are absent from this review, so local verification checks artifacts rather than repeating GPU computation.

No caption-quality metrics were measured and no trained checkpoint was saved. Existing CIDEr/CHAIR results remain unchanged.

Next: prepare a controlled comparison against this CE+.05SCST control, replacing the diluted unsupported-object fraction with a penalty per unique unsupported class. Keep coverage reward; explicitly version the training parser fixes for turkey/desk. Use the strict runtime and fresh original checkpoint in both arms. Because both penalty and parser would change, report the result as a combined revised reward; isolating their effects requires an additional ablation. Do not increase the training budget or select on test data before validating the revised reward.
