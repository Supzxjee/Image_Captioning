# Author-style beam SCST, one full TRAIN epoch — 25/11/2026

User requests one epoch following the authors' SCST mechanism, then reports 16h GPU available. Prepare a user-run Kaggle notebook; never run models/Torch/scorers locally. This supersedes the earlier assumption that no GPU remains for this particular trial; zero-text seed43/44 remain queued, not cancelled.

Question: does changing the bounded SCST recipe to five-beam, within-image mean-reward baseline improve VALID CIDEr, while tracking CHAIR? Not a controlled attribution of the effect of beam baseline alone: duration, policy mode and optimizer rate also change versus the previous sampled/greedy experiment.

## Fixed recipe before code

- Start fresh from full-prompt seed42 best CE epoch9, SHA `b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef`, mounted at `/kaggle/input/datasets/ducanh3/upt-coco-full-cache`. Never start from SCST1500/3000, baseline or zero-text weights. No API/secret required for this mounted source.
- Reuse `/kaggle/input/datasets/ducanh3/csa-scst-full-karpathy-cache-v1`, original parts82/83/84 and all113287 unique TRAIN IDs in their sealed order. Read patch NPYs directly; no cache regeneration, full patch copies or encoder/detector inference. Materialize metadata only with all original SHA contracts. VALID5000 remains disjoint.
- One epoch means exactly113287 real image events, once each; effective global image batch32,3541 optimizer steps, final real batch7. No duplicated/padded image contributes to the objective. Preserve unused final slots only to keep DDP collectives balanced.
- Five-beam generation during **train mode with gradients**, max20, PAD/BOS forbidden as in existing decoder policy, EOS included and subsequent log probabilities zero. Fixed20-position denominator, matching the public `mean(log_probs,-1)` over a beam_len20 tensor. All five rewards from the same image define its baseline. Loss is negative mean-over-beams of advantage times mean token log probability. Pure SCST; no CE/CHAIR/object/CLIP reward or reranking.
- Fresh Adam, lr4e-6,betas(.9,.99),eps1e-8,weight_decay0; FP32, no AMP/compile, no gradient clipping (finite norm monitored). Original encoder/cache architecture retained; all existing trainable captioner/alignment parameters learn, frozen buffers remain buffers.
- T4x2 DDP. **Adaptation for memory:** four images/GPU/microbatch, four accumulation rounds gives global32, preserving the effective image batch. This is not an exact dropout/RNG replay of authors' single local batch16. No automatic batch/precision/optimizer fallback after OOM.
- Reuse our sealed cached TRAIN CIDEr-D reward implementation/tokenization/DF. It is not a claim of byte-identical tokenizer/scorer reproduction of the authors' public code. Original backbone differs too. This tests an author-style beam SCST adaptation, not paper reproduction or a promise of +13CIDEr.
- Endpoint fixed at one epoch. Compare endpoint VALID top1 beam5 with the already audited CE-best VALID reference; compute CIDEr/BLEU1/4/ROUGE-L/CHAIRs/i/Recall/NLL. No TEST, no rerank, no selection between partial checkpoints using TEST. Report all unfavorable metrics; no automatic second epoch.

## Guard, budget, continuation

Host only static/metadata/NumPy/mock checks. User-run preflight checks CUDA, checkpoint SHA, all source/cache/ID/vocabulary contracts, beam paths against existing eval beam search, EOS masking and autograd loss signs, per-image baseline, tail7 distributed weighting, finite gradients into core/M3, real peak memory, two warmup updates, one timed update in each of the three shards and one tail update (all six discarded). Use the slowest measured shard/rank update for the projection; this remains a small sample, not a completion guarantee. Training starts in a separate fresh process; discarded preflight updates never initialize the real run. If OOM/nonfinite/identity failure, stop with review, never change recipe silently.

Preflight prints measured throughput and a projection, not a guaranteed ETA/quota bill. A parent invocation cap11.5h reserves time for validation/checkpoint packaging. It is not a claim that10/16 quota hours equal wall hours or that the epoch fits a session. If the measured epoch projection exceeds14h, stop before full training for review. Save atomic latest checkpoint every100 updates, full optimizer/RNG per rank/cursor/identity; interruption writes a final partial checkpoint. Resume only this exact recipe/source/seed/cache/software and cursor. Do not restart the epoch or reuse preflight state. Mounted RESUME_ROOT is supported; if only Save Version remains, user supplies actual version for a pinned API continuation later.

If a cap interrupts training, keep the entire output, not only review ZIP. Evaluation of the completed endpoint can run separately if its phase fails; never retrain to recover CPU scores.

## Primary sources checked

- Paper, objective III-D / implementation IV-B:15CE+30SCSTepochs and CIDEr-D reward, DOI https://doi.org/10.1109/TCSVT.2023.3343520.
- Public `train_sc`: https://github.com/CrossmodalGroup/CSA-Net/blob/main/engine/caption_engine.py#L359-L457 — five beams, reward mean baseline, mean token logp, no CE term, `model.train()`.
- https://github.com/CrossmodalGroup/CSA-Net/blob/main/configs/caption/coco_config_csa_net_end_to_end.yaml — beam5/max20,SCSTlr4e-6,15/30epochs.
- Public train script reloads bestVALID each SCST epoch; this one-epoch adaptation begins at CEbest, with exact latest-state continuation inside that epoch. We do not copy the public per-epoch reload behavior into interruption recovery.
