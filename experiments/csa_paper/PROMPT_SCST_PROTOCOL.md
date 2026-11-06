# Prompt CSA → SCST + CE → fixed reranking

06/11/2026. User approves continuing the prompt pipeline after the matched fresh-XE result. The goal is higher CIDEr while retaining the measured object-hallucination benefit, not claiming a gain in advance. Author prepares code; user runs Kaggle.

## Fixed comparison

- CONTROL starts from selected best10, own notebook `ducanh2403/2-vi-cai-kia-qua-nhieu/178`, SHA `3dbc24409259f2e1a062c674fd5cec495e12aa168bcc1b747c4c5fc1beec1c46`.
- PROMPT starts from selected best9, own notebook `ducanh3/notebook7bc83c4c76/58`, SHA `371f8d8ad4635617d0ca2a111aeba1fa81894631475d629ef81c31a71f26dc96`.
- Selection was locked before CHAIR. Do not silently replace PROMPT9 by PROMPT10. Initial XE histories differ by one epoch: this compares the two selected pipelines, not an exactly equal total training budget or an isolated causal text effect.
- Keep noun/verb banks, ROI134, full prompt175, CSA and learned adapters. CONTROL adapter remains frozen/bypassed; PROMPT adapter is restored, never reset to zero. CLIP/UPT cached features remain frozen.
- Both: **one complete TRAIN image epoch**, 113287 images, 14161 updates, nominal global8, final7 split4/3 without duplicate padding. Reuse `scst_full_data.full_schedule` and `scst_pilot_data.event_schedule`; same IDs/reference choice/order/seeds.
- Fresh Adam5e-6, betas(.9,.99), clip1, FP32, dropoutOFF; loss = globally token-normalized CE + .05 × image-mean SCST. Reward is TRAIN-reference CIDEr-D using all113287 TRAIN images for IDF; sample vs current-policy greedy. No object reward/new loss, no VALID/TEST references in training reward. No extra XE or automatic second epoch.

## Evaluation and output selection

- ORIGINAL and fixed END, each full Karpathy VALID5000, all25010 references for NLL. Original per-image beam5/max20/raw decoder scores. Save all5 candidates at both phases. ORIGINAL CIDEr must replay historical C10/P9 within1e-7 before the first optimizer update.
- Score CIDEr/BLEU4/ROUGE-L/CHAIRs/CHAIRi/recall/length for both phases; no TEST. Fixed END is reported even if worse. No posthoc checkpoint selection; preserve source weights, final weights, optimizer/RNG/cursor and ORIGINAL/END candidates.
- Apply **the same existing selector** at both phases: .3 minmax(decoder raw logprob) + .4 minmax(CLIPScore) + .3 normalized detector consistency, detector confidence>=.5, penalty1, earlier-rank ties. Frozen official CLIP ViT-B/16 revision57c21647, no BLIP/oracle/coefficient search. Caption references are excluded from inference inputs and only used for scoring after selection.
- Report four conditions per arm: ORIGINAL rank0 / ORIGINAL fixed rerank / END rank0 / END fixed rerank. Comparisons within-arm separate training from reranking; across arms report initial quality differences. CHAIR measures object mentions, not relation correctness. Prior repeated VALID/TEST exposure remains disclosed.

## Runtime and checks

### V2 operational amendment,06/11/2026

User reports PROMPT V1 successfully staged allROI119shards and TRAIN schedule, then hit the600s subprocess deadline in reranking `prepare`, before metric/model/DDP guards or training. No per-step log was present, so the exact slow substep is not established by this trace. This is not model divergence and no SCST checkpoint is evidenced by the printed filename. User explicitly requests a fresh run, not recovery.

V2 retains sourceXE178/58, model/loss/schedule, four-hour parent cap and selector. It replaces repeated per-JPEG path probes with one directory scan, indexes only selectedVALID detector entries after verifying the full source SHA, and hashes/verifies images with8I/O threads in stable COCO-ID order. Hash-stage/path-scan/progress logs identify where time is spent. Rerank preparation cap is1800s within the unchanged parent cap. Output `_full1_v2` preserves V1; missing checkpoint is now reported honestly. This is an I/O fix, not a new experiment/model or an increase in total parent GPU budget. Actual mounted-Kaggle speed remains unmeasured by author.

- Each notebook uses its own account's exact source version, never requests the other account's private output. Kaggle Secret `codex` (or configured name) only if downloading weights; accept a local mount with the exact SHA instead. Attach ROI134, prompt175, COCO2014 raw JPEGs+datasetJSON, original detectorJSON; no bank130 input needed because banks are inside source weights.
- User reports current quota10h/15h. Both notebooks use a4h parent wall STOPcap; this is not an ETA or Kaggle billing/quota guarantee. Preparation, guards, training, metrics and reranking all count against this cap. Runtime projection/checkpoint protects partial training; never mark partial as complete or restart training automatically.
- User-run loaded-model sampling/rescoring parity + finite/DDP-gradient checks discard their updates. Author runs only NumPy/AST/mock tests and validates the exact Git source bundle, not Torch/models/scorers/notebooks.
- Keep the current best historical pipeline. This experiment does not overwrite it or promise that its historical SCST/rerank gains transfer to PROMPT.
