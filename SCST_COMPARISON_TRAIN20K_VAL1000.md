# QFormer: matched CE continuation versus SCST

Status: PREPARED_NOT_RUN. The user launches Kaggle manually. No new performance claim.

## Question and fixed protocol

Does sequence-level CIDEr optimization improve the caption generator beyond continued cross-entropy training? This is an established training control, not the proposed scientific contribution. Do not add relation rewards before this comparison is measured.

- Start both arms from the original QFormer epoch10, 32 queries, 2 layers; frozen cached CLIP vision; existing original prompt cache. No E3, UPT or new relation cache.
- Select 20,000 train/restval images deterministically (seed 20261016), and 1,000 validation images (seed 20261017). Save every identity in protocol.json; exclude all test images. References used for training/reward come only from the selected training images.
- One pass, batch 8, 2,500 optimizer updates, AdamW LR 5e-6, default AdamW weight decay 0.01, gradient clip 1, FP32, seed 42. All existing trainable encoder/decoder parameters remain trainable; CLIP features stay frozen.
- CE uses the five reference captions; SCST uses one sampled caption and one greedy baseline per image, temperature 1, maximum 29 actions after BOS. Use the unnormalized sum of action log probabilities, include EOS, exclude all actions after EOS. No forced EOS at the length limit. Preserve the existing decoder's PAD/BOS probabilities.
- Disable dropout in BOTH arms, retaining gradients. Sampling, greedy baseline and differentiable rescoring therefore use the same policy. Generate without retaining rollout graphs, then recompute sampled-sequence likelihood with gradients.
- Budget matching means identical images, batch order, learning rate, optimizer and updates. It does not mean equal FLOPs, runtime, or reference/sample caption counts. CE continuation differs from historical training because dropout is disabled; disclose this when reporting.
- Reward: CIDEr-D with clipping, sigma 6 and x10 scaling, fixed image-level document frequency from the selected 20,000 training images, cached reference vectors. NLTK Treebank normalization avoids launching Java every batch. This differs from official Stanford PTB normalization. A unit test matches the official scorer on identically normalized text; evaluation always uses the official PTB pipeline.

## Evaluation

After both final checkpoints are fixed, evaluate original, CE and SCST on the same 1,000 validation images using unchanged beam5. Report both rank0 and the existing fixed decoder/CLIP/OCC weights .3/.4/.3. No new ranker, UPT or weight tuning. Compute BLEU1–4, METEOR, ROUGE-L, CIDEr and CHAIRs/i/Recall for all six policies. CHAIR evaluates objects, not relation correctness. Primary exploratory gate: raw SCST CIDEr beats both controls, CHAIRs/i no worse than either, Recall within 1 percentage point of the better control. Report fixed-rerank outcomes separately; do not select whichever decoding policy looks better afterward.

This is one seed and previously inspected validation data, not an untouched confirmatory set. Do not compare its CIDEr numerically to historical full-val5000/test5000 numbers. A positive result supports a subsequent full-validation and multi-seed comparison before a locked test run. A negative result triggers loss/reward/rollout diagnosis, not automatic repeated coefficient sweeps.

## Manual Kaggle use

Import SCST_COMPARISON_TRAIN20K_VAL1000.ipynb into **QFormer UPT Val5000 20261015**, or replace the old experiment cell with kaggle_scst_comparison.py. Do not append it after the old ranker launcher. Select GPU T4 x2 and Internet on; use the five existing inputs. Paths are editable at the start of the launcher.

Preflight runs only `tests/test_scst.py` via file discovery, avoiding third-party `tests` package collisions. The repository also declares its own test package. The actual unittest output is retained in `/tmp/scst_preflight.log` and included in any raised preflight error; training remains blocked on failure. A generic CalledProcessError from the old launcher alone does not identify which test failed.

GPU0 runs CE and GPU1 runs SCST independently; these are not two DDP ranks. Both GPUs shard beam inference afterward. HDF5 features are read in place, without copying 37 GB to /tmp. Concurrent input reads can limit speed; no runtime estimate is promised. The launcher prints startup logs, worker output and a 30-second heartbeat. Stop if a worker fails; no silent fallback to another checkpoint.

Return **/kaggle/working/qformer_ce_scst_train20k_val1000_review.zip**. It contains protocol, training histories, predictions, candidate scores, metrics and logs; it excludes model weights. Keep the two checkpoints separately:

- /kaggle/working/qformer_ce_scst_train20k_val1000/ce/checkpoint.pth
- /kaggle/working/qformer_ce_scst_train20k_val1000/scst/checkpoint.pth

The source checkpoint's optimizer is not resumed. Model-only checkpoints are written atomically at completion; interruption mid-training is not automatically resumable. Use a fresh OUTPUT for a new run. Never replace the input checkpoint store with validation output.

## Sources

- SCST: https://arxiv.org/abs/1612.00563
- Object hallucination / CHAIR: https://arxiv.org/abs/1809.02156
