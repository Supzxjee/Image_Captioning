# CSA adaptation — full TRAIN XE, first 10 epochs

02/11/2026. User approved full training for **10 epochs first**, with validation in the same notebook. Validation images are never training examples. This is the self-written CSA adaptation with cached CLIP ROI/text inputs, not exact published full-system reproduction or a new method.

## Locked run

- Fresh CSA weights; no Q-Former, A/C0, or 256-image pilot checkpoint. Base `model.py` unchanged.
- FullROI Dataset134 and semantic-bank Dataset130 only. Verify existing packages and stage read-only arrays once. TRAIN113287 images/566747 caption events, val5000 images/25010 references, test0. Original raw validation captions are for scoring only; inference starts at BOS, never a reference prefix.
- Ordinary two-GPU DDP, FP32, TF32/AMP off, dropout.2. Global32 **caption events**,16/rank; all references once per epoch. Image-grouped order seeded42+epoch as in the measured benchmark. Tail27 splits16/11. No shared-encoder training in this first baseline; within-batch sharing changes dropout correlations and saved only5.47% elapsed in benchmark.
- 17711 updates/epoch; endpoint10 =177110 updates/5667470 caption events. Adam betas(.9,.99), eps1e-8, decay0, grad norm clip1. Linear warmup from8e-6 at update1 to1e-4 at update53133 (three full epochs), then constant1e-4. No SCST/RL in this phase. These operational conventions are declared, not all directly specified in the paper.
- DDP loss `2 * local_sum_token_NLL / global_nonPAD_target_tokens`; EOS included, PAD excluded. No average of rank means. Verify current-environment benchmark loss/gradient guards before training; its models/optimizer are discarded.

## Validation and selection

After every completed epoch, run both teacher-forced NLL on all25010 references and reference-free **beam5** on all5000val images, partitioned across GPUs without duplicate padding. Beam score is raw cumulative log probability, max20 new tokens including EOS, BOS/PAD prohibited. Batched beam loop is checked on real inputs against the existing per-image beam implementation before the first training update. Eval mode only; learned image memory reused across the five beams within that evaluation.

PTBTokenizer + pycocoevalcap1.2 CIDEr on the entire fixed5000-image corpus; official implementation, natural score scale (not multiplied100). Choose greatest validation CIDEr, ties keep earlier epoch. CIDEr is not a grounding/hallucination metric. No test scoring, CHAIR claim or candidate reranking in this notebook. Save raw generated captions/references, NLL, CIDEr and timings each epoch for subsequent review. Metric dependency/Java preflight runs before expensive training.

## Persistence and budget

Atomic `checkpoint.bin` every1000 updates, at train-epoch end **before validation**, and after validation. Includes model, Adam state, epoch/batch cursor, counters/history/best metric, both-rank Torch/CUDA/Python/NumPy RNG states, source/input/policy/software identity. `best.bin` preserves best validated weights (full resumable checkpoint); latest checkpoint is the continuation source. Preserve **entire output folder**, including both checkpoint files/receipts. Optional RESUME_ROOT must point to that folder, verified against its receipts and identities, never silently restart or switch weights. Mid-epoch continuation regenerates the deterministic order, skips completed batches and restores both RNG states. Failed validation resumes validation without retraining the epoch.

Train-only projection ordinary2GPU =5.178h for10 epochs from short measured benchmark; validation/checkpoint/metric time is extra and mature decoding duration is not yet measured. Notebook wall cap10.5h (not ETA or a billing guarantee), graceful internal stop with3min headroom and checkpoint. Does not automatically proceed to15epochs/RL. A preflight runtime failure must stop before training; no tolerance loosening. Review ZIP excludes weights/feature arrays but contains source receipts, progress, evaluation records, captions and logs. User executes Kaggle; agent only static/NumPy/mock checks.
