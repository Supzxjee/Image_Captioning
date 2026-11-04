# CSA10 continuation: XE versus word–region supervision

Locked 2026-11-04 UTC+7 after user approved 2,000 updates per arm and VALID first.
This is continuation, not a fresh 10-epoch training or an exact GVD reproduction.

- Source model: exact original CSA epoch10 checkpoint `272f843a21b3a63c61323f748ed916fa3ae0cf2777549061439c07b410006950`,
  source139, mounted `ducanh2403/csa-full-xe10-v139/checkpoint.bin` (NOT best9).
- Data: sealed 10k TRAIN/restval cohort from CPU audit V2, exact output150 of
  `ducanh2403/2-vi-cai-kia-qua-nhieu`; 50,022 caption events,52,956 eligible mentions.
  Preserve all events; only valid mention positions receive grounding loss.
- Both arms: same model/weights, final object-attention instrumentation, seed42,
  grouped image order reshuffled per subset epoch, 64,000 events total, global32,
  ordinary two-rank FP32 DDP (no image-sharing/AMP), gradient clip1.
  Fresh Adam LR1e-5, betas(.9,.99), eps1e-8, weight_decay0; all existing trainable
  parameters updated, frozen noun/verb buffers unchanged. No old Adam/RNG resume.
- XE arm lambda0; GROUND arm lambda.1. XE normalized over global non-PAD tokens;
  grounding normalized over globally eligible mentions. DDP mean compensation
  applies separately; an empty-label rank contributes graph-connected zero.
- Grounding: final decoder layer's ACTUAL object Q/K, per-head pre-dropout masked
  log-softmax. `-log(mean_heads(sum_positive_regions probability))`; all positive
  regions for that category, no uniform-instance target. Same dropout/context
  calculation in both training arms. Inference restores original MHA forward.
  No new learned parameters; checkpoints retain original state_dict key schema.
- Evaluate fixed endpoint on all VALID5000, beam5/max20/original per-image decoder;
  separate teacher-forced NLL, CIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/counts/length.
  Keep endpoint, no best-checkpoint selection/sweeps/TEST generation here.
- Save latest atomic checkpoint each500 updates plus endpoint, Adam+both-rank RNG,
  event cursor/identity, receipts. Resume only exact same arm/protocol; do not
  restart automatically after failure. Deadline checkpoint preserves progress.

## Qualification and limits

CPU coverage passed97.91%, but the fixed100 cases have NOT been human verified.
User subsequently approved implementation/pilot without submitting human labels.
Amend the former mandatory human gate to an explicit exploratory **unreviewed
weak-label pilot**, not human-verified/training-gold data. Do not claim95/100passed.
Report this status in both outputs; retain review pages for later inspection.
This amendment does not lower IoU or change mappings after seeing metric gains.

Current budget user reports20h/23h per account. Notebook wall4h per arm is a hard
cap including setup/staging/metrics, not a runtime prediction or billing promise.
Source150 cache is saved by a CPU-only transport notebook before GPU use. No local
models/Torch/autograd or Kaggle runs by assistant; only metadata/NumPy/static tests.

Pre-GPU checks in notebook: stage/check exact inputs/label joins; user-run tiny
attention forward/backward parity, masking/empty-label/positive-mass tests and
two-rank gradient comparison with unequal/zero eligible-label counts. Both arms
use the same instrumentation; same-shape/dropout0 logits must match original.
No argmax/beam-ID equality across different batch sizes is required.

Continue screening if GROUND−XE CIDEr>=.01 and CHAIRs<=−.003, CHAIRi nonincreasing,
recall>=−.005, <=2word fraction<=+.01. Report paired image bootstrap CI after both
ZIPs arrive. Passing point gates is not automatically significant/novel, and
attention localization is not proof of causal visual evidence. No extra arms or
TEST tuning without discussing results. Old R-family failure remains recorded.
