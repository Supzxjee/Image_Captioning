# Batch-level feature sharing on restored CSA structure

10/11/2026. User approved trying batch-level feature sharing. Keep FP32, native PyTorch DDP/T4×2, all model layers/banks/prompt masks/decoder unchanged. Reuse TRAIN256 smoke stage and four existing Datasets. No Accelerate, FP16, full training, VALID/TEST, CIDEr, new cache extraction or pretrained caption weights. Last budget6hown/30hNgân; bounded75min parentcap including preparation, not ETA/billing.

## Semantics

Deduplicate by caption_image within EACH GPU's local batch. Stage patches and prompt tensors once per unique image, execute2OSA+1ORA and prompt adapters once, then index_select the TWO packaged memories back to caption order before the ordinary decoder. Do not deduplicate across ranks or detach memory: backward must sum all caption contributions into each unique image computation. Caption count, target tokens, loss normalization and decoder work are unchanged. Global token-normalized XE uses world_size/global_target_tokens to compensate DDP gradient averaging, including unequal tail16/11.

Uniform random-caption batches may have almost no repeated images; sharing is not guaranteed to help. Measure both original random-caption order and image-grouped order on the SAME1280TRAINcaption events. Image-grouped order permutes image groups and retains every caption exactly once; it is a proposed sampler change, not numerical equivalence to the historical random sampler. Ordinary vs shared within each order receives identical event IDs.

## Numerical checks and training caveat

User-run eval/dropoutOFF: compare logits, token loss and all parameter gradients of ordinary/shared DDP to full-global-batch ordinary reference on grouped32, groupedtail27, random32. Assert inverse reconstruction preserves event/image order. Hook encoder to prove actual row counts. No requirement for bitwise equality across batch sizes; FP32 tolerances: logits rtol1e-4/atol5e-5; global loss absolute5e-5; each gradient tensor relativeL2≤.01 and maxabs≤5e-5 when reference norm>1e-7, otherwise maxabs≤3e-7. Nonfinite or disconnected gradients fail the benchmark.

User-run timed training: dropout.2 remains ON. Sharing makes repeated captions use the same encoder-dropout realization for their image; ordinary mode samples separate realizations. Therefore ordinary/shared training trajectories are NOT claimed bitwise identical. Decoder dropout still sampled per caption. No caption-quality or learned-prompt-value claim from timing.

## Timing

Two orders ×four passes (ordinary/shared/shared/ordinary, ABBA) ×40optimizer updates; each pass starts fresh same seed42, Adam(.9,.99)/lr1e-4/eps1e-8/wd0/clip1, discards10timingwarmup steps and measures30.320optimizer updates total, no retained checkpoint. Alternate order per repeated mode controls simple warmup/order bias; tiny staged pool and accelerator scheduling remain limitations.

Synchronize ranks and CUDA before each timed update, include collation/H2D, forward/backward/DDP and optimizer; report slowest rank step time. Record encoded-image counts, caption/token counts, peak memory and per-step times/loss. Save partial progress and numerical errors before raising; no auto restart or extension. Speed ratio compares ordinary vs shared WITHIN each sampler. Groupedshared vs randomordinary also changes sampler and is labeled separately. Unfavorable ratios remain valid results.

Prepare user-run notebook and source pins; author only NumPy/mock/AST checks with Torch import blocked. Original smoke/model/results stay immutable. Actual numerical/performance checks are pending until user sends csa_feature_sharing_benchmark_v1_review.zip. Decide full-training sampler/sharing only after results; no long-run authorization inferred from this benchmark.
