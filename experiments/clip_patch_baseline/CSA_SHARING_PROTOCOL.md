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


## Amendment — Accelerate, user approved10/11/2026 before GPU benchmark

Supersede the native-DDP benchmark notebook with an Accelerate-managed two-GPU variant. Keep FP32 (`mixed_precision="no"`), accumulation1/global32, exact same model/samplers/ABBA40×8 and numerical guards. No FP16 or speed claim. Old notebook stays pinned to7dc06ec and remains historical, not overwritten. New output `csa_feature_sharing_accelerate_v2`, new notebook `CLIP_PATCH_CSA_FEATURE_SHARING_ACCELERATE_2GPU.ipynb`.

Pin HuggingFace Accelerate1.15.0 wheelSHA97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec from PyPI; user notebook downloads/verifies wheel and installs --no-deps into isolated/tmp target (never reinstallsTorch/HFhub). Existing dependencies must import successfully BEFORE expensive preparation. Source metadata and actual runtime version/torch/CUDA/device/DDP/precision recorded.

Accelerator initializes the process group/device and prepares models/Adam, handles backward and clip_grad_norm_. Use unwrap_model instead of .module. Explicit gradient_accumulation_steps1 preserves token-normalization factor2/globaltokens, including16/11 tail. Reference full-global model uses ordinaryTorch backward, which is the numerical reference rather than the distributed trainer. Do not pass already manually rank-sliced NumPy batches to an Accelerate-prepared/sharded dataloader; no second sharding/padding/repetition. Torchrun still only launches2processes; Accelerator owns distributed model preparation inside them. PyTorch collectives remain for exact token counts, stats and error broadcast.

Call accelerator.free_memory BETWEEN fresh-model passes: Accelerator otherwise retains model/optimizer references and could accumulate GPU memory. End_training plus guarded process-group cleanup at exit. Do not prepare a model twice. Package/AST/mock tests are author-safe; actual Accelerate/NCCL/gradient parity/memory/speed remain user-run. Official references checked: https://huggingface.co/docs/accelerate/en/package_reference/accelerator and https://huggingface.co/docs/accelerate/en/package_reference/kwargs and https://pypi.org/pypi/accelerate/1.15.0/json .


## Amendment — V2 stop / V3 numerical-reference correction,10/11/2026

V2 source7c70747 installs/initializes Accelerate1.15.0 correctly, bothT4/FP32/accum1. Stops BEFORE any optimizer update/timing on grouped32 ordinary(shared=false), not on sharing. Loss delta9.587e-7; cumulative maxgradient relativeL2 .00091958, maxabs7.6472e-5 exceeds fixed5e-5. The log does not retain offending tensor's scale or raw gradients, so numerical rounding/batch-shape sensitivity is plausible, not proven; no speed or quality conclusion.

V3 keeps model/FP32/Accelerate/samplers/batch/optimizer/timing/caps unchanged. Reference scalarNLL remains full-global ordinary. Reference GRADIENTS now accumulate ordinary rank-shaped microbatches16/16(or16/11) on oneGPU with the SAME globaltoken denominator. This is mathematically the full-global XE gradient but reduces FP32 kernel/ReLU differences induced by full32 vslocal16 matrix shapes. Log micro/full NLL consistency; do not change DDP loss scaling2/globaltokens.

Replace dimension-dependent absolute-only maxgradient rejection by a scale-aware diagnostic for EVERY parameter: finite+connected, relativeL2≤.005 (stricter than prior.01), maxabs≤3e-7+.005*maxabs(reference). Near-zero reference norm≤1e-7 retains maxabs≤3e-7. Record referenceL2/maxabs, errorL2/maxabs, normalized errors, maxerror coordinates and actual/reference values for every tensor, retaining diagnostics even on failure. A half/doubled gradient or localized missing-caption error must fail pure numerical fixtures. These are engineering FP32 acceptance tolerances, not measured proof of correctness. Actual revised GPU parity is pending; V2 is NOT retroactively labeled PASS.

New notebook/output `CLIP_PATCH_CSA_FEATURE_SHARING_ACCELERATE_V3_2GPU.ipynb` /`csa_feature_sharing_accelerate_v3`; oldV2 notebook remains immutable on7c70747. User-run V3 still gates all timing behind6numericalcases. No localTorch/Accelerate/model/scorer execution; no auto retry or full training. Fresh preparation is the simple supported route because no live/tmp retention is known; do not claim failed output contains staged NPYs.
