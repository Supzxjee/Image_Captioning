# Frozen semantic bank + real-input CSA smoke — COMPLETE

User review ZIP SHA256 `7b14d35af654b630f6a5addf9943dbe5910074c9f772e7d8ee38026558f15f9e`; exact source `086094e3d6a84cbe10b90ea237e290c7f38fd8d1`. [Structured result](REAL_BANK_SMOKE_RESULT.json), exported text in `real_bank_smoke_review_text/`.

## What completed

- Frozen CLIP text encoding for 5,269 nouns and 1,594 verbs, FP32 normalized 512-dimensional vectors. Corrected bank cec76a02… and original126 input identities match protocol. Model revision57c216…, official checkpoint SHA ec89c7b0… reported verified by pinned worker; loading has no missing/unexpected/mismatched keys.
- Real-input validation and forward/backward on 4 fixed TRAIN images, including a genuine full-frame fallback. Fresh self-written CSA, 2 XE updates, batch2, Adam LR8e-6/betas(.9,.99), clipnorm1. Bank buffers unchanged. Runtime reaches summary only after finite logits/loss/all-gradient checks, nonzero designated projection/aggregation/gate gradients and actual-bank permutation check.
- Step1 loss9.1521139145, pre-clip gradient norm8.6012582779; step2 loss9.1589727402, norm8.2083129883. Different batches: these values do not establish a loss trend.
- Beam5/max20 on val IDs234592/535808. First repeats “space”20times; second repeats “terrier”20times; neither emits EOS. These are genuine generated outputs, not good captions. Two updates of a random-initialized model do not establish useful caption learning or failure of CSA.
- Peak allocated GPU memory1001.1875MiB in the CSA smoke process. This is not CLIP-cache peak, total GPU usage, elapsed time or an estimate for full training. No worker duration was recorded; no runtime extrapolation.

## Audit and limits

27 metadata assertions PASS: five Git blob hashes, exact input package, present cache-manifest SHA, statuses, source/model/bank identities, empty loading warnings, split IDs, finite step values, decoded vocabulary/token consistency and final log markers. No local Torch/model/encoder/scorer execution. The full input NPZ and `semantic_banks.npz` are not included in review, so their binary contents were checked by the user-run worker, not independently reread locally. Reported semantic cache SHA `e77608563fb23622d52d27ada06f49e6e578ef573c9fe41861878c36357fa698`.

Pip emitted gradio/diffusers dependency conflict warnings in a temporary environment using inherited system-site packages; they did not prevent the actual CLIP/cache/CSA execution. No need to modify the successful environment just for those warnings.

This validates real-input execution, not semantic alignment accuracy, CIDEr/CHAIR improvement, or exact original CSA system reproduction. Visual revision/detector provenance remain unknown; current ROI/text encoder choices are declared adaptations. No caption checkpoint or quality metrics were created.

## Next step

Preserve `/kaggle/working/csa_real_bank_cache_smoke`, especially `semantic_banks.npz`, `cache_manifest.json`, and `cache_receipt.json`. Obtain exact output version to create a reusable Dataset without re-encoding. Then prepare a bounded real-data XE learning pilot (initially the existing256TRAIN/64val); schedule and GPU budget must be explicit before a longer run. Do not jump to the full15XE+30RL recipe or treat this smoke as a quality baseline.
