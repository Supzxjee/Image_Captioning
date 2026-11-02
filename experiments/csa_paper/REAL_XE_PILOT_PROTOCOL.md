# CSA real-data XE learning pilot

02/11/2026. User confirmed both prepared Dataset mounts and reports19GPUhours. This is the bounded pilot following successful real-input smoke, not a full COCO run or a new method. User runs all model work.

## Fixed inputs and scope

- ROI/vocab/word metadata: `/kaggle/input/datasets/ducanh2403/csa-real-v126-lemma-fixed`, same original8+corrected2 SHA/input gate as real-bank smoke.
- Frozen semantic vectors: `/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130`; all6payload fullSHA from reviewed Output130 and bank order/norms checked before use. No text encoding/CLIP/model downloads. Require raw NPZ bytes, fail on expanded/missing file.
- Exactly256TRAIN images ×5references=1280captionevents/epoch;64val ×5=320valevents; no test. Training bank derived from fullTRAIN remains declared. Training targets remain original caption sequences, not generated prompts.
- Fresh CSA paper-profile modules, rawROI768→512, frozen noun5269/verb1594banks, K150, heads8, FFN2048, dropout.2,3decoderlayers. No Q-Former/C0/author code/new objectives.

## Schedule and budget

15epochs of this small subset;40optimizerupdates/epoch=600total;19200captionevents. OneGPU0 FP32, microbatch4, accumulation8 gives effectivebatch32. This differs from proposed full-system2GPU AMP scheduling but preserves token-mean XE/globalbatch32. Adam(.9,.99),eps1e-8,weightdecay0; globalclipnorm1. Warmup starts8e-6 at first update, linearly reaches1e-4 at update120, then constant. Epoch shuffling deterministic NumPy(seed42+epoch), no dropped/duplicated events. XE includes EOS/excludes PAD, token counts computed over entire effectivebatch before microbatch backward; never average microbatch means.

Hard worker cap5400seconds, internal stop at5100seconds with periodic atomic checkpoint after each completed epoch. Caps are not ETAs. Record first10updates timing and estimated remaining training (excludes evaluation/I/O) for information; no fullCOCO extrapolation. OneGPU only; user can enable a singleGPU. Require3GiB free for checkpoint temp/optimizer state. If timeout/failure, save partial progress and archive logs; do not report a partial run as15epochscomplete. This implementation does not auto-resume or auto-tune LR/epochs. A saved endpoint/last-epoch checkpoint permits separately reviewed continuation.

## Learning evidence

Before update0: exact train-probe first32TRAINimages/all5refs, all64val/all5refs teacher-forced token-mean NLL; beam5/max20 on all64val. Each epoch evaluate all320valcaptionevents, trainprobe at epochs1,3,5,10,15. Decode all64val again at epoch15 only. Validation refs are used only for teacher-forced NLL, never inference prefixes. Inference uses features/banks/BOS/vocab. Store captions+allreference strings in review for later assessment, clearly labeled development subset.

Final endpoint checkpoint only (no best-checkpoint selection or score fishing). Log actualupdates/images/captions/validtokens/gradient norms/LR/time/memory. Save runtime device/software and frozen-bank equality. Check finite forward/loss/gradients beforestep, all model parameter gradients present onfirstupdate; representative projection/aggregation/gates nonzero. Preserve exactdata/source/policy identities in checkpoint/receipt.

Compare initial vs final NLL and EOS/repeated-token counts, but do not use generic fluentcaption/EOS as proof of visual grounding. Tiny-subset overfitting is possible. No CIDEr/CHAIR reward or claims, no comparison with A or paper benchmark, no RL, no test, no semantic-grounding accuracy claim. Useful learning pilot precedes larger preparedROI coverage and officialmetric pipeline.

## Agent checks

Only metadata/NumPy/static fixtures locally: exact1280/320event coverage, deterministicshuffle, referenceindex collation/EOS, token-weighted accumulation reference example,600update/120warmupschedule, cacheSHA manifests, notebookemptyoutputs/AST/sourcepins. Never importTorch/runtime/model locally. User-run CUDA behavior remains unverified until review ZIP.
