# Full native-detail H/L cache — user-run notebook

Prepare features for the already proposed shared-adapter pilot, **not a new experiment or training run**. Full means locked TRAIN10k + reused val5000 (15,000 images), not all COCO. No test images or reference captions select the views. The original A path and its existing cache are preserved.

## Run

Use `NATIVE_DETAIL_CACHE_15K_2GPU.ipynb` with **T4 ×2**, Internet ON, **Save & Run All**. Attach the same three inputs as the passed resource benchmark:

- `vuthetam/mscoco-2014`: original JPEGs and `dataset_coco.json`.
- `ducanh2403/pair-source-fullframe-ce`: exact A at `upt_hidden_caption_A/checkpoint.bin` (SHA256 `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`).
- `ducanh2403/visual-cache`: old global features for numerical compatibility preflight.

First-cell paths may be edited to the actual Kaggle mount. No API token is needed. Root has not run this notebook; model-free local tests cover metadata/protocol/package behavior, while the notebook runs four synthetic HDF5 tests before model extraction and then GPU compatibility checks.

Expected tensor payload: **6,021,120,000 bytes**, plus HDF5/metadata. The previous256-image benchmark projected38.8 minutes for extraction and roughly55 minutes with similar preparation. This full job additionally verifies hashes and feature readback, so that projection is not a guarantee. It excludes adapter training and caption evaluation.

## Output and preservation

`/kaggle/working/native_detail_HL_15k` contains:

- `cache_spec.json` and `manifest.json`: immutable extraction contract and exact image order/group mapping.
- `train_000.h5`…`train_019.h5`, `val_000.h5`…`val_009.h5`: 30 shards,500 images each. Each has H/L float16 arrays `[500,196,512]`, `coco_ids`, Karpathy `image_ids`, normalized `xy[196,2]`, frozen model/runtime/spec identities and completion marker.
- A matching JSON receipt for every committed shard, including complete-file SHA256.
- `cache_receipt.json`: emitted only after both workers complete and all30 shard hashes/identities/counts pass. Expected `status=CACHE_COMPLETE`, `images=15000`, `benchmark_only=false`.
- Per-attempt protocol/runtime/preflight/progress/worker logs; failures are not labeled success.

**Keep the entire folder as a Kaggle dataset**. The neighboring `native_detail_HL_15k_review.zip` contains only JSON/log/source and is for review; it is not the feature cache. No dataset is published/uploaded automatically.

## Resume contract

Each500-image shard is the resume unit. A shard is written to an owned `.partial.h5`, checked for shape/IDs/xy, dtype, finite values, FP16 conversion and last-batch readback, then atomically renamed and given a receipt. Checkpoint/CLIP/projection/source/geometry/group identities are fixed. Whole feature files receive hashes; frozen weights are checked before/after each worker.

- In the same surviving session, rerun with the same `OUTPUT`. Existing completed shards are verified and skipped.
- To continue in a new session, first preserve the old folder as a dataset. Attach it and set `RESUME_FROM` to the directory directly containing `cache_spec.json`. Verified committed shards are copied to the working output; the input dataset is unchanged. An unreceipted source shard is skipped and regenerated. Copying takes disk space and time.
- This is **completed-shard resume**, not mid-batch resume: at most500 uncommitted images per GPU are redone after interruption. A valid local final HDF5 with no receipt can have its receipt reconstructed after full verification; a `.partial.h5` is redone.
- Resume rejects changed source/spec, payload corruption, wrong IDs, wrong model identities and changed compute-runtime identities. It does not silently mix encodings across library releases.
- Resume cannot recover files that Kaggle has already deleted. Save & Run All and preservation of the output remain necessary. Do not replace a preserved dataset version with an incomplete rerun.
- Run only one extraction process against a given output folder at a time.

## Encoding is unchanged from the passed benchmark

Four fixed normalized quadrants (TL/TR/BL/BR), fractional boundaries for odd image sizes. H uses native RGB source; L first resizes the whole image224×224. Each quadrant becomes224×224 bilinear. CLIP ViT-B/16 revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`, eager FP32, no TF32/autocast, original A normalization. Drop CLS, average-pool14×14 to7×7 **before** A's frozen projection768→512 and ReLU. Four tiles concatenate196 tokens, with coordinates in the full image frame.

Two independent workers alternate shards: each processes7,500 images, batches4images/32views. Setup reuses the approved benchmark's input hashes, original-image availability checks, pinned CLIP prefetch and eight numerical global-cache comparisons (`atol=.01,rtol=.005`). This is numerical compatibility, not bitwise identity or a claim about the old unrecorded model revision.

No adapter/decoder forward, CE, SCST, reward, optimizer or caption evaluation is added. Resource feasibility does not establish caption gain or full training-budget feasibility. After the cache review, prepare the shared-adapter pilot and measure its staged cost separately.

## Local checks

`python experiments/native_detail_cache/make_notebook.py` builds the portable unexecuted notebook. `python experiments/native_detail_cache/test_full_cache.py` performs six metadata/packaging checks locally; four HDF5 checks are skipped if h5py is absent and execute in Kaggle before extraction. No local model execution is required or performed.

## Save the audited completed cache as a new dataset

Use `SAVE_NATIVE_DETAIL_HL_15K_DATASET.ipynb`: CPU/no accelerator, Internet ON. No Add Input is needed. Download the exact completed source notebook Output **`ducanh2403/2-vi-cai-kia-qua-nhieu/89`** using the API token. The installed official pinned CLI must first pass a mocked request-routing check with `version_label=v89`, before any real download. The downloader filters to63 required payload files in a fresh temporary folder and requires the exact audited receipt fingerprint; no latest/other-version fallback is allowed.

Enable a Kaggle Secret containing the API token; default label is `KAGGLE_API_TOKEN`, configurable through `SECRET_NAME`. Set `OWNER` to the token account and choose the new dataset `SLUG`. The token stays in memory/the authenticated child process environment and is redacted from subprocess output; it is not written to the staged dataset.

The notebook verifies all30 actual HDF5 hashes and33 required JSON files against `upload_inventory.json`, then copies them byte-identically into a flat64-file dataset folder (including its transport receipt). Original historical logs/source stay in the original notebook Output and GitHub audit. Roughly12.6GB free temporary/working disk is required for downloading and staging the verified copy. The transport receipt records the source kernel version89. This is a transport-only notebook: no GPU, model, cache regeneration or training.

The API creates a **new Private** dataset, preserving raw HDF5 and disabling CSV conversion. Existing datasets are not updated. The existing validated Kaggle uploader script/pinned official CLI revision is reused. A returned creation acceptance is distinguished from dataset processing readiness; check the dataset page if status is pending or unavailable, rather than rerunning create.

Default target: `ducanh2403/native-detail-hl-15k`. The cache files are directly at the dataset root. After attaching that dataset, send the actual input path containing `cache_spec.json`; do not append the old working folder name. Nine model-free transport/notebook tests cover byte-identical repeat-safe copy, missing/corrupt files, unsafe staging/inventory, receipt fingerprint, notebook syntax/source, token redaction, version89 routing/fallback rejection, payload filtering and JSON provenance. Live Kaggle download/upload is left to the user.
