# Native-detail cache benchmark — 256 images / two T4 GPUs

This is a resource check before the proposed H/L pilot, not a caption-quality experiment. No training, SCST, adapter, decoder, reference-based selection or caption evaluation runs here. Run only on Kaggle; the notebook is distributed unexecuted.

## Run

Open `NATIVE_DETAIL_BENCHMARK_256_2GPU.ipynb`, attach:

1. `vuthetam/mscoco-2014` — original images and `dataset_coco.json`.
2. `ducanh2403/pair-source-fullframe-ce` — exact A at `upt_hidden_caption_A/checkpoint.bin`, SHA256 `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.
3. `ducanh2403/visual-cache` — old global CLIP features for compatibility checks.

Select **GPU T4 ×2**, **Internet ON**, **Save & Run All**. No Kaggle token is needed. Paths are in the first code cell. An existing output folder is preserved and causes an explicit stop.

Return `/kaggle/working/native_detail_benchmark_256_review.zip`. The small review contains protocol, timings, runtime manifests, source, hashes and logs, not model weights or image features. A ZIP on failure is diagnostic; only `benchmark_summary.json: status=COMPLETE` means success.

## Fixed extraction

Use the existing locked TRAIN10k/val5000 group archive, SHA256 `ebf880fcc0ae23388057abc84ce796b64ee1f0f6c16f804d1cdbf98e73fb78fd`. Check all 15,000 source paths and ID mappings. Sort TRAIN rows by SHA256 of `native-detail-benchmark-v1:<coco_id>`; first16 warmup, next256 measured. Split alternating rows across GPUs. No caption text or validation outcomes select the sample.

- H: four fixed normalized quadrants of native RGB image →224×224 bilinear.
- L: whole RGB image →224×224 bilinear → same quadrants →224×224 bilinear.
- Fractional crop boxes preserve normalized boundaries on odd-sized images. Tile order: top-left, top-right, bottom-left, bottom-right. No EXIF transpose, matching the original image loader.
- Shared pinned `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`, FP32/eager attention, original normalization, no autocast/TF32.
- Drop CLS; spatial average-pool14×14 to7×7 **before** frozen A projection768→512 and ReLU. Store four views as196×512 FP16 per condition and global normalized xy centers.
- Each GPU: eight warmup images, 128 measured images, four images /32 views per batch. Two independent workers, synchronized measurement release; no DDP/NCCL.

Preflight compares four fresh global224 encodings per worker against the old stored cache at its original FP16 verification tolerance (`atol=.01, rtol=.005`). This is observed numerical compatibility, not proof that an old missing revision field establishes identical historical weights. Both new H/L streams use the same newly pinned weights. Frozen model digests must remain unchanged; output shapes/finite values/FP16 conversion/readback are checked.

## Interpreting the report

The concurrent wall interval includes reading/decode, resizing/normalizing, transfer, CLIP encoding, pooling/projection, CPU copy, FP16 verification, HDF5 writes and flushes. Setup/download/hash checks, warmup and post-readback checks are reported separately. Phase totals are per worker, not additive wall time. Source JPEG bytes/resolutions, peak GPU memory and actual HDF5 bytes are retained.

The full proposal requires 15,000×2×4 =120,000 view encodes. Uncompressed FP16 tensor payload is **6,021,120,000 bytes** total (3,010,560,000 per condition), excluding metadata, original A stage, training states and review outputs. Benchmark HDF5 files hold only256 images (~103 MB total), are marked `benchmark_only`, and cannot serve as a full training cache.

`cache_only_linear_wall_hours` scales the measured two-GPU interval by15000/256. It is **not a guaranteed ETA**: image sizes, cold/warm filesystem cache, runtime contention and later batching may change throughput. It excludes full A staging, model startup, training and caption evaluation. Kaggle GPU-hour accounting may differ from wall hours. Do not infer that the full pilot fits remaining quota merely because cache extraction does.

The benchmark does not test the central research claim. Caption gain and H-vs-L evidence still require the separately approved, matched training/evaluation pilot. Fixed quadrants can cut objects; pooled tokens may discard useful detail.

## Local verification

Run `test_tile_data.py` with PIL/NumPy only, then `make_notebook.py`; neither loads a model. The local checks cover geometry, coordinate order, deterministic sampling, concurrent-wall estimates and source/notebook syntax. GPU compatibility, memory and speed remain unmeasured until the user runs Kaggle.
