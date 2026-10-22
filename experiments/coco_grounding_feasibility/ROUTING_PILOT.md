# COCO noun–patch routing B/R

Status: notebooks prepared for the user; no local model inference or training.
Interface smoke previously passed; this does **not** establish caption improvement.
Astra approved the following fixed pilot on 2026-10-22.

## Before running

Upload `coco_noun_box_train10k_v11_dataset.zip` as a private Kaggle dataset,
suggested slug `coco-noun-box-train10k-v11`. Its seven flat JSON files include
`manifest_receipt.json` (SHA256 `bde873ba77124333488b263a052419dc80e412c1e089178cdf423d2836e1b146`).
This is a 2.67 MB label package, not the full COCO annotation archive.
It contains 17,904 retained noun spans on 4,495 of the locked 10,000 training images.
Person is 4,476 spans (25%); this does not imply 25% of loss influence or gradient.
The remaining images still contribute normal caption CE/SCST.

Use `COCO_ROUTING_B_2GPU.ipynb` and `COCO_ROUTING_R_2GPU.ipynb`.
One notebook uses **both T4 GPUs**, and the two notebooks may run on two accounts.
Internet ON; Save & Run All. Attach six inputs and edit the first cell paths if needed:

| Dataset | File/folder |
|---|---|
| qformer-scst-object-w05-legacy | qformer_scst_object_w05_legacy.pth |
| prompt-cache | prompt_clip_tokens_cache.pt |
| visual-cache | visual cache folder |
| mscoco-2014 | dataset_coco.json |
| objectdetectionecache | objectdetectioncache.json |
| coco-noun-box-train10k-v11 | seven flat JSON files |

No UPT/VG feature cache, original images or full annotation ZIP is needed.
Private inputs must be accessible to the second account.
Staging reads 11,000 cached feature rows on CPU before training; it prints progress.
HDF5 staging is kept in `/tmp`, not in the checkpoint dataset.

## Locked experiment

Both start from original checkpoint `5f8eead3…`, not historical A-end.
Same A train10k/dev1000, five references per image, seed42, global batch8,
1250 updates, FP32, eval mode with gradients, strict math SDP.
AdamW LR5e-6, weight decay .01, global gradient clip1.
Steps1–200 use caption CE; later add .05 SCST using
CIDEr-D + .5 object_count_v2 with full TRAIN IDF.

At every update, add `.05 * L_aux`:

* **B**: negative log raw decoder→Q→all-spatial-patch mass at supervised noun positions.
* **R**: negative log raw decoder→Q→bbox-patch mass at the same positions.

Last-layer head means, CLS/legacy mass excluded without renormalizing the slices;
bbox masks use fractional patch overlap after the pinned 224×224 square resize.
Mean token loss inside a noun span, then mean over globally supervised spans.
For local sum s and total span count N, each DDP rank returns `2*s/max(N,1)`;
zero-label ranks participate, and both-zero batches produce zero auxiliary loss.
No parameters are added; attention observers are absent from inference.
This trains a weak noun/class patch-routing signal, not verified relation grounding.

Preflight is inside each notebook, with zero optimizer updates. It checks
uneven/zero DDP counts, multiword means, five-caption expansion, causal positions,
CE parity, finite nonzero auxiliary gradients, unchanged weights and parameter set.
Do not skip FAIL checks or loosen tolerances. User runtime checks remain unexecuted
until the notebooks run on Kaggle.

Preflight correction: compare actual five-reference decoder input expansion exactly,
not logits computed at two different batch sizes (FP32 GEMM can round differently).
Same-batch observer logits/CE parity, original tolerance and gradient checks remain.
For an already prepared session that failed this old check, `resume_preflight.py`
updates only the preflight source/protocol hash, preserves a protocol backup,
and reuses staging. It refuses recovery if training history/checkpoint exists.

Training records actual image order, local/global span counts, person counts,
Q/spatial/CLS/bbox mass, main and auxiliary loss, and total preclip gradient norm.
Historical A used the same source sampler/seed but did not byte-log its actual order.

## Evaluation and interpretation

Only original and endpoint raw greedy on the same dev1000. No checkpoint search,
reranking, test use or hyperparameter sweep. Preserve both review ZIPs and `.bin`
checkpoints. Review ZIP excludes weights/staging.

R must exceed A by CIDEr≥.01 with paired95% CI lower>0, exceed original, have
CHAIRs/i≤A, and Recall≥A−.01. Only then do a small blind factuality review.
Attributing a benefit to bbox position additionally requires R−B paired95% lower>0.
If R≈B and both beat A, only a generic image-routing benefit is supported.
Improved auxiliary loss/attention mass alone does not establish better captions.

The development set has already been used; one seed cannot establish stability
or an independent test gain. COCO class annotations are incomplete weak labels;
unique annotation count does not guarantee a unique physical object.
