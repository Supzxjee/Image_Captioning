# Noun evidence G/S pilot

**G/S result (2026-10-23): both completed and audited; both performance gates FAIL. Keep A.**
See [GS_RESULT.md](GS_RESULT.md). Matched S−G CIDEr CI includes zero; no bbox-benefit claim.

G is a global-pooling supervision control. S uses fractional bbox pooling in the region
loss. Both freeze A and train the same 108,752-parameter patch/class head for 2,500
updates on two T4 GPUs. At inference both pool globally and apply the same bounded,
compound-guarded noun-logit correction. Neither requires boxes at inference.

Use **NOUN_EVIDENCE_S_2GPU.ipynb** in Đức Anh's notebook and
**NOUN_EVIDENCE_G_2GPU.ipynb** in Ngân's notebook. Internet ON; T4×2; Save & Run All.
The requested S destination is
https://www.kaggle.com/code/ducanh2403/2-vi-cai-kia-qua-nhieu/edit .

Attach these existing inputs, with access granted to each running account:

| Input | Default path |
|---|---|
| Exact trained A | `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin` |
| Legacy prompt | `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt` |
| Visual cache | `/kaggle/input/datasets/ducanh2403/visual-cache` |
| Karpathy JSON | `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json` |
| Object reward detections | `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json` |

No extra dataset is required. The pinned repository contains a 960,954-byte ZIP with
five flat label JSON files; SHA-256 checks precede extraction and training. The same
ZIP can be uploaded as a Kaggle dataset if desired. It contains derived TRAIN labels,
not images, model weights or validation supervision.

The notebook stages 15,000 FP32 visual/legacy rows in `/tmp` using persistent source
HDF5 handles. Preflight checks zero-shift A parity, compound guards, five-reference
expansion, rollout/teacher scores, uneven/zero DDP region means, gradients and frozen
A. A failure stops training and preserves logs. Do not infer success from an archive
or KEEP message alone.

Outputs: `noun_evidence_S_review.zip` / `noun_evidence_G_review.zip`, plus **head.bin**.
The head file needs the separate exact A checkpoint; it is not a standalone captioner.
`run_summary.json` and both COMPLETE worker receipts establish completion.

Matched raw greedy A/endpoint evaluation uses reused val5000, with old dev1000
secondary. Compare G/S source, label receipt, initial head, frozen base, epoch order
and runtime before metrics. All numerical gates remain in [DESIGN.md](DESIGN.md).
No novelty or improvement is claimed from the negative G/S results.

Local verification is model-free: `python experiments/noun_evidence_proposal/test_metadata.py`.
The notebook contains GPU preflight; no model was run locally.
