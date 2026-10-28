# Next step: label review and exact cache copy

28/10/2026. No training notebook or GPU job is launched by these artifacts.

## 1. Review the32-record assistant draft

Open `review/assistant_review.html`. Each record shows its source statement/label, native image and predicted S/O boxes. Check all records and especially #16/#22 (ambiguous spatial wording) and #32 (possible source disagreement). Enter your name, explicitly confirm each record and export `vsr_frame_audit32_review.json`.

The Codex draft is **not a human label**. The frozen manifest is unchanged. An exported review is not automatically a training approval: uncertainty/disagreements must first be assessed for a systematic mapping or ontology problem. Do not silently drop/relabel individual samples to pass the gate.

## 2. Run CPU preparation in parallel with review

Import `PREPARE_RELATION_FRAMES_V2_CPU.ipynb` into Kaggle. Set **Accelerator=None**, **Internet=ON**, then **Save & Run All**. It checks/copies existing caches only, so it may run while human review remains pending. It does not train a model or approve the label gate.

Add these three existing datasets and adjust only their actual mount paths if necessary:

| Dataset | Default configured path |
|---|---|
| Original visual cache | `/kaggle/input/datasets/ducanh2403/visual-cache` |
| Original prompt embeddings | `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt` |
| Existing predicted boxes | `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json` |

No Acheckpoint, RGBdataset, API token or new detector/CLIPmodel is needed for this step. Original prompt `.pt` is verified against its pinned hash before CPU deserialization; it can require several GiB of RAM. Do not replace it with the UPT prompt JSON or the VSR statements.

The notebook fetches four small pinned assets from core commit `dd49cbb`, checks their exact Git-blob byte hashes, then invokes the copy script in an isolated source folder with CUDA hidden. Notebook size is about5.7KB, below Kaggle's1MBsource limit. It contains no embedded model or feature tensors. Static code cells compile; no Kaggle execution or actual prompt-cache load was performed locally.

The script verifies that all612images have original visual/prompt embeddings and the expected predicted-box bank, writes source-compatible arrays, then rereads each saved row for exact equality. It does not instantiate an encoder/detector/captioner or reconstruct prompt text. Missing prompts fail closed and remain a prerequisite, not a reason to substitute labels.

## Outputs

- Keep `/kaggle/working/relation_frames_v2_cache` in full. `inputs.h5` stores copied feature/prompt tensors; the directory also has the manifest, image metadata, predicted boxes and a completion receipt.
- Send `/kaggle/working/relation_frames_v2_cache_review.zip`. This is the small metadata/log archive and deliberately excludes `inputs.h5`.
- `relation_frames_manifest.bin` contains the original gzipped metadata bytes; the extension avoids automatic Kaggle decompression. It is not a model checkpoint.

The expected receipt is `CACHE_COPY_COMPLETE_PENDING_LABEL_AUDIT`, with `training_approved=false`. If it fails, send the reviewZIP/log and preserve the partial output; the notebook does not retry training or use another source. Use a fresh output/runtime rather than overwriting an existing result.

## Validation and limits

Three CPUarray/identity tests passed; both blank/draft review pages passed syntheticDOM checks for32records, explicit confirmation, reset-after-edit and unapproved training export. Manifest/draft/source binding and notebook static compilation/hash checks passed. Original32images and contact sheets were visually inspected; native browser layout is not independently verified by the syntheticDOM tests. HDF5header/ID coverage is612/612, but actual copied tensors and original prompt coverage remain **user-run Kaggle checks**.

After the human review and CPUreceipt are assessed, proceed to frame pretraining/preflight and the three fixed caption arms. No claim of novelty, grounding improvement or CIDErgain follows from preparation success.
