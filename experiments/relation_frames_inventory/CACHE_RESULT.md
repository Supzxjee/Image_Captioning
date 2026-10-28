# Relation-frame v2 cache review — accepted metadata

28/10/2026. **DATA_AND_CACHE_METADATA_ACCEPTED_MODEL_PREFLIGHT_PENDING**. Source notebook version105 is user-reported; no source notebook or API was executed by Codex.

- CPUpreparation returned0, completed612/612images and original-prompt entries in **371.47s (~6m11s)**.
- Source commit `dd49cbb9bf281b69b733bb06a818cffbbc24164e` and preparation-file SHA match the pinned notebook package. No CLIP/detector/captioner was instantiated according to that source and its log.
- The binary V2manifest matches the frozen bytes exactly. The images/box metadata payload hashes also match the receipt.
- Independently checked all612image ID/name/dimension bindings and **1,280 selected entity-to-predicted-box bindings** (320pairs×2images×2roles).
- Both original prompt/detector cache hashes match the baseline sources. No VSR or GT-name prompt substitution occurred in the pinned copy procedure.
- Human audit was completed separately:32/32confirmed,16supported/16contradicted. The preparation receipt's pending-label flag is an earlier snapshot, not a demand to repeat the human review. Astra accepted the exploratory audit gate.

The small reviewZIP deliberately omits `inputs.h5`. Its declared SHA is `daf32d52859452eb2beb8c35e2aad6ed6418cf52c68e4ba646be633151a7a1e9`; all-row readback is reported by the pinned preparation worker, **not independently repeated locally**. The saver and later model preflight must verify the full file. No caption/grounding improvement is measured here.

## Preserve exact Output105

Use `SAVE_RELATION_FRAMES_V105_DATASET.ipynb` on Kaggle with **CPU / Accelerator=None / Internet ON**, no Add Input. Enable the configured Kaggle Secret `KAGGLE_API_TOKEN` (or set SECRET_NAME to its actual name, e.g. `codex`). OWNER must be the account allowed to access the source notebook and create the dataset.

The notebook reuses the established pinned official Kaggle CLI and mocked version-routing preflight. It downloads only the five cache files from **`ducanh2403/2-vi-cai-kia-qua-nhieu/105`**, finds the exact receipt hash, checks every payload including fullHDF5, and copies unchanged bytes to a clean staging folder. It never falls back to latest or silently overwrites another dataset. It creates a new **Private** dataset with an explicit valid `other` license declaration and preserved upstream rights. API acceptance does not claim processing has finished.

Default slug: **`ducanh2403/relation-frames-v2-cache-v105`**. Expected attached path:

`/kaggle/input/datasets/ducanh2403/relation-frames-v2-cache-v105`

Check the actual mount path after processing. Payload files should be at the dataset root: `inputs.h5`, `relation_frames_manifest.bin`, `images.json`, `detector_boxes.json`, `receipt.json`, plus `transport_receipt.json`. Five files avoid the previous source-folder/500-file issue. The `.bin` manifest remains byte-identical and does not need manual decompression.

The saver is about13KB and its code cells compile. Synthetic transport tests cover exact copy/resume, corrupt payload rejection, no overwriting changed files, safe paths and pinned105/private/unexecuted notebook settings. Live Kaggle authentication/download/upload is **not run locally**; no credentials are requested in chat. Dataset path is not yet confirmed available.

Next experiment is the locked positive-only versus relation-counterevidence frame pretraining, followed by caption controls only if its gate passes. Preserve the dataset first so this input is reusable across the two accounts. Model/gradient/DDP preflight and any training remain unexecuted.
