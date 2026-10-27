# User-run C/H/T package

This package implements the reviewed v0.3 recipe; it has not been run on a local model. `test_gain_data.py` and NumPy arithmetic checks are the only locally executed checks. Torch, frozen-A parity, numerical-gradient and two-GPU checks run on Kaggle before training.

The two experiment notebooks share ONE preparation dataset. First use **PREPARE mode in the C/H notebook**, with CPU acceleration sufficient: it copies existing CLIP/prompt tensors for exactly2kTRAIN+1kDEV and writes the common initial head and object targets. It does not extract new features or caption/train. Save the entire preparation output as a private Kaggle dataset and attach that dataset to BOTH accounts. Then switch C/H to TRAIN mode and run C/H and T in parallel, each T4x2. C and H restart from the SAME zero-branch initial state, never H from C's trained checkpoint.

This extra preparation pass is needed for matched immutable data across accounts; it is not a third experiment notebook. Do not run two independent preparations or use a small review ZIP as the preparation dataset.

Share the private preparation dataset with Ngân before attaching it in her account. Select the same Kaggle container image and T4x2 hardware; the final comparison rejects mismatched Torch/CUDA/backend/runtime manifests. User reported25GPUhours remaining on EACH account; this is recorded separately from the conservative7.8device-GPUhour cap and is not pooled across accounts.

Required once during preparation:

- A `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`.
- Original prompt embeddings `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt`.
- `/kaggle/input/datasets/ducanh2403/visual-cache`.
- COCO JSON `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json`.
- Audited detector cache `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json` for the EXISTING object-reward neutral policy. This is not the UPT prompt JSON or a new relation reward.

Required during C/H and T: the SAME preparation dataset, checkpointA and COCOJSON; CHAIR cache/software downloaded at a pinned revision. Original prompt/visual/detection inputs are no longer needed once the preparation payload is verified. No original image folder, new UPT extraction, prompt-text JSON, CLIP text encoder or API token is required.

The pure metadata package contains6,177eligible groups on1,999of2,000selectedimages;38groups excluded for multipleinstances. Same-label aliases are retained only for the common referent. Both epochs select3,601distinct groups; other groups remain unused under the fixed one-group-per-image/epoch schedule. These are training coverage counts, not caption results.

Implementation limits explicitly permitted by the protocol: no held-out box-label bank is currently packaged, so held-out/reference and generated-phrase grounding evaluations are marked UNAVAILABLE. Caption metrics and training attribution may be reported, but they do not establish grounding correctness or the proposed paper contribution. A later grounding evaluation would require a separately verified held-out annotation input/policy, not treating missing mappings as negative labels.

The actual throughput,64/128quadrature-gradient tolerance, baseline caption parity and training efficacy remain unverified until user-run preflight. Every failure is archived; a computational stop is INCOMPLETE, not a negative quality result. Do not lower thresholds or reduce the500updates to make a failed preflight pass. The fixed cap is not a runtime prediction.

The small experimental `checkpoint.bin` contains the NEW head and optimizer, not a replacement full A checkpoint. Future inference must load the exact frozenA first and then `head_state_dict`. Preserve both inputs and the run receipts.
