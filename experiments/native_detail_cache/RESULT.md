# Full native-detail H/L cache: completed; receipt audit passed

User review ZIP SHA256: `53a6c2ec27fab6eee2a9af002196e6f248bc8aacbe795cff829ce81f55cacff7`.

The user-run Kaggle job completed the locked **TRAIN10k + reused val5000** extraction. Both workers generated7,500 new images each; none was reused. All30 committed shard receipts agree with the final `CACHE_COMPLETE` receipt, exact locked group/image order, source hashes, frozen CLIP/projection tensor identities and common runtime digest. The saved sources match the approved notebook overlays. Receipt audit: **PASS**.

| Item | Observed result |
|---|---:|
| Images |10,000 train +5,000 val =15,000|
| Shards |30 ×500 images, each containing H and L|
| Actual HDF5 bytes, both conditions |6,021,652,800 bytes ≈6.02 GB|
| Parent run wall time |3,161.53 seconds ≈52 min42 sec|
| Preparation |847.99 seconds ≈14 min8 sec|
| Worker0 extraction/check time |2,204.79 seconds ≈36 min45 sec|
| Worker1 extraction/check time |2,274.20 seconds ≈37 min54 sec|
| Peak GPU allocation, each worker |715,478,016 bytes ≈0.715 GB|
| Peak GPU reservation, each worker |964,689,920 bytes ≈0.965 GB|

The prior benchmark projected roughly55 minutes with similar preparation. This observed full run is consistent with that planning estimate; no further extraction-speed tuning is necessary to proceed. Training and caption evaluation have not been timed.

All10 notebook tests passed, including the four HDF5 transaction/resume/corruption tests that were pending locally. Eight global image encodings passed the original mixed FP16 cache tolerance. CLIP and projection digests match the passed resource benchmark and are unchanged according to both workers' completion checks. Real interrupted full-cache resume was not exercised: this was a fresh successful extraction, so do not describe synthetic resume tests as a demonstrated full-job recovery.

Pinned cache identities:

- Spec SHA256: `8feb15976e280eb54f8437a3dbe3a6fa6726c50f6abcc86fa12d542c26c1f88a`.
- Manifest SHA256: `6857c40a90c9aac7f7a4dbd504325e5628c75d4ed6e5f7cb24fc9dcdb9e21c6b`.
- Runtime SHA256: `07daad3787ae7cbc444c4ee200ba2aed721b62eadb14056257047335b60e323a`.
- Exact A remains `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.

Next: **preserve the entire `/kaggle/working/native_detail_HL_15k` folder as a dataset**, including30 HDF5 files, their JSON receipts, spec, manifest and final receipt. The small review ZIP is not that dataset. Obtain the dataset's actual input path directly containing `cache_spec.json`, then prepare the already proposed single shared-adapter H/L pilot. Rehash actual HDF5 inputs in the consuming notebook before training; measure staging and training/evaluation cost before committing to5,000 updates.

No change to A, no adapter training, no CE/SCST optimization and no caption evaluation occurred. This result establishes usable cache provenance at receipt level, not caption gain, a new alignment method or full training-budget feasibility. The review ZIP omits feature payloads: local audit checks receipt/source/count consistency and the recorded runtime assertions; it cannot independently recompute the6 GB of HDF5 hashes or inspect feature values. The consuming notebook must perform those checks on the actual dataset.

Reproduce locally using `audit_result.py` with the ignored cached `review/loaded.json`; the ZIP was loaded once. Three audit regressions verify this result and reject a missing shard or modified saved source. No local model execution was performed.
