# H/L native-detail benchmark: resource check passed

User-supplied `native_detail_benchmark_256_review.zip` SHA256: `d514be28c3ab30d2c4811c2737e8ee26700cdcd8fb557af197721fb7ec01ef39`.

Both T4 workers completed128 measured images,32 batches each. The locked sample contains256 distinct TRAIN images plus16 separate warmup images; no val image was selected. Runtime source hashes match the approved notebook overlays, A/COCO/group identities match their pins, both runtime manifests and frozen model digests agree, and reported timings recompute from the workers' release/start/end clocks. Review audit: **PASS**.

| Measurement | Result |
|---|---:|
| Measured images / views |256 /2048|
| Concurrent extraction wall time |39.736 seconds|
| Preparation before workers |920.999 seconds ≈15.35 minutes|
| Parent run total, including setup |999.450 seconds ≈16.66 minutes|
| Peak GPU allocation per worker |716,280,832 bytes ≈0.716 GB|
| Peak GPU reservation per worker |981,467,136 bytes ≈0.981 GB|
| Both benchmark HDF5 files |102,782,016 bytes ≈103 MB|
| Free output disk at finish |20,837,474,304 bytes ≈20.84 GB|

At this throughput, **15,000 images /120,000 views project to38.8 minutes for the extraction loop**. Adding the observed preparation cost gives54.2 minutes, before other startup and postchecks. Treat roughly55 minutes as a planning estimate for similar cache-only conditions, not a guaranteed ETA or a full-pilot duration.

Full H/L FP16 tensor payload is **6,021,120,000 bytes total**, around3.01 GB per condition. The observed free disk is enough for that payload in this run. It does not establish enough space for all other stages, model/optimizer states and archives together. Do not count free space as a training-budget check.

About72–73% of measured per-worker phase time was JPEG decoding/CPU preprocessing. Transfer+CLIP+pooling/projection+copy took about10.7–11.0 seconds per worker, versus28.6 seconds CPU preprocessing. The current bottleneck is the measured CPU input path, not GPU memory. There is no need to introduce an optimization experiment before this cache: future parallel loaders would change the throughput estimate and should be measured rather than assumed.

Eight fresh global224 encodings passed the original mixed tolerance (`atol=.01, rtol=.005`) against the old FP16 cache. The largest absolute difference was0.0303955; this is not a contradiction, since allowed error is `.01 + .005*abs(reference)` elementwise. This is **numerical compatibility on eight examples**, not bitwise equality or proof of the historical unrecorded CLIP revision.

Recommendation: resource feasibility supports preparing a **resumable full H/L cache** for the already locked train10k+val5000 groups, then preserving it as a dataset. Keep the original A path. Measure staged training/evaluation cost before committing the remaining GPU quota to the shared-adapter5000-update pilot. There is no reason to repeat this256-image resource benchmark unchanged.

No training or caption evaluation ran. This result says nothing yet about CIDEr, hallucination improvement or a novel alignment contribution. The small ZIP omits HDF5 payloads, so finite-value/readback and unchanged-weight claims are verified as consistent runtime receipts, not by independently re-reading the feature bytes locally. No local model execution was performed.

Reproduce the audit with `python audit_result.py` after restoring the ignored `review/loaded.json`; it caches the supplied ZIP text once. `test_audit_result.py` checks the real receipt set and rejects tampered timing/source. The benchmark notebook itself is unchanged.
