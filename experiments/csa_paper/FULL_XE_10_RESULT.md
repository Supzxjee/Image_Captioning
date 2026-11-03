# Full CSA XE10 V2 — user-run complete

Reviewed03/11/2026. ZIP SHA256 `7ff1ba033f560b2a0565cb30d7bae137eba8fca9a59db45e9e947b7cb9db286f`, pinned core `e655a565d8be354abe579b8a2bb59e7d1733803b`. [Metadata audit](audit_full_xe10.py), [recomputed summaries](FULL_XE_10_RESULT.json), small runtime receipts in `full_xe10_review_text/`. Large per-image review cache remains local/ignored. No model/Torch/checkpoint/features or official scorer executed by agent.

## Completion and selection

User run completed all10epochs:113287TRAIN images,566747caption events per epoch,17711updates/epoch;177110updates,5667470caption exposures and64979000target tokens total. Ordinary2T4 FP32/global32captions, no sharing/RL/test. Full5000val/25010references scored after every epoch with original per-image beam5/max20/raw logscore.

| Epoch | CIDEr valid ↑ | NLL TRAIN ↓ | NLL valid ↓ | Mean caption words |
|---|---:|---:|---:|---:|
|1|0.275163|4.084960|3.339067|2.7004|
|2|0.353263|3.201070|3.062390|3.6730|
|3|0.504482|2.912814|2.926397|5.0114|
|4|0.746480|2.750411|2.757025|6.6074|
|5|0.882422|2.638444|2.635113|7.4546|
|6|0.960949|2.564916|2.523538|7.8154|
|7|0.963904|2.508349|2.492312|7.9300|
|8|0.976159|2.469921|2.463262|8.0436|
|**9**|**1.002933**|2.441058|2.446409|8.2630|
|10|0.997179|2.417765|2.435452|8.0850|

CIDEr uses natural evaluator scale (×100 best=100.2933). TRAIN NLL is observed during parameter updates; valid NLL is computed at an epoch endpoint, so their small ordering reversals do not imply data leakage. Both NLLs decrease throughout; CIDEr drops0.005754 at10 after its best at9. This is not enough to prove overfitting or convergence. Keep the prelocked best-CIDEr rule; do not select epoch10 merely because it is last.

## Checkpoints to preserve

- **Best quality:** `/kaggle/working/csa_full_xe_10_v2/best.bin`, validated epoch9/update159399; reported SHA `f09e11c374477be93cc142a238dbd0e206060f33fa4fab5988e767ea7d259fe6`.
- **Latest continuation:** `/kaggle/working/csa_full_xe_10_v2/checkpoint.bin`, trained through epoch10/update177110; reported SHA `272f843a21b3a63c61323f748ed916fa3ae0cf2777549061439c07b410006950`.
- `epoch=10` in best receipt means next training cursor after epoch9. `epoch=11` in latest receipt means next cursor after epoch10, **not11 trained epochs**. Cursor0/phase=train is consistent with completed endpoint10.
- Review ZIP excludes both binary checkpoints. Hashes are worker reports, not independent local binary verification. Preserve the entire output folder, including receipts; exact notebook version requested for Dataset transport. Do not try to recover weights from this small ZIP.

**User confirmed source version139.** [SAVE_CSA_XE10_V139_DATASET.ipynb](SAVE_CSA_XE10_V139_DATASET.ipynb) is prepared: CPU/Internet/API Secret, no AddInput/model run. Downloads exact139, verifies8payload SHAs (both checkpoints,5resume/result JSONs,original complete review ZIP), copies bytes unchanged, creates NEW Private `ducanh2403/csa-full-xe10-v139`. The ZIP preserves the remaining caption/reference/log files; flat checkpoint/receipt files support continuation. Source weights are not deserialized.3local fixture/static tests and5original ZIP JSON-byte hash comparisons passed; API creation/actual weight transport remains user-run pending. Keep Output139 and ROI134/banks130 until saved. This does not extend training.

## Caption and runtime checks

All10×5000predictions have exact val ID coverage, no duplicate/rank overlap, proper token-to-text correspondence and EOS/PAD/max-length policy. At best9:0empty captions,409captions≤2words (8.18%),2915unique caption strings;4925EOS-terminated/5000 (98.5%),75reached20tokens. These are diagnostics, not correctness or hallucination labels. Unlike the tiny256-image pilot, generation is no longer dominated by two-word captions; short generic outputs still exist.

Training worker21726.78s=**6.035h**. Validation total2446.40s=40.77min (3.82–4.62min/epoch). Epoch phase time minus validation≈5.346h includes training/checkpoint/associated overhead, not a pure kernel benchmark. Input staging84.68s and parent pip/preflight/archiving are outside worker time. No end-to-end account billing or full-run peak-VRAM claim. Failed old batched decoder is not reported passing; V2 original beam preflight and six DDP checks passed.

## What this establishes / next decision

This establishes that the self-written CSA adaptation can complete fullTRAIN XE and learn substantially beyond the tiny pilot. It does **not** establish improvement over the prior Q-Former system, an exact CSA reproduction, a new alignment contribution, reduced hallucination, or independent test performance. Do not compare CIDEr1.0029 on5000val directly to old4000/dev1000 numbers. Even historical val5000 baseline comparisons differ in training/decoding and need a matched evaluation for strong claims.

**CHAIR has not been computed.** First preserve best/latest, then evaluate object hallucination on the already-generated best9 captions using the proper annotations/scorer; that needs no new GPU caption generation. After that discuss extending XE to15 or matched SCST/branch experiments; none is auto-approved by this result. Image-conditioned-bank stream remains an unapproved candidate, not a demonstrated fix. Do not interrupt baseline assessment with a new full pipeline claim.

Audit127grouped metadata checks pass:17Git blobs, cache/input/reference identities, exact val IDs against five pinned fullROI shard digests, six DDP guards, all per-epoch records/caption summaries, checkpoint-selection arithmetic, all1771progress logs and actual3epoch warmup. These checks verify artifact consistency; official CIDEr was not recomputed locally, visual semantics were not independently judged, payload arrays/weights were not opened.
