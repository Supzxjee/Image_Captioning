# Expanded TRAIN pool: 30k, fixed 2,000-update continuation

2026-11-04 UTC+7. User approved expanding to30,000 TRAIN images, with both arms
starting again at original CSA10, NOT the prior XE/GROUND pilot endpoints.
Keep global32/2,000updates/64,000events, Adam/lambda/decoder/VALID policy unchanged.
Prior10k gains failed locked gates; this is a new exploratory experiment, not a
way to relabel that failure or guarantee improvement. TEST remains unused.

## Current authorized implementation: CPU label expansion

Select first30,000 of the SAME SHA-ordered Karpathy TRAIN/restval population.
Require exact original10k as prefix, checked against its known cohort SHA.
No new CLIP/detector/ROI extraction; reuse cached v134 boxes/offsets and same
lexicon, IoU>=.5/all-instance positives/invalidGT/crowd/UNK skips. No tuning the
lexicon or threshold alongside dataset size. Preserve old10k artifacts unchanged.

Output new cohort/summary/mentions/filtered instance annotations and review samples.
CPU only, same two inputs fullROI134 and COCO Karpathy JSON. Full label data must
be preserved; small review ZIP excludes mentions and instance payloads. After the
user-run result, seal actual counts/hashes and prepare transport for its exact
Save Version; do not invent a source version or accept unchecked generic cache.

## Sampler decision locked: option1, 2026-11-04

User: “ok cách1 nhé”. Round-robin one caption per image before a second reference
pass, with deterministic shuffled reference order per image and shuffled image
order per pass (seed42). First30k events cover all30k images once, next30k cover
all again, final4k give a third exposure to4k images. Same schedule for XE/GROUND.
Do not call64k events one full epoch over all ~150k references. Historical10k
comparison changes BOTH sampler and image pool; it does not isolate image-count
causality. Within30k the XE/GROUND contrast remains matched.

No quota from before the last
two pilots is assumed current. After labels arrive and sampler is chosen, prepare
exactly two new matched arms; no seed sweep/fulltrain/SCST/TEST automatically.

## Artifact/budget confirmation and implementation, 2026-11-04

User confirms source **Output155** of `ducanh2403/2-vi-cai-kia-qua-nhieu` and
budgets “như cũ”, interpreted as20h/23h per account as previously confirmed.
Audit source24ccd81 complete:30k images/150076refs/174465mapped/158197eligible.
Seal exact summary/cohort/label files via CPU byte-preserving save155. Dataset
slug `csa-grounding-audit30k-v155`; creation/mount remains user-run/unconfirmed.

Emit two new30k GPU notebooks using original139latest272f843a and same recipe as
the10k pilot: lambda0/.1, freshAdam1e-5/betas(.9,.99),global32/FP32/clip1,
2,000updates/64kevents, original inference MHA/beam5/max20/VALID5000. Source-file
instrumentation and early CUDA/DDP/metric guards remain. Store actual30k exposure
histogram in both stage receipts; require30k unique,26k2x/4k3x. Do not reuse
old10k pilot endpoints, replace the detector, increase lambda/updates or run TEST.
Automatic unreviewed weak-label status remains explicit. Parent4hcap is a hard
stop limit, not ETA or quota-billing forecast. Preserve old notebooks/data/results.
