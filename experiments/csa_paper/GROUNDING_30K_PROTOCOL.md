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
