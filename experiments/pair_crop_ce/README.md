# Fullframe CE versus pair-crop CE — preparation only

Status: **TRAIN BLOCKED pending human labels and feature compatibility.** No encoder, notebook, inference or training has been run by the agent.

## Two tasks now

1. Open local `review/index.html`: up to10 skateboard and10 surfboard candidate groups, plus one old fit group235799 to reconfirm. Export `pair_crop_candidate_review.json`. Judge only the selected green person and orange object. Both descriptions may be true or uncertain; do not force a contrast. The 224px field is descriptive, not an eligibility filter.
2. Import `INSPECT_VISUAL_CACHE_METADATA.ipynb` into Kaggle, attach only visual-cache, Accelerator=None, Internet off. The user runs it and returns `visual_cache_metadata.json`. It reads HDF5 headers and ID arrays, not feature values, and downloads/runs no model.

The portable local `review/pair_crop_candidates_review.zip` includes the HTML,21 JPEGs, manifest and image receipts. Images and raw human exports remain ignored by Git. `CANDIDATE_MANIFEST.json` records the candidates with no semantic labels. After review, take the first4 valid groups in each family in the existing fixed hash order; never replace failures beyond these10/family. If fewer4+4 are valid, preparation is inconclusive and training stays blocked.

## Candidate provenance and limits

The original VCOCO action-role miner has no new same-image contrasts left after known exclusions. Full UPT selected relation proposals contain12 skateboard and13 surfboard images with distinct selected people and distinct target objects. We take10/family by fixed hash, not caption outputs. UPT stand_on/riding and carry/holding are **mining hypotheses only**, not semantic labels.

The local VCOCO instances file covers5400images, not full COCO. Requiring annotation matches would retain only one surfboard image and no skateboard image. Accordingly, use a matched COCO annotation box at IoU>=.5 where available; otherwise show the original UPT detector box and require human confirmation. Box provenance is recorded per event. Both future arms must use identical confirmed boxes. This new probe may mix annotation and confirmed detector boxes; do not call all boxes GT or conflate its scores with the old GT-only pilot.

548 known prior IDs are excluded from new candidates, including all30 original feasibility groups. Exclusions additionally scan local experiment JSONs under12MB, not all historical runs/remote logs. Train/restval may have been seen by the caption backbone. No independent whole-system test claim.

JPEG decode, dimensions and EXIF orientation were checked. dHash64 distance<=6 flagged no near duplicates among the20new images or against53old feasibility JPEGs. This limited perceptual-hash screen does not certify semantic or exhaustive historical independence.

## Locked experiment plan, not yet training notebooks

- FULLFRAME-CE and PAIRCROP-CE both re-encode pixels with the same frozen CLIP ViT-B/16 revision57c216476eefef5ab752ec549e440a49ae4ae5f3, FP32 eval, noTF32/autocast. RGB, strict decode, orientation absent/1; bilinear antialias224square, CLIP normalization, no center crop/letterbox.
- Crop: union of selected person/object; expand10% per side; floor left/top, ceil right/bottom; clamp image boundaries. `crop_geometry.py` implements exact coordinates and patch-center mapping back to original image space. Interaction mask stays the original unexpanded union. Explicit8Dgeometry remains original-image based, normalized fit-only.
- Cache P_full andP_crop FP16 and read FP32. Recompute one shared G32x512 from newly encoded fullframe197tokens after quantization, using frozen baseline projection/ReLU/QFormer. G is identical for both events in one image and for both arms.
- Before training compare new fullframe features against available original cache rows: elementwise abs<=.01+.005abs(old), relativeL2<=1e-3. Thresholds are technical prespecified gates, not an inference of semantic quality; do not tune them to pass. Missing metadata is unknown; failed tensor compatibility blocks training pending explanation.
- Reset both arms to preserved baseline SHA5f8eead3..., same initialization hashes and seed42. Same evidence6tokens andK/VLoRArank8; same CE includingEOS; no CROSS/SCST/routing/newloss.300updates, globalbatch8groups, endcheckpoint only, original optimizer settings. Never use the trained CE adapter as only one arm's start.
- Keep14fit groups unless blind recheck235799 cannot confirm its original contrast; then remove the whole group from both arms, leaving13, no substitute/relabel. Regenerate and lock fit-only schedule/statistics. The7oldholdout groups are exposed development cases.
- Primary decision: blinded human judgment on new8same-image groups, both target descriptions correct and sufficiently specific; allow paraphrases and distinguish generic, false, wronginstance and unresolved. Include all16events; missing/truncated captions are not full successes.
- Practical GO: at least6/8groups correct, netgain>=3, regressions<=1, and gain in both families, robust to unresolved judgments. Not statistical proof. Valid run not meeting this stops this variant; missingdata/technicalfailure/unresolveddecision is inconclusive. No crop-margin/seed/step search to rescue results.
- Two eventual training notebooks, one arm per notebook using2T4, sharing a hashed feature cache from the first run. No training notebooks until labels and input checks resolve.
- After a GO, next milestone must evaluate full captions with real UPT proposals and matched caption CE/SCST+object baseline. Crop/CE/LoRA are not claimed novel, and phrase gains do not imply CIDEr gains.

## Local validation

Three pure crop geometry tests PASS; AST and unexecuted metadata notebook PASS; candidate uniqueness and blank-label checks PASS; isolated DOM test covers startup, navigation, blank export guards and JSON export PASS. No real-browser screenshot QA. Local h5py/torch are unavailable; actual HDF5 reading and all GPU/feature checks remain for Kaggle. Local operations only read annotations/prompt JSONs and collected official public JPEGs.
