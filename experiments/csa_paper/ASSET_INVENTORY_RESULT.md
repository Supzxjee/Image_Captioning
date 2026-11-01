# Actual CSA assets — CPU inventory COMPLETE

02/11/2026. User-run ZIP SHA256 `6861902ed82d1a0f087f90c9c0b8656becb4512aa7203fb4308819bfff3c5942`, core `9167917bd7cdf6a6284a332b4218923bcdb59d4b`. Five text/JSON members cached in `asset_review_text/`. Four source hashes independently checked against Git and the notebook package.

## Evidence

- All123287 COCO IDs/filenames match Karpathy across3H5shards (41095+41096+41096); no missing/duplicate IDs according to completed user-run assertions. Shape197×768/rawCLIP/fp16, complete flags/coco_id declaration and common bilinear224 preprocessing metadata pass. **This verifies metadata/ID tables, not the actual feature arrays, complete H5 byte hashes or pixels.** Legacy model revision remains null.
- TRAIN=train82783+restval30504=113287images; val5000/test5000. Actual TRAIN caption events **566747**, not exactly5×113287. Vocabulary10000 including4specials; payload SHA `924be210aa0daa8b0968a6d69ad2c66b3de603f49c35f4aa67247c75d6173fc7`. Agent independently reproduced the exact ordered vocabulary and caption-event count from the locally available, SHA-matched source JSON using NumPy/stdlib only. No POS or text encoder run.
- Pinned detector source hash passes; every dataset image has a record, all stored boxes pass finite/confidence/positive-xyxy numeric checks. **867045 stored detections**, mean7.0327366/image, max87. No image exceeds cap150. Thus the whole cache is not globally restricted to6/10boxes; this does **not** recover producer confidence threshold, NMS or possible prior truncation.
- **1052 genuine empty records (0.8533%)**, distinct from missing records. Future ROI preparation uses exactly one full-frame mean token for these images per protocol, not synthetic object detections. No such ROI features have been produced in this inventory.
- Twelve deterministic RGB header samples, three per original split, opened successfully; no sampled box exceeded image boundaries. Numeric checks and this sample do not prove all-image box-coordinate/source-image correctness. Before pooling each image, use its true dimensions and retain boundary/error counts. Producer provenance is still unknown.
- User-run3parser/static tests PASS. Worker22.0924s; not full notebook wall/billing or model throughput. No feature arrays, Torch/model, scorer, GPU, ROI extraction or training used in this inventory. Summary and wrapper agree COMPLETE and training_ready=false.

## Decision and next stage

**Proceed to build actual semantic banks and a bounded real-data preparation/smoke**, keeping the approved architecture. Freeze TRAIN vocabulary from this result; build noun/verb POS lists solely on TRAIN, encode with pinned frozen CLIP text, L2normalize512D. Do not use val/test captions to construct bank words. Actual bank sizes may differ from paper and are not known yet.

Real-data stage must load/check actual patch arrays for selected images, obtain each image's dimensions and pool predicted ROIs with the existing fractional-area rule. Keep unknown detector/visual-encoder revision explicit; no claim of exact CSA full-system reproduction or independently recovered extraction provenance. Do not repeat this CPU inventory simply to move forward. No independent test/metric result, full training run or new method established here.
