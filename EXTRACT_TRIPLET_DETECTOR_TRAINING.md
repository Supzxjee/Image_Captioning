# Detector-box training features with a matched GT-box control

This next experiment tests the GT-box→detector-box training gap after R72. The candidate-text verifier retains more true relations than its relation-bank control across three seeds, but still loses supported relations and can misbind actors. This workflow prepares the data to test detector-domain training; it does **not** claim the gap is already fixed.

## Run on Kaggle

Copy the complete cell from [kaggle_triplet_detector_training_features.py](kaggle_triplet_detector_training_features.py), or import [TRIPLET_DETECTOR_TRAINING_FEATURES.ipynb](TRIPLET_DETECTOR_TRAINING_FEATURES.ipynb).

- GPU **T4 x2**, Internet **on**.
- Add only MSCOCO and the existing detection cache. Paths are filled:
  - `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json`
  - `/kaggle/input/datasets/vuthetam/mscoco-2014/images`
  - `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json`
- No checkpoint input, old notebook output, prompt cache, grounded cache, feature ZIP upload or API secret is needed.
- Pinned core: `b034fee`; CLIP revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`.
- Output: `/kaggle/working/triplet_detector_training_features_v1_review.zip`. Send that ZIP and its Kaggle version number.

The GT reference metadata is in the repo; official annotations do not need another download. Kaggle matches cached detections and extracts frozen detector-crop features on both GPUs. No model head is trained there. GT-crop features for the control already exist locally and are reused, avoiding a second extraction.

Wait for Adding data to complete. The cell prints matching summaries, per-worker feature progress and30-second heartbeats.

## Locked data and matching

Reference train:1,776 pairs/826images, pooled and deduplicated from the original official training caches. Reference dev:489pairs/182images, unchanged from the expanded actor experiment. Train/dev are disjoint and use caption train/restval only. The prior100-image audit and364pair transfer cohort do not become training sources.

Use the same detector snapshot, confidence0.5, same-class NMS0.8 and top10 objects as the full-candidate inference workflow. Match by class-aware IoU-descending one-to-one greedy assignment with IoU>=0.5. Input/protocol/reference hashes are checked. Matching is an approximate label-transfer proxy; IoU alone cannot establish visual action/contact or disambiguate every merged detection.

Preserve source labels and GT IDs/boxes. Only crop coordinates/confidence/IoU are replaced by matched detector values. Missing roles remain unknown; unmatched proposals do **not** become false labels. Report pair/positive coverage, missed subjects/objects, invalid/missing cache cases and per-image matching. Exact matched counts are unknown until the real run; do not claim all reference pairs survive.

## Controlled local training after the ZIP arrives

Two models use **exactly the same matched train/dev identities and labels**:

1. `gt_control`: original cached GT-crop features and GT geometry, restricted to the matched subset.
2. `detector`: new detector-crop features and detector geometry for that same subset.

Architecture, candidate-specific text, seed42,30epochs, optimizer, losses and train-only threshold-fit rule remain identical. Text features are checked against the original frozen cache. This comparison isolates the box-feature/geometry domain under the fixed subset; comparing directly against the old full1,776-pair run would confound box changes with sample removal.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_triplet_verifier_cached.py --v2-source 'C:/Users/leduc/Downloads/instance_contrast_alignment_v2_diagnostic_1024_review.zip' --actor-source 'C:/Users/leduc/Downloads/actor_all_endpoints_features_v1_review.zip' --candidate-source 'C:/Users/leduc/Downloads/actor_soft_fusion_full100_features_review.zip' --scored 'C:/KhoaLuanTotNghiep/Code/output/actor_soft_fusion_full100_audit' --detector-source 'PATH_TO_NEW_ZIP' --train-condition gt_control --output 'C:/KhoaLuanTotNghiep/Code/output/triplet_detector_gt_control_v1'
# Repeat with --train-condition detector and output triplet_detector_trained_v1.
```

Both versions will be evaluated on the same detector transfer/audit inputs. Keep all missed true relations and false removals visible; do not select a seed or threshold from the donut/frisbee examples. These are already inspected development cohorts, not a new heldout test or full detection AP. Independent full-candidate labels and later caption evaluation remain necessary.

## Verification completed before real extraction

- Four existing matcher tests pass, covering wrong-class rejection, one-to-one identity, coordinates and preservation of unknown labels.
- A new CLI integration check passes for an explicitly locked nonlegacy cohort; the old500-pair default remains enforced.
- A synthetic matched-package test runs both training conditions. When box/features are identical, both models produce identical weights and thresholds. This tests paired alignment, not real domain adaptation quality.
- Workflow/loader/trainer syntax and notebook cell are verified. **Real detector-training extraction, matching coverage and model results are still pending.**

Protocol: `audit/triplet_detector_training_protocol.json`. Source GT reference metadata files are derived from the provided official COCO/V-COCO caches; no images or manual audit annotations are published with them. No weekly report document is edited.
