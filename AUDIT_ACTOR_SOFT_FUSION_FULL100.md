# Frozen actor soft fusion: full detector candidate audit on100 new images

This evaluates the candidate method from R65/R66 beyond the previously inspected20-image and GT-matched pools. No head, weight, threshold or detector rule is refitted. Real extraction and annotation have **not yet run**; preparation and synthetic pipeline checks are complete.

## Kaggle run

Copy the complete cell from [kaggle_actor_soft_fusion_full100_features.py](kaggle_actor_soft_fusion_full100_features.py), or import [ACTOR_SOFT_FUSION_FULL100_FEATURES.ipynb](ACTOR_SOFT_FUSION_FULL100_FEATURES.ipynb).

- **T4 x2**, Internet **on**.
- Add MSCOCO and the existing object-detection cache. Paths are prefilled:
  - `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json`
  - `/kaggle/input/datasets/vuthetam/mscoco-2014/images`
  - `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json`
- No checkpoint input, grounded cache, prompt cache, old notebook output or API secret is required. Kaggle extracts frozen features; scoring uses saved local heads later.
- Pinned core commit: `20439e4`; CLIP revision: `57c216476eefef5ab752ec549e440a49ae4ae5f3`.
- Output: `/kaggle/working/actor_soft_fusion_full100_features_review.zip`. Send this ZIP and the Kaggle version number.

Wait for Adding data to finish. Official V-COCO val metadata is downloaded to obtain the eligible image IDs; there is no additional full COCO image or instance-annotation download. Model download and extraction print progress/30-second heartbeats.

## Frozen selection and model protocol

The exclusion manifest contains **1,711** images: the previous1,388-image manifest plus all expanded actor train/dev images. Select100 images by a deterministic shuffle (seed20261014) from eligible official V-COCO val IDs with caption train/restval metadata. Scores, detector confidence and GT relation labels do not select images. Missing cache records fail without replacing images or changing count.

Feature output preserves the complete selection list and eligible ID pool. The local scorer reconstructs the seeded selection, checks zero excluded overlap, input/protocol/head hashes, caption split, feature coverage, and exact candidate reconstruction. The detector-cache hash is locked to the R57 source. These images are new relative to the recorded scorer/development images, not necessarily CLIP pretraining or caption-model training.

Detector settings remain confidence0.5, same-class NMS IoU0.8, top10objects. Generate every ordered person→other-object pair, including other people and excluding self. “Full” means this retained inventory, not all raw detector boxes or general object→object relationships. There are six action predicates. No GT pair matching or labels are used during candidate generation/scoring.

All three checkpoints are hash-locked in `audit/actor_soft_fusion_full100_protocol.json`:

- R50 residual head `86930afcbed9573a4a48a2a0b5a7d94507d52b67aaa63b73e5a5f7717907a438`.
- Expanded actor head `cdfaae489407b13d32eb065d63eaa101f008303f61484d2a5c81d9258f0e7295`.
- Soft fusion head `5e7f838fe1fb585e3d41a090bef1617f61e868451225b7827ce5681b8b4f865d`.

## Local scoring and review after receiving the ZIP

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' score_actor_full100_frozen.py --source 'PATH_TO_NEW_ZIP' --residual-head 'C:/KhoaLuanTotNghiep/Code/output/clip_prior_residual_v1/residual_head.pt' --actor-head 'C:/KhoaLuanTotNghiep/Code/output/actor_all_endpoints_verifier_v1/actor_head.pt' --fusion-head 'C:/KhoaLuanTotNghiep/Code/output/actor_soft_fusion_v1/fusion_head.pt' --output 'C:/KhoaLuanTotNghiep/Code/output/actor_soft_fusion_full100_audit'
```

Outputs include all candidate scores, baseline/fusion structured prompt caches, full and marked JPEGs, paired-review HTML/CSV, a blind claim CSV and separate method-membership mapping. Ground-truth/human fields are left blank; independent annotation is pending. Score-margin ordering is not calibrated across predicates.

Review both methods' selected claims with statuses such as supported, contradicted or unclear. Record wrong actor, wrong object class, wrong action, duplicate physical object and missing visual evidence. For blind review, use `blind_claim_review.csv` without the method-membership mapping or paired-method HTML.

**Selected-claim review supports precision only.** To estimate recall, independently enumerate supported instance-level relations for each image within the retained inventory and six-predicate vocabulary, using `image_coverage_review.csv`. Do not infer recall from selected claims, object-only counts or absent V-COCO links. Detection-missing objects remain outside this conditional retained-inventory scope. Unclear/unannotated claims cannot silently become negatives.

Report counts before/after, accepted supported/contradicted/unclear claims, removed false links and lost true links. No automated quality metric is produced without annotations. Keep heads and thresholds fixed throughout; if this audit informs another method change, it becomes development evidence rather than an untouched benchmark.

## Completed checks

Two selection tests passed: image-order-independent seeded selection with excluded/caption-heldout filtering, and failure on insufficient pool. A synthetic100-image end-to-end fixture exercised scoring, both prompt caches, CSV and HTML outputs and rejected a changed head hash before inference. Blind review/coverage outputs and workflow/notebook syntax were also checked. Synthetic fixture results are not real image-quality results and are not entered as experiment metrics.
