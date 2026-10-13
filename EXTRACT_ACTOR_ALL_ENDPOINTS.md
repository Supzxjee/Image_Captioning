# Extract all confirmed actor contrast endpoints

This run expands actor verifier supervision. It does **not** train a caption model or actor head on Kaggle. Local trials are documented in `ACTOR_COMPATIBILITY_LOCAL_RESULT.md`: the nonlinear verifier fixes one inspected wrong-person case but has19/40 dev actor wins; the compact verifier abstains for all20 images. Neither is promoted.

## Run on Kaggle

Import `ACTOR_ALL_ENDPOINTS_FEATURES.ipynb`, or paste the complete cell from `kaggle_actor_endpoints_features.py`.

- GPU: **T4 x2**; Internet **on**.
- Only MSCOCO input is needed. Paths are already filled:
  - `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json`
  - `/kaggle/input/datasets/vuthetam/mscoco-2014/images`
- Pinned code: `7eb37b7`; frozen CLIP `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`.
- No Q-Former checkpoint, relation prompt cache, visual-cache, detection cache, Kaggle API token or old notebook output is required.
- Wait for Adding data to finish before running. The cell prints stage changes and30-second heartbeats.

Official V-COCO train/val labels and COCO instance annotations are downloaded. The instance annotation archive may take time; it is metadata, not another full image download. Both GPUs extract frozen subject/object/union and six relation text features. No `--train` flag is invoked.

Output:

`/kaggle/working/actor_all_endpoints_features_v1_review.zip`

Send that ZIP and the Kaggle version number. Training and verifier evaluation will then run locally. Exact pair/edge/image counts are printed after preparation; they are not invented in advance.

## Data selection

Use every confirmed actor contrast endpoint from the eligible official training split, instead of selecting1,024 general relation pairs. Enumerate all confirmed other-actor negatives, without the old four-negative cap. Unknown action/role links are never turned into negatives. This tests expanded actor supervision; it does not provide exhaustive wrong-object labels.

New development candidates come from official V-COCO val after excluding1,388 image IDs recorded in `audit/actor_verifier_dev_exclusions.json`: previous diagnostic/VLM images and the388-image R53 cohort. Train/dev images must be disjoint. Caption validation/test images are filtered from both. New development images are new relative to these listed scorer experiments, not necessarily CLIP pretraining or caption-model training.

Boxes and classes are GT supervision. Detector performance and relation-prompt accuracy still need separate evaluation. The existing20-image set remains a previously inspected development audit. The exclusion manifest is a record of used images, not a claim that every image previously used in all possible experiments is known.

## Checks and later local command

Official label hashes, input hashes, exclusion hash, split overlap checks, pair hashes and the pinned feature model are saved in the archive. Five actor/supervision checks and16 existing protocol tests passed locally. Workflow syntax and notebook cell were checked; the real expanded feature extraction has not yet run.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_actor_compatibility_cached.py --train-source 'PATH_TO_NEW_FEATURE_ZIP' --candidate-source 'C:/Users/leduc/Downloads/residual_detector_full_candidates_20_features_review.zip' --scored 'C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored' --output 'C:/KhoaLuanTotNghiep/Code/output/actor_all_endpoints_verifier_v1' --all-actor-edges
```

This command retains the fixed30-epoch development configuration and train-only threshold rule. Do not alter the existing outputs or choose an epoch based on the child image.
