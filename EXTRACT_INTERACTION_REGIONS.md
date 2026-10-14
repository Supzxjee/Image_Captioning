# Shared-union interaction-region diagnostic — R82

The scalar soft-prior experiment did not improve whole-triplet transfer acceptance. This next prototype tests richer spatial evidence: subject/object features are pooled from the **same union crop**, rather than two independently resized crops or one global union score.

This is a feature extraction workflow, not a trained relation model. No improvement or scientific novelty is claimed yet.

## Run on Kaggle

1. Create a separate notebook and select **GPU T4 x2**, Internet enabled.
2. Add only `vuthetam/mscoco-2014` with the COCO JSON/images paths already used. No detector input, checkpoint notebook output, prompt cache, API secret, or old output version is required; exact existing detector-pair metadata is pinned in the repository.
3. Import `INTERACTION_REGIONS_FEATURES.ipynb`, or copy the entire cell from `kaggle_interaction_regions_features.py`.
4. Run All / Save & Run All. Both GPUs independently extract frozen CLIP features; progress is printed per25pairs and parent heartbeat per30seconds.
5. Send `/kaggle/working/interaction_regions_v1_review.zip` after `REGION FEATURES READY`.

The cell pins core commit`558b518`, CLIP ViT-B/16 revision`57c216476eefef5ab752ec549e440a49ae4ae5f3`, and the existing COCO metadata hash. It creates a distinct output. Dataset mounting happens before the Python cell; heartbeat messages cannot describe Kaggle's pre-kernel input mounting.

## Extracted evidence

For each pair, crop both complete detector boxes with.1 union padding, resize that complete crop to224x224, and retain the CLIP vision patch sequence on its14x14grid. Official Hugging Face [CLIP documentation](https://huggingface.co/docs/transformers/model_doc/clip) describes `last_hidden_state` and pooled CLS outputs.

Pool six768-dimensional descriptors:

1. Subject region.
2. Object region.
3. Rectangle intersection.
4. Center-to-center corridor outside both rectangles, expanded byhalfpatch.
5. Context outside both rectangles.
6. Union CLS.

Patch supports use fractional overlap areas, preserving small sub-patch boxes rather than relying on patch-center membership. Exact empty regions receive zero features and valid=False. Final patch hidden states receive the existing frozen postLayerNorm; pooling outputs are L2 normalized. These are768D visual hidden features, **not**512D CLIP image/text similarity embeddings. No patch/text cosine interpretation is assumed.

Box supports are geometric approximations. An overlapping rectangle is not a contact label; corridor/background descriptors are contextual, and CLIP self-attention mixes information across the crop.14x14resolution may still merge small objects. Resizing whole crops preserves their content but distorts aspect ratio. These limitations are part of the diagnostic.

## Cohorts and planned ablation

| Split | Pairs | Use |
|---|---:|---|
| Train | 1,141 | Same known detector training labels; unknown masked |
| Dev | 310 | Same detector development pairs |
| Transfer | 364 | Reused R55 matched detector pool |
| Full100 | 1,321 | All detector candidates on inspected100images |

Total3,136pairs. Metadata/source hashes are locked in `audit/interaction_region_protocol.json`. Labels and pair identities are copied from previously verified source archives; no spot-check labels enter training. These are development cohorts, not fresh heldout evaluation.

`InteractionRegionVerifier` is implemented as a candidate-text-conditioned head over ordered region descriptors, validity, support mass and box geometry. Planned control repeats the same union CLS descriptor in every valid region slot, with identical masks/geometry/text/dimensions and training schedule. This tests local region **content** beyond aggregate union evidence and geometry. The head has not been trained on real extracted region features yet.

## Validation and size

Four local geometry/head tests cover fractional tiny boxes, exact empty supports, crop coordinate remapping, actor order, candidate text conditioning, and identical-size union-only ablation. The complete3,136pair/shard validator also passes with synthetic features and the actual frozen pair metadata. Synthetic tests are not model-quality evidence. Real pretrained GPU extraction remains pending.

Kaggle validates shard/index coverage, split order, metadata/model hashes, train/evaluation separation, region shapes, normalization, empty masks and ROI geometry. Output contains pooled regions/masks/mass and metadata, not full196token grids or CLIP weights. Raw feature tensors are approximately58MB before serialization/compression; actual ZIP size is unknown until extraction. No epoch checkpoints are included.
