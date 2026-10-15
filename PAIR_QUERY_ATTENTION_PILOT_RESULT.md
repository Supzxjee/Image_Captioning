# Candidate-conditioned patch reader: local pilot

**Result: no demonstrated benefit from candidate-conditioned attention over its query-blind control. Keep the caption/relation baseline.** The source-trained attention reader ranks 29/34 reviewed target contrasts correctly versus CLIP's 28/34, but the query-blind reader has exactly the same correct/incorrect cases. Image-macro accuracy is lower than CLIP, and calibrated selection accepts no claims. Nested target adaptation reduces all learned readers to 27/34. No production promotion or caption gain follows.

## Question and scope

Does reading individual CLIP patch tokens with a subject–object–predicate query improve relation discrimination over pooling the same tokens? This pilot changes the trainable reader while keeping the frozen image/text backbone, pair inventory, supervision and training schedule matched. It does not change the caption decoder.

The motivation follows interaction readers such as [HOICLIP](https://openaccess.thecvf.com/content/CVPR2023/html/Ning_HOICLIP_Efficient_Knowledge_Transfer_for_HOI_Detection_With_Vision-Language_Models_CVPR_2023_paper.html) and [CLIP4HOI](https://papers.neurips.cc/paper_files/paper/2023/hash/8fd5bc08e744fe0dfe798c61d1575a22-Abstract-Conference.html). Cross-attention alone is established prior work, not a novelty claim for this project. This is a small mechanism test, not a reproduction of either complete HOI system.

## Frozen patch evidence

The old region cache contains pooled descriptors, so it cannot recover individual patches. Re-extract all 1,141 source-training pairs and 310 existing development pairs from 744 official COCO images using the already cached CLIP ViT-B/16 snapshot `57c216476eefef5ab752ec549e440a49ae4ae5f3`. Source image bytes/dimensions and model/config hashes are recorded. The 61 existing review images are checked against their previous image hashes. Other required images are downloaded from the official COCO image host; the full COCO dataset is not downloaded.

Preserve both endpoints in a union crop with 0.1 padding; resize the entire RGB crop to 224x224 with bicubic resampling and apply standard CLIP normalization. Retain all 196 final 768D patch tokens after vision post-LayerNorm plus CLS, cached as float16. CPU extraction saves bounded shards; a verified prefix can resume. Batch size affects runtime, not the prescribed preprocessing.

For CPU head training, normalize each patch, apply one label-free fixed QR Gaussian projection (768x64, seed 1729), then normalize again. All controls use the identical compression. Append five geometric region support values and patch XY coordinates to each token. These supports are box geometry, not contact labels. The subject and object query inputs are ordered area-weighted means of the compressed patches, accompanied by the original ten box-geometry values and frozen candidate-specific 512D CLIP text.

The 14x14 grid is coarse and CLIP tokens are contextualized; neither region membership nor an attention map proves physical contact. The fixed compression is also a limitation of this small CPU pilot.

## Matched readers

Each learned reader instantiates the same 30,817-parameter head with hidden size 32:

1. **Mean pooling:** equal weights across patches; pair/text query still enters the final scorer.
2. **Query-blind attention:** learn patch weights using a zero pair/text input to the query networks. The patch keys still contain candidate-specific box support; this removes candidate/predicate information from the *reader query*, not all instance information from the system.
3. **Candidate-conditioned attention:** the ordered subject/object features, geometry and candidate text determine the query that reads patch keys/values.

Mean pooling has inactive key parameters, so equal instantiated parameter counts do not mean equal active capacity. The query-blind reader is the additional learned-attention control. The final scorer receives the candidate query, attended value and their product in every learned condition. Frozen union/text CLIP is reported separately as a reference. No hard R50 proposer gate is imposed in this pilot.

## Two fixed supervision stages

**Source training:** Exclude all 61 images in the annotation manifest, including uncompleted cases, from source fitting and calibration. Split the remaining source-training images with seed 1337 into 444 fit images and 111 calibration images. The original 128-image development split remains outside both. Use source-known labels only; action-absent zeros make a triplet false, whereas other unlinked targets remain unknown. Train each reader for 30 full-batch AdamW steps, seed 42, learning rate 0.001, weight decay 0.01, gradient norm cap 1. The loss is masked BCE plus 0.2 type-balanced actor/predicate hinge loss with margin 1. Fit per-predicate thresholds only on the separate source-training calibration images, requiring at least five known accepts and at least 90% empirical known-label precision. This rule can abstain and is not a precision guarantee on development data.

**Target adaptation:** Start from each preserved source-trained reader and use exactly the five nested image folds from R101. Fine-tune on the confirmed target substitutions of the inner fit images only, for 30 steps with learning rate 0.0003, weight decay 0.01 and gradient norm cap 1. Use equal-image reviewed BCE plus 0.2 logistic target ranking. Multiple true targets need not compete; unreviewed alternatives are not invented negatives. Inner calibration images alone fit the shared target temperature/acceptance tails using the existing 90% / five-known rule. Each outer reviewed image is predicted once. These target-only decisions do not include an action gate or a prompt cap.

The source stage never uses human relation labels. The separate nested target stage uses only its inner fit/calibration labels. Both the existing development split and the human review pool have been inspected repeatedly in previous experiments: image separation prevents this run's fit/evaluation overlap but does not turn them into a fresh independent benchmark. One seed and 34 reviewed contrasts cannot establish publication-level robustness or caption improvement.

## R105: extraction and consistency

All 1,451 pairs / 744 images were extracted successfully. Projection of the new CLS tokens through the frozen CLIP visual projection reproduces the original cached union features with minimum cosine **0.9999997616 across all pairs**. This includes the initial 100 single-image records and the resumed batch extraction. The recorded 530.7-second extraction time covers the resumed 1,351 pairs only; it is not the total download/extraction time. The earlier 100 pairs and image preparation took additional time.

Source fit has **721 pairs on 444 images**; source calibration has **187 pairs on 111 images**. The other 233 source-training pairs belong to the excluded 61-image review inventory. Fit supervision contains 463 positive, 3,526 known-zero and 337 unknown predicate cells; training contrasts comprise 269 actor and 1,894 predicate edges. These are the existing GT-matched detector candidate pools, not natural full-proposal prevalence or end-to-end HOI detection AP.

## R106: source-trained readers

All source heads use the same fixed schedule and separate calibration set. The human target pool is excluded from both source fitting and source calibration, but remains reused development evidence across the project.

| Source-trained reader | Macro AP on source dev known labels | Reviewed target order | Same-class target order | Image-macro target accuracy |
|---|---:|---:|---:|---:|
| Frozen CLIP reference | 0.172907 | 28/34 | 18/21 | 83.33% |
| Mean patch pooling | 0.209715 | 28/34 | 19/21 | 77.27% |
| Query-blind attention | 0.211461 | 29/34 | 20/21 | 81.82% |
| Candidate-conditioned attention | 0.211823 | 29/34 | 20/21 | 81.82% |

Attention and query-blind attention have **zero disagreement on correct/incorrect target ordering**. Compared with CLIP, each fixes four contrasts in three images and regresses three contrasts in three images. The extra net contrast win does not improve image-macro accuracy. Repeated predicates on one image are correlated, so 29 versus 28 is not evidence of a reliable improvement. Candidate-conditioned AP exceeds the query-blind control by only 0.000362 on this single reused development split; this is not a demonstrated practical advantage.

Every arm fails to reach the fixed calibration precision/support requirement for all six predicates, giving thresholds of 1.01. Consequently all arms accept **zero claims**, on both the source dev split and reviewed targets. Their selection precision is undefined and recall is zero. Higher rank/AP values do not make this selector usable.

On the first 50 development pairs, the candidate reader's mean attention entropy is 5.270621 versus 5.278115 for uniform weights. Its mean maximum patch weight is 0.006819 versus uniform 0.005102. Predicate-dependent attention has mean L1 difference 0.014039. Thus the implemented reader does respond to predicate queries, but in this diagnostic its weights remain close to uniform. These observations neither prove that query conditioning caused the score changes nor explain all errors.

## R107: nested target adaptation

| Adapted reader | Out-of-fold target order | Same-class target order | Image-macro target accuracy | Reviewed true / false accepts |
|---|---:|---:|---:|---:|
| Mean patch pooling | 27/34 | 15/21 | 87.12% | 0 / 0 |
| Query-blind attention | 27/34 | 15/21 | 87.12% | 0 / 0 |
| Candidate-conditioned attention | 27/34 | 15/21 | 87.12% | 0 / 0 |

All three adapted readers have the same correct/incorrect target cases. Candidate attention fixes three but loses five contrasts relative to its source-only version. Those five losses are concentrated in two images and concern riding/sitting_on. Mean pooling fixes four and loses five. Image-macro accuracy rises despite worse pooled and same-class ordering, illustrating why the aggregate measures must be reported together. No one metric is selected after seeing results to declare a winner.

All 68 unique reviewed claims abstain under the fixed inner-calibration policies. No actor gate or cap is involved in this result, so these abstentions cannot be attributed to the old R50 veto. Low acceptance is a property of this pilot's scores/calibration at the fixed rule; lowering thresholds after inspecting these examples would be a different experiment.

## Verification and decision

**15 relevant tests passed. All 18 saved heads were strictly reloaded.** Source logits, all out-of-fold logits/decisions, calibration temperatures/policies, development metrics and paired target changes reproduce exactly. The image-exclusion protocol was independently reconstructed. Extraction shard/image/model hashes and the new-versus-original CLIP consistency check pass.

Retain these heads as development artifacts and preserve the current baseline. Do not generate a full prompt cache or train the caption decoder from these scores. This particular small head, compressed feature representation and limited supervision have not shown a useful advantage from conditional patch reading. That is not evidence that all HOI cross-attention approaches fail.

A more substantial next hypothesis is to transfer an already interaction-trained representation, then test target attachment and calibration with matched controls, rather than adding more losses to this small reader. A directly conditioned person-pose backend and an interaction-pretrained HOI model are both **untested** here. No new external HOI checkpoint has been downloaded or evaluated, and there is no new CIDEr/CHAIR result or established paper contribution.

## Reproduction

Code: `extract_pair_query_patches.py`, `captioning/pair_query_attention.py`, `run_pair_query_attention.py`, `verify_pair_query_attention.py`, `tests/test_pair_query_attention.py`.

Local outputs: `C:/KhoaLuanTotNghiep/Code/output/pair_query_patches_v1` and `C:/KhoaLuanTotNghiep/Code/output/pair_query_attention_v1`. Image bytes, patch tensors, individual annotations and predictions remain local. Public aggregate files contain only protocol/result information.

The existing CPU torch environment is in `C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps`; Transformers 4.57.1 and its dependencies were installed separately in `.tmp/patch-deps`, without modifying the global environment. The caption backbone was already cached; it was not downloaded again.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/patch-deps;C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
$py='C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$clip='C:/Users/leduc/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3'
$package='C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1'
& $py extract_pair_query_patches.py --model $clip --package $package --output C:/KhoaLuanTotNghiep/Code/output/pair_query_patches_v1
& $py run_pair_query_attention.py --source C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip --patches C:/KhoaLuanTotNghiep/Code/output/pair_query_patches_v1 --package $package --review "$package/human_batch1_review.json" --reference C:/KhoaLuanTotNghiep/Code/output/factorized_action_target_relative_cv_v2 --model $clip --output C:/KhoaLuanTotNghiep/Code/output/pair_query_attention_v1
& $py verify_pair_query_attention.py --source C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip --package $package --review "$package/human_batch1_review.json" --output C:/KhoaLuanTotNghiep/Code/output/pair_query_attention_v1
```
