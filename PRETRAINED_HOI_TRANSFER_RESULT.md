# Frozen interaction-pretrained UPT: target attachment audit

**A useful positive recognition signal, with insufficient coverage for replacement.** Frozen UPT-R50 trained on HICO-DET correctly orders **20/20** comparable reviewed target contrasts versus CLIP's **17/20**. It fixes three errors on two images and regresses none on that subset. Only 20/34 contrasts meet the fixed localization and ontology requirements, so the prespecified coverage gate fails. No caption score or production improvement is claimed.

## What changed

Use the [official UPT implementation and pretrained HICO-DET model](https://github.com/fredzzhang/upt), rather than fit another small head on the local labels. UPT is the model from *Efficient Two-Stage Detection of Human-Object Interactions With a Novel Unary-Pairwise Transformer*, CVPR 2022. This is an external strong baseline, not our new architecture or a reproduction of its full HICO benchmark.

Use HICO-DET relation pretraining to avoid deliberately evaluating a V-COCO-trained relation model on the V-COCO training images in our review pool. The detector's COCO object pretraining may have seen these images, and exact web-image overlap against HICO has not been exhaustively checked. Do not describe this as guaranteed unseen-image evaluation. No local fitting or threshold search occurs.

Pinned sources:

- UPT: `192743045e4c5f6827299fdc15809f5b97ac1b8e`.
- DETR submodule: `820d17abc697e8dd5f49d4cd1fdbde412545bac5`.
- HICO metadata: `9dd59c1f6ee2ddc011567765d0bc644858391fea`.
- Pocket transformer modules: `bff287098bb73427fd8d5525b2ac9867fb56629d`.
- Official [UPT-R50 HICO weights](https://drive.google.com/file/d/1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z/view): 219,754,181 bytes, SHA256 `12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c`.

All **766 state tensors** strictly load into the 54,762,778-parameter model. Upstream model source and forward equations are not edited. CPU inference uses its original resize-to-short-side-800/max-size-1333 rounding and ImageNet normalization. Preprocessing was compared directly with the pinned upstream transformation on all 22 images and is bit-exact.

The existing torch nightly cannot load the stable TorchVision 0.24 C++ extension on this Windows host. A compatibility layer enables pure-Python ResNet imports, supplies deterministic pure-torch IoU NMS with a stable input-order tie break, imports only the needed upstream Pocket transformer file, and disables an unnecessary ImageNet initialization download before strict checkpoint loading. PIL handles image IO. These compatibility details are disclosed; the official benchmark mAP was not rerun in this port. Dependencies are isolated in `.tmp/upt-deps`.

## Fixed evaluation protocol

Run the frozen model on the 22 images containing the existing 34 eligible human target contrasts. Verify original image bytes and preserve the supplied subject, relation and two alternate localized targets.

Associate source detector instances with UPT detections by **class-consistent one-to-one maximum-total-IoU matching**, keeping matches with IoU >= 0.5. Matching consults boxes/classes, not positive/negative relation labels. One UPT detection cannot serve as both alternate instances.

Map the six project predicates to HICO `hold`, `ride`, `sit_on`, `carry`, `eat`, `talk_on`. Primary comparison requires that all three instances localize and **both** target object/verb combinations are represented in the pretrained HICO ontology. A missing or untrained combination is unscored, not a negative label. This scope restriction must accompany the accuracy.

Primary scores are the learned verb logits before detector confidence and the native object/verb prior. Secondary scores use the original UPT output, which multiplies sigmoid verb scores by the native prior. Detection confidence alone is a control. Always compare CLIP on the exact same covered subset; do not compare 20/20 with CLIP's 28/34.

The go rule was fixed before inference: at least 80% of all 34 cases covered and at least three more raw-logit wins than same-subset CLIP. Passing would permit a larger audit, not automatic production use. All previous development-pool limitations remain.

## Coverage and measured results

| Coverage state | Contrasts |
|---|---:|
| Complete source review pool | 34 |
| All three instances localize | 27 |
| Localize and both object/verb combinations are supported | **20** |
| Localized but outside the trained ontology | 7 |
| Localization incomplete | 7 |

The seven localization failures comprise three missing positive targets and four missing alternate targets. All subjects localize. Primary coverage is **58.82%**, below the 80% go requirement.

| Same 20 covered contrasts / 15 images | Correct target ordering |
|---|---:|
| CLIP prior | 17/20 |
| UPT raw learned verb score | **20/20** |
| UPT official score including native priors | 18/20 |
| Detection-confidence-only control | 13/20 |

| Same 16 covered same-class contrasts / 12 images | Correct target ordering |
|---|---:|
| CLIP prior | 13/16 |
| UPT raw learned verb score | **16/16** |
| UPT official score including native priors | **16/16** |
| Detection-confidence-only control | 11/16 |

Raw interaction scores fix three CLIP errors on two images: carrying on image `107843`, and riding/sitting_on on image `329469`. All fixes concern same-class alternate targets. The official prior preserves these same-class fixes but regresses two different-class contrasts relative to its raw score. Its detection-confidence control is worse, so the raw-score gains cannot be reproduced by simply selecting the higher-confidence detector box in this audit.

Three illustrative overlays were inspected for gross subject/target misassignment. Matching IoUs for the corrected triples range from about 0.757 to 0.952. The quad-bike/motorcycle examples attach the rider to the vehicle under that rider; gold UPT boxes align with the supplied instances. This is a purposive, method-visible inspection using existing human labels, not an additional independent accuracy benchmark. The figure remains local at `pretrained_upt_hico_target_audit_v1/fixed_target_matches_3.png`.

**20/20 is not a reliable 100% generalization claim or a precision estimate for generated prompts.** The sample is small, repeatedly inspected, and conditions on a known true versus known false target. Three corrected predicates are concentrated on two images. Action-existence rejection, calibrated absolute acceptance, natural full-candidate precision and caption quality are not measured here.

## Exploratory fallback diagnostic, separate from the fixed result

After seeing the teacher outputs, a deterministic **pairwise decision** rule was checked: use the UPT preference if both candidates meet localization/ontology requirements; otherwise retain the original CLIP preference. The route uses no correctness labels, per-image exceptions or fitted score threshold.

This produces **31/34** correct comparisons versus CLIP **28/34**, and **21/21** same-class comparisons versus **18/21**. UPT handles 20 comparisons and CLIP handles the other 14.

This is **post-hoc exploratory evidence**, not a passed fixed criterion. It does not provide a unified comparable score over arbitrary candidate sets, ensure transitive preferences, calibrate relation acceptance, or generate captions. It does not change the failed 80% teacher-coverage gate. No production prompt cache is generated from this result.

## Verification and decision

Twelve relevant tests pass across matching/NMS/preprocessing, caption-evidence aggregation and reviewed targets. All 22 cached output hashes, instance assignments and target metrics were replayed. **30,601 official output score cells** were reconstructed exactly using the upstream vectorized sigmoid/prior computation. One full model forward pass reproduces exactly. The official secondary gaps are taken directly from stored native outputs; a scalar-versus-vector sigmoid rounding discrepancy discovered during verification was corrected without changing any aggregate result.

Keep the frozen external baseline and preserve the positive attachment signal. Do not replace the existing caption pipeline or claim a new scientific contribution. The immediate research direction is to use interaction-pretrained evidence where applicable while retaining uncertainty/fallback; it still needs broader, independently fixed coverage/acceptance evaluation and an actual caption experiment before claiming CIDEr/CHAIR improvement. Do not return to another loss sweep on the tiny randomly initialized heads.

Artifacts: `audit/pretrained_upt_hico_target_result.json`, local `C:/KhoaLuanTotNghiep/Code/output/pretrained_upt_hico_target_audit_v1`, and its compact review ZIP. Raw images, case notes and model weights are not committed.

## Reproduction

Clone the official source and its pinned DETR/HICO submodules to a separate reference directory; use the revisions above. Only code/metadata and the pretrained checkpoint are needed, not the HICO image dataset. With the isolated CPU dependencies available:

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/upt-deps;C:/KhoaLuanTotNghiep/Code/.tmp/patch-deps;C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
$py='C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$common=@('--upt','C:/KhoaLuanTotNghiep/Code/.tmp/upt-reference','--pocket','C:/KhoaLuanTotNghiep/Code/.tmp/pocket-reference','--checkpoint','C:/KhoaLuanTotNghiep/Code/.tmp/upt-r50-hicodet.pt','--source','C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip','--package','C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1','--review','C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1/human_batch1_review.json','--output','C:/KhoaLuanTotNghiep/Code/output/pretrained_upt_hico_target_audit_v1')
& $py run_pretrained_hoi_audit.py @common
& $py verify_pretrained_hoi.py @common
```
