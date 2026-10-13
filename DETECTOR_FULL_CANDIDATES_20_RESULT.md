# Detector full-candidate audit: 20 images

The frozen residual head is not ready to build a full relation-prompt cache. It accepts a wrong actor–food link and confuses a worn backpack with holding. The broader candidate pool exposes failures hidden by a GT-matched pair evaluation.

## Protocol and counts

- Input: `residual_detector_full_candidates_20_features_review.zip`, SHA256 `2085e94ac3a03293fa8f8a2853f74e70b6912bf7e20f81e605106b5509913cdb`.
- Feature workflow: `85c7a46`, core `f9bd244`; Kaggle version not supplied.
- Frozen R50 checkpoint SHA256 `86930afcbed9573a4a48a2a0b5a7d94507d52b67aaa63b73e5a5f7717907a438`; no training, threshold adjustment, or rule changes.
- 20 development images previously included in the R53 cohort, 138 retained detections, 600 ordered person–other-object pairs, six predicates per pair (3,600 scores).
- Detector confidence >=0.5, same-class NMS IoU 0.8, at most ten detections/image. “Full” refers to this retained inventory, including other people, not every raw detection or general object–object relation.
- Six claims pass frozen thresholds; the existing cap of three relations/image removes one. Five selected claims appear in three images; 17 images use object-only prompts. This fallback fraction is not recall.
- Input hashes, model revision, exact candidate reconstruction, feature coverage and numerical checks passed.

## Visual inspection of all five selected claims

This is an assistant visual inspection of full and marked images, not independent human annotation or benchmark ground truth. Statuses distinguish visible support from uncertainty; no precision estimate is inferred.

| Image suffix | Selected claim | Visual assessment |
| --- | --- | --- |
| 304248 | person holding backpack | Contradicted action: orange backpack hangs from shoulder/back; holding is not visibly supported. |
| 318672 | person sitting_on bed | Unclear posture: person reclines against pillows with legs on bed. Contact with bed is clear; a precise sitting label is ambiguous. |
| 528582 | foreground child eating hot dog | Eating is supported, but exact hot-dog identity/filling is unclear. Same physical food is also detected as sandwich. |
| 528582 | foreground child eating sandwich | Supported at broad sandwich/bread-food level: child has food at mouth. |
| 528582 | background person eating hot dog | Contradicted actor binding: background person is separated from food held by foreground child. |

The additional pre-cap claim is the same background person eating the sandwich. It is also a wrong actor binding; the cap hides one error without validating the remaining claims.

The sandwich and hot-dog boxes overlap almost exactly (IoU about 0.993). Same-class NMS intentionally keeps both competing class labels, so instance IDs derived from class+box treat one physical object as two. The selected relation list therefore duplicates a physical food interaction. Exact hot-dog taxonomy remains uncertain, regardless of this duplication.

Scores are similarities, not probabilities. Holding-backpack clears its threshold by only 0.000457 and sitting-on-bed by 0.001014. In contrast, the wrong background-person/hot-dog link clears the eating threshold by about 0.0961. Simply increasing a global threshold cannot be assumed to fix actor binding while preserving useful relations.

## Next development step

1. Preserve this run and frozen outputs as a development failure audit; do not overwrite it with post-hoc filtered prompts.
2. Resolve competing labels for the same physical detector region before generating semantic prompts. Use a separate grouping/label-resolution stage rather than indiscriminate class-agnostic NMS, which can erase genuinely distinct overlapping objects.
3. Add an explicit same-object/different-person contrast to alignment, using confirmed training links and known negatives only. Missing links remain unknown. The union crop can include the true actor even when the supplied subject is wrong; a candidate scorer must demonstrate that it uses the particular subject crop.
4. Test actor competition and counterfactual subject sensitivity offline on existing features first. Do not require a single actor per object: some interactions involve multiple people. Geometry can supply evidence, but a hard distance rejection alone would not establish semantic alignment.
5. Recheck action specificity (holding versus wearing/carrying) and audit a new disjoint candidate cohort after the development change is fixed. This 20-image set cannot serve as an untouched final benchmark after inspecting these failures.

No caption generation, fusion, hallucination benchmark, or caption-quality improvement was measured here.

## Artifacts and reproduction

Local results: `C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored` contains all candidate scores, selected prompt cache, CSV and HTML with full/marked images. Visual annotations are stored separately in `assistant_visual_review.json`; original predictions and the human-review CSV fields are preserved.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' score_detector_full_candidates.py --source 'C:/Users/leduc/Downloads/residual_detector_full_candidates_20_features_review.zip' --checkpoint 'C:/KhoaLuanTotNghiep/Code/output/clip_prior_residual_v1/residual_head.pt' --output 'C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored'
```
