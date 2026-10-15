# Pose-guided riding/sitting target evidence: R102–R104

Implemented frozen pose extraction, target-local anatomical features, a matched detector-box anchor control and CPU cross-validation. The current pose feature branch does not improve target ordering. Its most important limitation is coverage on the cases that CLIP gets wrong:3 of4 wrong comparisons lack usable pose.

## R102: extract subject-bound pose evidence

Use [Google MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python), package0.10.35, official Lite model version1. Model SHA256: `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`. The frozen model outputs image-normalized landmarks with predicted visibility/presence; these are not ground-truth contact or action labels.

Run on the existing61 local annotation images. The complete matched candidate pool on these images has130 unique detector subject boxes. Extract from15%-padded subject crops, allowing up to4 poses per crop. Retain only a uniquely matched pose with at least3 confident torso points, a confident hip, torso/hip agreement with the subject box and landmark-box IoU>=0.15. Visibility and presence minima are0.5; competing acceptable pose matches need a0.1 score gap. These are fixed engineering rules, not validated pose-identity probabilities.

| Initial ROI extraction status | Subjects |
|---|---:|
| Passes predicted subject-match rules | 92 |
| Detected pose does not confidently match subject | 16 |
| No pose | 19 |
| Ambiguous | 3 |

All source image bytes/dimensions were checked against the existing image provenance. Image dimensions were added from that verified provenance without modifying extracted landmarks, allowing aspect-correct geometry. Four illustrative scientific overlays were inspected for gross coordinate/subject alignment; this is not an exhaustive human keypoint-accuracy evaluation. Raw images, overlays and landmarks remain local.

## R103: actual joints versus box-derived anchors

Focus on the already reviewed21 riding/sitting_on comparisons in12 images/14 subjects:17 same-class and4 different-class target substitutions. Ten focus subjects pass pose matching; sixteen comparisons have usable pose. Missing/ambiguous pose gives an exactly zero learned correction, retaining the frozen CLIP prior.

Use the exact R101 five image folds and their separate inner-calibration images. Each reviewed case is evaluated out of fold once. Only focus training cases receive pose-head supervision; inner calibration retains all original calibration cases for the same shared threshold rule. Focus training counts by fold are15,8,15,14,9. No outer evaluation labels select weights, epochs or thresholds. This remains a reused single-reviewer development pool.

Three arms:

- Frozen relative CLIP prior.
- Box-anchor control: deterministic hip/knee/ankle positions from the subject detector box.
- Predicted pose: actual confident hip/knee/ankle positions and visible knee angles.

Both learned arms use the SAME pose availability, landmark visibility and subject-matching quality. Thus their difference concerns anatomical coordinates/angles given those masks, not merely whether a pose detector succeeded. Both have26 parameters, zero initialization, a0.5-logit correction bound,30 AdamW steps,lr0.01,weight decay0.01,seed42 and identical image-weighted BCE+.2 confirmed-target ranking. Other predicates are unchanged. Multiple supported targets remain allowed and unknown alternatives are not negative labels.

The12 features describe hip displacement from the target, hip/knee/ankle distance to its box, hip-inside fraction, knee angle, explicit visibility masks and subject-match quality. Distances/angles account for image aspect ratio. These are evidence for learning, not rules such as hip-inside implies riding.

| Focus result | CLIP prior | Box anchors | Actual pose |
|---|---:|---:|---:|
| Correct target ordering,21 comparisons | 17/21 | 17/21 | 17/21 |
| Correct ordering on17 same-class comparisons | 15/17 | 15/17 | 15/17 |
| Reviewed true/false claims retained after frozen action gate and cap | 0/0 | 4/0 | 4/0 |
| Recall of21 reviewed positives | 0% | 19.05% | 19.05% |
| Total claims selected from matched candidate pool | 0 | 6 | 4 |
| Selected claims lacking a human label | 0 | 2 | 0 |

The two learned arms retain exactly the same four human-supported claims, in3 images across2 folds. The extra2 box-control claims lack human review; their absence in the pose arm does not prove better hallucination control. Four reviewed successes do not establish reliable100% precision. The CLIP selector's precision is undefined because it selects none under this calibration rule.

On the16 pose-available comparisons, CLIP is already correct on15; the5 missing-pose comparisons contain3 of its4 errors. Those missing-error cases cover2 images, including two predicates for the same rider/vehicle instance. Missing pose gives no opportunity for this branch to fix those errors. One error has pose but is not fixed. This limits what the null result can say about pose evidence in general; it does not validate the current feature/head either.

The actor/action classifier and its calibration are frozen from R101. These results therefore test target-local pose evidence, not a newly trained posture-aware action classifier. A learned pose correction is not used as a mandatory action prerequisite, and no caption decoder is adapted.

## R104: label-blind full-image context fallback

To check the extraction limitation, run full-image MediaPipe inference for all subjects with no_pose/no_confident_subject, independently of their relation labels or model errors. Process22 images with up to8 poses per image. Keep the same visibility, matching and ambiguity rules. Preserve all92 accepted ROI poses exactly; do not override the3 ambiguous ROI cases. A full-frame pose reused by competing fallback subjects is rejected as ambiguous.

This recovers2 additional subjects:94/130 now pass the predicted-match rules. Both recoveries lie outside the existing cross-validation image pool. Feature rows in all CV fit/calibration/evaluation images are unchanged, so no extra CV training is necessary and no improved relation result follows. The3 initially missing hard comparison cases remain without usable pose. Neither source threshold nor matching confidence was lowered to obtain a result.

## Decision

Keep the CLIP/caption baseline. No ranking advantage over the matched box control or CLIP was observed, so this branch is not promoted. There is no CIDEr/CHAIR result or established paper contribution.

The next extraction hypothesis is a pose estimator conditioned directly on an existing person box, avoiding another person-detection stage within the crop. It must still expose uncertainty and verify correct instance attachment. Such a backend has NOT been run or validated here. Given the current coverage, more loss/epoch trials would not fix cases whose correction is deliberately unavailable. Any later backend or interaction-appearance experiment must retain the current controls and distinguish development from independent evaluation.

## Verification and reproduction

18 relevant tests passed. They cover wrong-subject rejection, ambiguous matches, hidden hips, missing evidence, identical proxy/pose visibility, zero-initialized bounded updates and exact missing-pose fallback, as well as existing image-fold and reviewed-target tests. All15 saved fold heads were strictly reloaded. Out-of-fold logits, decisions, capped selection, calibration temperatures/policies and aggregate metrics were recomputed exactly. Learned weights are nonzero; no-op training is not the reason for equal rankings.

Source files: `captioning/pose_evidence.py`, `extract_subject_pose_evidence.py`, `complete_pose_context_fallback.py`, `run_pose_target_cv.py`, `tests/test_pose_evidence.py`. Public aggregate/provenance files are in `audit/subject_pose_evidence_v1_result.json`, `audit/pose_target_cv_v1_result.json` and `audit/subject_pose_context_fallback_v2_result.json`. Local package dependencies are isolated in `C:/KhoaLuanTotNghiep/Code/.tmp/pose-deps`; no global environment was changed. Only a small pose model was downloaded, not another captioning backbone or COCO dataset.

Local outputs: `C:/KhoaLuanTotNghiep/Code/output/subject_pose_evidence_v1`, `C:/KhoaLuanTotNghiep/Code/output/pose_target_cv_v1` and `C:/KhoaLuanTotNghiep/Code/output/subject_pose_context_fallback_v2`. The diagnostic overlay is `subject_pose_evidence_v1/pose_diagnostic_4.png`. Raw human notes, images and case-level predictions are not published in these aggregate files.

```powershell
python extract_subject_pose_evidence.py --package "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1" --model "C:/KhoaLuanTotNghiep/Code/.tmp/pose-models/pose_landmarker_lite.task" --output "C:/KhoaLuanTotNghiep/Code/output/subject_pose_evidence_v1"
python run_pose_target_cv.py --source "C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip" --package "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1" --review "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1/human_batch1_review.json" --poses "C:/KhoaLuanTotNghiep/Code/output/subject_pose_evidence_v1" --r101-directory "C:/KhoaLuanTotNghiep/Code/output/factorized_action_target_relative_cv_v2" --output "C:/KhoaLuanTotNghiep/Code/output/pose_target_cv_v1"
python complete_pose_context_fallback.py --source "C:/KhoaLuanTotNghiep/Code/output/subject_pose_evidence_v1" --package "C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1" --model "C:/KhoaLuanTotNghiep/Code/.tmp/pose-models/pose_landmarker_lite.task" --output "C:/KhoaLuanTotNghiep/Code/output/subject_pose_context_fallback_v2"
```
