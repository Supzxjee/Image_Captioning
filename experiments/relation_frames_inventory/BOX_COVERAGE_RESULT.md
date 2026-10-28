# VSR full-instance and detector coverage

28/10/2026. **DATA_COVERAGE_NO_GO for the exact initial data policy**, not a negative model result. No neural runtime, feature extraction, image download or GPU execution.

## Sources and checks

Recovered the existing complete official COCO2014 annotation archive and existing object detector cache. No download was needed. Verified archive SHA `031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009`, individual official train/val member hashes and ZIP CRCs, and detector SHA `498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c`. Streamed only metadata using existing ijson 3.4.0; kept annotations for the previously declared candidate IDs.

Both sources cover **all 3,103 candidate images**. All **4,399 candidate assertions** have exactly one non-crowd instance of each of their two distinct COCO classes. Thus the partial VCOCO-file limitation in the earlier inventory is resolved; missing GT is not the current bottleneck.

For predicted boxes, select the top12 by existing confidence, with original index breaking ties. Require the exact entity class and IoU >= .5 with its unique GT box for both entities. Coordinates are clipped to the official image dimensions. Do not use a GT bbox as the model input; GT only verifies the match. No relation truth is inferred from box geometry.

## Coverage

| Stage | Train pairs | Held-out pairs |
|---|---:|---:|
| One-to-one capacity after full GT checks | 1,115 | 116 |
| Fixed hash pool, before checking detector success | 512 | 116 |
| Both images and both entities pass detector matching | **297** | **72** |
| Required total | 256 | 64 |

Across the full record cohort, 3,358 assertions pass detector matching and 1,041 fail. Re-pairing all detector-valid records would yield capacity717/76, but that is only a diagnostic upper bound, **not a substitute selected pool**.

The total pair target is possible in the fixed pool. The per-relation requirement is not:

| Relation | Fixed-pool train pairs | Fixed-pool held-out pairs |
|---|---:|---:|
| touching | 82 | 27 |
| under | 29 | 7 |
| on | 27 | 6 |
| on top of | 36 | 6 |
| contains | 18 | 6 |
| beneath | 15 | 5 |

Only **touching** meets both >=16train and >=8held-out; the initial policy requires four such relations. Even before fixed-pool selection, the detector-valid re-pairing upper bound has only touching/under meeting8held-out, so the issue is not solely the512pair cap.

## Decision and interpretation

No 256/64 training manifest is emitted. Do not lower IoU, replace failed detections with GT boxes, convert missing assertions to negative labels, or resample a split seed until quotas pass.

This is an **initial data-design failure**, not evidence against relational representation learning. The quota8perrelation and four-relation requirement are heuristic feasibility requirements. Adequate total pairs but poor coverage can justify a separately specified pre-training protocol revision; it must not be silently reported as passing this original gate. No model score has been inspected, so a transparent revised data design remains possible.

Compact coverage results were sent to Astra for one data-design decision; root has not revised the quotas, pairing or split and has not built or launched training notebooks. Feature/RGB availability, original prompt construction for decoder probes and the32-record label audit remain unchecked.

## Reproduce and limits

`check_box_coverage.py --archive <official annotations_trainval2014.zip> --detections <pinned objectdetectioncache.json> [--ijson-path <existing package folder>]`.

The script fails on different inputs/candidates, duplicate detector or GT image identities, conflicting image/assertion labels, invalid boxes or confidence values. Synthetic checks cover IoU, clipping/degenerate boxes, pair-count inflation, label conflicts and insufficient subset capacity. Actual joins preserve the image-disjoint derived split and authoritative VSR labels.

The initial sandbox could not access the existing ijson installation; the same metadata command was rerun with approved access. No model library was imported in either attempt. Machine-readable report: `BOX_COVERAGE_RESULT.json`; detailed intermediate pair rows are Git-ignored. Pair counts may share images within a split and must not be treated as independent-image counts.
