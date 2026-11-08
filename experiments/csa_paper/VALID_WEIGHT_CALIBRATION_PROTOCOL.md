# VALID reranking weight calibration — 08/11/2026

User approved this experiment after CPU189. Status: PREPARING; no new result.

Keep CONTROL180/PROMPT60, their saved five captions/image, normalized decoder/CLIP/object scores and original candidate order. No model, detector, candidate generation, training or TEST access. Original rule: D=1−C−O, C=.4, O=.3.

Before scoring, lock the shared 5000 VALID IDs by ascending SHA256 of `valid-weight-v1/42/{coco_id}` (ID breaks hash ties): first 2000 TUNE, remaining 3000 CHECK. Save manifest and SHA. VALID has already influenced development; CHECK is an internal verification split, not a fresh independent test.

Search one shared triplet for both arms: C=c/10, O=o/10, integers c,o≥0, c+o≤10 (66 rules). D is computed as `1.0-C-O`, exactly as original code. Candidate ties prefer earlier beam rank. No repetition filter.

TUNE eligibility: for EACH arm, CIDEr ≥ original, CHAIRs/i ≤ original, and object recall ≥ original (numerical tolerance 1e-12). Among eligible rules maximize mean CIDEr over the two arms; ties within 1e-12 prefer smallest L1 distance to original, then C, O ascending. Original is always eligible. Write selected weights, data/split identities and TUNE results to `selection_lock.json` BEFORE scoring CHECK. Do not try another rule after CHECK.

CHECK acceptance: each arm satisfies the same nonworsening guards, and mean CIDEr strictly improves >1e-12. Otherwise retain original. Acceptance is a point-estimate decision, not proof of statistical significance. A fresh TEST experiment would require a separate decision; this notebook never opens TEST.

Official pycocoevalcap1.2 CIDEr/PTB and pinned standalone CHAIR parser/cache. First reproduce historical fullVALID rank0/selected per-image CIDEr and CHAIR counts. Then recompute CIDEr with TUNE reference corpus alone, and separately CHECK corpus alone (independent IDF). Do not compare their absolute scores to fullVALID scores. CHAIR uses its standard fixed COCO object annotations, solely for evaluation; inference selection still uses existing detector-derived scores. CHAIR does not measure all relation/attribute errors.

Local validation: metadata and synthetic fixtures only. Official metrics/search run by user in one CPU Kaggle notebook (Internet ON; no Dataset/API token/GPU). One-hour STOP cap is not an ETA. Preserve logs/raw per-candidate scores and the review ZIP on failure. Do not edit the user's report.
