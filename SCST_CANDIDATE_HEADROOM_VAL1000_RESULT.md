# Candidate diagnosis after SCST: improve selection before another generator run

No model training, inference or Kaggle notebook was launched. Official PTB/CIDEr was computed locally for all5 candidates of each of3 pools on the same1000 validation images; each scoring pass retains the same reference corpus/IDF. Raw and fixed scores reproduce the reviewed metrics within1e-10. Archive CRC and identities/counts pass.

## Reference-aware ceiling versus actual selection

| Candidate pool | Raw | Fixed selector | CLIP-only selector | Reference-aware oracle | Oracle minus fixed |
|---|---:|---:|---:|---:|---:|
| Original QFormer | 1.218230 | 1.275819 | 1.243849 | 1.555513 | .279694 |
| SCST+CE | 1.230521 | 1.260458 | 1.232715 | 1.568405 | .307947 |
| SCST+CE count reward0.5 | 1.233177 | 1.274298 | 1.247862 | 1.571073 | .296774 |

The oracle chooses the highest reference-CIDEr candidate separately for each image. It sees the answers and is NOT deployable, a trained ranker, or evidence of an achieved improvement. Its gap cannot be assumed learnable. CLIP-only uses the existing image-caption CLIPScore and deterministic first-index tie handling; it is descriptive, not a newly tuned policy.

For count0.5,720/1000 images have a candidate with strictly higher CIDEr than the fixed choice;550 have at least+.1 per-image gain. Mean distinct PTB-normalized captions4.574/5; no pool has all5 normalized captions identical. There are427 images with different explicit object-class sets among candidates. Oracle and fixed choices differ in object-class sets on181 images. Thus the CIDEr opportunity is not confined to object-set changes: lexical realization and other content differences matter too. These are lexical diagnostics, not ground-truth object/relation correctness labels.

The trained pools have slightly higher oracle CIDEr than the original pool on this subset, yet their fixed selections do not beat original fixed. This supports prioritizing selection as an actionable bottleneck, without ruling out generator improvements. Simply using CLIP alone reduces CIDEr in every pool; no evidence supports replacing the fixed selector with CLIP-only.

## Avoid repeating the earlier unsuccessful pilot

This is not the first headroom diagnosis. `CAPTION_CANDIDATE_HEADROOM_RESULT.md` already measured baseline beam5 oracle1.513905 versus fixed1.210723 on5000 validation images. A later regional learned ranker reduced hallucination16→13 on512 images but lost CIDEr1.263698→1.251104. Do not mix those subsets with the current1000-image metrics, or present oracle headroom as a new research contribution.

The proposed next step is a **gain-aware selective selector**, using the existing `CaptionGainRanker`/gain-guard design as the starting point. It should compare each alternative against the fixed choice, learn both positive and negative CIDEr changes from train-only candidate pools, and keep the fixed caption when predicted gain is uncertain. The regional evidence branch must be compared against decoder/CLIP/OCC-score-only and global branches; otherwise an improvement cannot be attributed to visual/relational alignment. Existing gain-guard code has not been verified on the new SCST candidate pools during this audit.

Before preparing another training notebook, inspect whether the earlier full candidate/feature training output remains available. Review ZIPs lack the feature cache, and the current1000-image oracle labels must not be recycled as training labels. If regeneration is necessary, use the same training-image cohorts and a locked generator for comparable train/tune/validation candidates. Train-only calibration and a separate held-out evaluation are required; do not fit or tune a ranker on these1000 validation references. No claim of scientific novelty is established merely by adding abstention or reranking.

Decision: defer the extra SCST seed and further reward-coefficient tuning; retain checkpoints and assess the existing gain-guard implementation/cache before proposing a concrete selector experiment. No new learned selector or GPU notebook is delivered by this diagnostic step.

## Artifacts and limits

Source `qformer_scst_ce_object_count_v2_w05_train20k_val1000_review.zip`, SHA256 `8a8b5005e0a27f7d0a35067634840b2c1c3e595d6e4ac9379bd6af73e5d510c9`. Summary: `audit/scst_candidate_headroom_val1000.json`. Detailed per-candidate matrices and reference-aware cases remain local at `C:/KhoaLuanTotNghiep/Code/output/scst_candidate_headroom_val1000`; they are diagnostics, not training data.

No CHAIR scores exist for every candidate in this bundle, so the oracle cannot be claimed to reduce hallucination. CIDEr overlap does not establish visual truth or correct instance relations. This is reused validation from one seed, not independent confirmation or test evaluation.
