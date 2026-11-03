# Saved-caption comparison before further CSA training

03/11/2026. User requested loss and hallucination comparison as well as CIDEr, and an improvement plan. Prepare CPU scoring of already-generated captions; no checkpoint/model/GPU execution. Do not change or extend training yet.

## Loss boundary

CSA XE: epoch1 observedTRAIN NLL4.084960/valid3.339067; epoch9 TRAIN2.441058/valid2.446409; epoch10 TRAIN2.417765/valid2.435452. Both losses decreased while CIDEr9→10 decreased1.002933→.997179. This does not establish overfitting/convergence or imply more XE cannot help.

Old C uses SCST continuation + rolloutKL + positive concept-mass term, not the same objective as CSA XE. No matched old5000val teacher-forced NLL exists in the evaluated artifacts. Both token vocabularies have10000 entries but only**6916 word types in common**, verified against old expanded metadata and new vocabulary. Direct scalar NLL/total-loss ranking across these tokenizations is not justified. Report missing comparable old NLL honestly; within-model loss trajectories and output metrics remain useful. New common teacher-forced inference would require matched targets/UNK treatment and both checkpoints, not this CPU-only scorer.

## Fixed cohorts and inputs

- New review exact Output139 ZIP7ff1ba03…, no need to download weights. Original all5000val captions at epochs9/10, best9 preselected by CIDEr before CHAIR; do not reselect after hallucination scores.
- Historical expanded4000 includes saved A/C0/C predictions, fixed checkpoint identities and metrics (ZIP4397bbc5…). All4000 COCO IDs are contained in new5000val.
-3992reference lists agree exactly;8images have6new references versus5old. Old references are the first5 of new lists for those8. Common4000 scoring uses exactly the old5references/image (20000total), pinned by canonical COCO-ID/reference hash. Recompute all conditions with the same PTB/CIDEr corpus. This prevents mixed cohort/IDF/reference-count comparisons; historical old point metrics must replay within declared tolerances before comparisons are reported.
- Also score CSA9/10 on their original5000val/all25010references; verify their stored CIDEr values. New5000 cohort and common4000 metrics must be reported separately.
- A/C0/C oldgreedy vs CSAbeam5 remain different operational pipelines/training stages. A common evaluator compares their delivered captions, not an isolated architectural effect. Historically reused development, not an independent test; no new claim of novelty/success from this evaluation alone.

## Metrics and runtime

CIDEr, BLEU4, ROUGE-L from pycocoevalcap1.2/PTB; CHAIRs/i, raw hallucinated-caption/mention counts, object recall and mean words. CHAIR standalone source4087a26211aa2339b9a76307cb8f0321ef691d0a/cacheSHA73ce1122…; same as historical evaluator. Verify baseline metrics and reaggregate rawCHAIR counts. CHAIR is object hallucination, not a general predicate/attribute truth metric.

No new generation, LM, Torch or checkpoint loads. CPU notebook downloads only the small139review and pinned compressed baseline text bundle, not1.2GB weights. Internet/Secret required for exact139 API route; AcceleratorNone/noAddInput. Never silently fall back to latest version. Keep original per-image outputs and comparison report; failures do not publish partial comparisons as complete.

### Authentication-cell fix,03/11

First user attempt failed with `NameError: OWNER` before authentication/download/scoring. The generator reused the Dataset transport authentication cell including two irrelevant OWNER/SLUG/TITLE upload guards, but the read-only scorer intentionally has no such globals. Remove those two checks in the generator and regenerate the notebook; retain exact139 routing, nonempty Secret handling, credential isolation and token redaction. Seven tests now pass, including executing the full auth cell with mocked Secrets/Popen and **no upload globals**, wrong-version/empty-secret failures, and credential redaction. These mocks contact no API/install nothing. Scorer source pins remain7078365 unchanged. If continuing the current failed session, delete only those two guards and rerun the auth cell onward; the setup cell already created OUTPUT and should not be rerun against that same directory. Fresh session can run the corrected notebook from the start. No metric result exists from the failed attempt.

### CIDEr replay failure and diagnostic V2,03/11

User progressed through A/4000 CHAIR (rounded6.0%/3.9%/46.8%), then strict historical CIDEr replay failed. That CHAIR output is **A, not CSA**. The original scorer did not save the actual CIDEr point before raising; log is insufficient to determine size or cause of discrepancy. Do not label it rounding/library error or treat CIDEr as validated.

Verified source difference: historical4000 scorer uses original `image_id` keys/order; V1 exported only COCO IDs and sorted by those. V2 baseline bundle retains original ID map/order and restores exact old tokenizer/scorer inputs for4000; full5000 retains original CSA reference order and rank-concatenated hypothesis order. Reference-text hashes, labels, corpus sizes and tolerances unchanged. This is a replay-fidelity fix, **not proof order caused the scalar drift**; actual NLP replay still required. Also align per-image CIDEr vectors by reference IDs instead of assuming hypothesis insertion order, as the official Cider wrapper loops over gts keys.

V2 saves actual/expected/delta for every condition, all per-condition raw scores, tokenized references/hypotheses, tokenizer/CIDEr source SHA provenance and comparison against stored CSA reference tokens. Replay failure is deferred until independent CHAIR conditions are all scored, not waived. Overall status remains METRIC_REPLAY_FAILED/INCOMPLETE on any discrepancy; CIDEr bootstrap comparisons are withheld when CIDEr replay fails. A separate chair_replay_valid flag distinguishes independently checked object metrics. No unverified complete-ranking claim.

Nine text/static/mock tests pass, including a deliberately wrong CIDEr replay fixture proving all7CHAIR conditions persist, overall status remains failed, actual/expected retained and no CIDEr paired ranking emitted. No real metric engine/model/API run by agent. Existing-session recovery cell reuses reviewedZIP/installeddeps/CHAIR and creates a freshV2output; fullV2notebook remains alternative for a fresh session. No tolerance increase or changed model/decoding/training.

## Improvement decisions to discuss after metrics

1. Small decode diagnostic on the **same best9checkpoint** (greedy vs existing raw-score beam5) is useful because409/5000captions are≤2words. Raw beam scores can prefer short sequences; actual cause not yet established, and no new policy/length-normalization sweep is approved.
2. Complete the planned paper-like XE phase to15epochs only if user approves continuation/budget; no claim that5moreepochs will close the quality gap.
3. Then consider matched SCST+CE anchoring to optimize caption reward; monitor CHAIR/recall/length so CIDEr gain alone is not success. Add object penalties only with reliable supervision/control; old rewards do not automatically transfer.
4. Image-conditioned noun-bank stream stays a distinct unapproved research candidate. Fix baseline training/decoding quality before attributing benefits to it; auxiliary-head-only control required if promising.
