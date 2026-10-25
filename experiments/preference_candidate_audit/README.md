# TRAIN candidate feasibility before preference training

2026-10-25. User reported **50 GPU hours**; per-session time limit and distribution between accounts remain unconfirmed. This is one user-run **T4×2 inference-only notebook**, not C/T training. Root has not run any local ML or submitted a Kaggle job. Ponytail workflow: reuse the existing A architecture, cache, noun-label package and notebook bootstrap; no new model or dependency family.

## Existing-artifact check

No **verified compatible TRAIN bank** was discovered in the checked repository and local output inventory. This is not a claim that none exists on an uninspected Kaggle account. Prior `audit/scst_candidate_headroom_val1000.json` and `audit/caption_candidate_headroom_val5000.json` concern held-out/reused evaluation, so cannot become preference supervision. Earlier learned-ranker preparation and strict gain-guard artifacts concern a different generator/protocol; gain-guard explicitly uses original epoch10, not exact A. They do not establish a bank for A SHA `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.

Available ingredients are exact A, global visual cache, legacy prompt tokens, pinned MSCOCO JSON, detector cache, and the sealed TRAIN10k noun-label package already in GitHub. No request to upload a checkpoint again is needed.

## Run

Open `TRAIN_CANDIDATE_HEADROOM_2000_2GPU.ipynb`; **T4×2, Internet ON, Save & Run All**. Five inputs are prefilled:

1. `ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`
2. `ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt`
3. `ducanh2403/visual-cache`
4. `vuthetam/mscoco-2014/dataset_coco.json`
5. `ducanh2403/objectdetectionecache/objectdetectioncache.json`

All five roots use their known full `/kaggle/input/datasets/...` mount paths in the first cell. The labels archive and conservative parser policy are pinned from the existing source; no H/L, UPT or original-image cache extraction is needed.

## Frozen protocol

Hash-order a fixed2,000-image subset of locked TRAIN10k, independent of captions and errors; verify train/restval membership and disjoint held-out IDs. Copy exact old FP16 global tokens and FP32 legacy tokens, with three readback checks. Hash input assets and archived source. No model forward in preparation. Two workers each infer1,000 images without DDP collectives, training or optimizer. A architecture and full base tensor digest must match.

Each image gets raw greedy plus four independent temperature0.8/top-p0.95 samples. Maximum length30 including BOS matches A; PAD/BOS probabilities remain allowed as in original A and invalid sequences are excluded from preference pairs, not silently resampled. Require EOS, no UNK/PAD/BOS inside the caption. Save raw token IDs, invalid sequences and all candidates. No duplicates are sampled again. Seed per image/draw makes sampling independent of worker routing; batched numerical differences across hardware remain possible. On the first actual batch, greedy must exactly match existing `rollout`, sampling must repeat, and endpoint A tensor hash must stay unchanged.

First500 images are a technical/timing segment, then continue to2,000 only if the1.5× projected worker duration fits a3-hour worker wall guard. Preparation/CPU scoring are additional and reported separately from worker time; this is not a Kaggle billed-quota promise. A timing failure returns partial data, not a negative headroom conclusion. No automatic10k bank or training follows.

Score with official PTB/CIDEr using **fixed TRAIN10k reference IDF**. Blank hypotheses on unselected TRAIN rows preserve the scorer corpus; only selected2k scores are reported. No validation or test references/annotations are used to choose a pair. Caption scores under TRAIN IDF must not be compared as model validation results. Oracle gain is hypothetical candidate selection, not measured captioner improvement.

For each image: support from sealed COCO/reference union; salient objects must occur in at least2 of5 references and be supported. Use the existing longest-alias/compound/ambiguity policy. Drop images with detector confidence≥0.9 classes unsupported by the annotation/reference union; missing labels do not prove physical absence. Drop ambiguous candidates. Among valid deduplicated candidates, choose maximum CIDEr-gap pair with deterministic ties, gap≥0.10, winner unsupported set empty, salient recall≥loser, winner/loser word-length ratio in[.8,1.25]. At most one pair per image. These conservative filters may severely reduce coverage; do not relax them after seeing output.

Numeric screen:≥400/2,000 images with valid pairs and guarded-oracle mean gain≥.05 over greedy. Oracle falls back to greedy if an eligible better candidate is unavailable and respects greedy salience/length safeguards. Report class coverage and conflict exclusions; a full10k bank would later need≥2,000 pair images and≥15 represented salient classes. These are decision thresholds, not universal statistical standards.

An independent hash-selected blinded30-pair HTML with X/Y and original images is emitted, plus a separate hidden key. Human checking remains required:≥27/30 prefer the selected direction and no systematic parser/compound errors. It is a small error screen, not population label-accuracy evidence. If fewer than30 valid pairs exist, the review remains incomplete; no pass is inferred. Images load via COCO HTTPS, not generated/fetched by root.

## Return and limits

Send `preference_candidate_train2000_review.zip`; preserve the whole output folder. ZIP includes candidates, scored captions, sample IDs, per-image weak labels, chosen pairs, runtime/timings, protocol/source hashes and `review_30.html`. `numeric_summary.json` must say `CANDIDATE_AUDIT_COMPLETE`; archive creation alone is not success. Human review and novelty assessment are separate pending gates.

This prepares the agreed feasibility check only. DPO/DiCO-style self-generated preference and object-aware filtering have precedents; no new alignment method is claimed. A candidate oracle gap does not demonstrate the model can learn it, does not test relation grounding, and cannot authorize choosing a favorable validation subgroup. No C/T training notebooks or test evaluation are launched.

Local: eight metadata/packaging tests pass; one synthetic Torch sampling test is deferred to Kaggle. Real A greedy parity, GPU throughput, HDF5 staging/readback and full scoring remain user-run checks. No local torch installation or model execution.
