# Object evidence directly into noun logits — proposal after research review

2026-10-23. **Candidate agreed by Astra after three exchanges; not trained and not a demonstrated improvement.**
No training notebook has been created. User review and the exact A checkpoint path/hash
are prerequisites for implementation. Local work here is metadata/lexical auditing only.

## Why this candidate

R changed raw attention-path mass without establishing a caption gain over A or B.
The new candidate gives recognized object evidence a direct, bounded path into the
decoder's vocabulary scores. It tests a narrower question than relation grounding:
can bbox supervision make image-specific object presence evidence useful to a strong,
frozen captioner? Increasing attention, matcher margins or corruption sensitivity
alone will not count as success.

Rejected alternatives for this pilot: absent-noun unlikelihood (weak negatives and
overlap with object reward); a separate relation/content planner (larger pipeline change,
insufficient reliable relation labels). No additional 128-image corruption probe.

Object-guided word generation has prior art: [Neural Baby Talk, CVPR2018](https://arxiv.org/abs/1803.09845).
[MARINE, ICML2025](https://arxiv.org/html/2402.08680v2) uses external object information,
guidance prompts and combined generation logits. This proposal instead learns a small
patch classifier with TRAIN supervision and adds bounded class evidence only to an
explicit vocabulary subset. These differences do not establish novelty or efficacy.

## Frozen baseline and module

Use A endpoint bytes SHA256:
`630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.
Do not silently use original5f8… or reconstruct A and call it the same checkpoint.
The user reports A is available in Kaggle output/dataset; exact path and byte verification
remain pending. None of seven top-level Downloads checkpoint candidates matched A.

Freeze the complete A captioner, including QFormer, legacy branch and decoder.
Use existing CLIP cache, no detector/CLIP re-encoding:

`196×768 patches → nonparametric normalization → Linear768→128 → GELU → Linear128→80`

The head adds108,752parameters and produces patch/class logits Z. CLS is excluded only
from this new branch; A's memory and preprocessing remain unchanged.

`g_c(I) = logmeanexp_p Z_pc` (temperature1).

`d_c(I) = clip(g_c(I) − logit(pi_c), −2, 2)`.

At decoding step t, add `.5*d_c(I)` to an eligible word's A logit before full-vocabulary
softmax; all other logits stay unchanged. Use these corrected logits in teacher CE,
sampled SCST teacher scoring and actual generation. No forced noun insertion.

The first version has no learned prefix/query gate: it is image-specific noun bias,
not relation grounding or within-class semantic disambiguation.
Last head layer starts with zero weights and bias logit(pi), making initial correction
zero mathematically; verify baseline greedy parity and same-batch numerical tolerance.
Do not compare logits from different GEMM batch shapes as a parity gate.

## Vocabulary and TRAIN labels

Freeze an eligible single-word alias mapping using TRAIN/source policy, excluding
ambiguous words. Preserve raw tokenizer IDs and punctuation variants.
Apply a finite longest-match compound guard using only generated prefix and candidate:
`hot + dog` must not receive dog-class evidence; `playful + dog` may.
Do not use future reference tokens or GT POS at inference.
This guard reduces known collisions; it is not a perfect semantic parser.

Presence labels for all80COCO classes on all train10k images:

* Positive: COCO annotation OR a trusted mention in any of five TRAIN references.
* Masked: no annotation positive and a possible but ambiguous caption mention.
* Weak negative: remaining classes. Annotation positives override ambiguity masks.

Use existing trusted alias/ambiguity rules, compounds first; no LLM/parser relabeling.
Small/crowd annotation positives remain presence positives even if ineligible for region loss.
Caption-only positives are not put into the region-negative set.

`pi_c = (n_positive_c + .5) / (n_observed_c + 1)`; observed excludes masked labels.
Count each image/class once, not each caption/span. Compute from train10k only.
The prior subtraction is a fixed score reference, not a calibrated likelihood ratio.

## Loss and exactly two trained arms

`L = caption_CE + scst_weight*SCST(CIDEr-D + .5*object_count_v2)
     + .1*presence_BCE + .1*region_loss`.

Captioner A stays frozen; only head parameters update. Presence BCE is plain unweighted
mean over observed image/class pairs, without per-image positive/negative rebalancing.
SCST weight0 for first200updates, then.05. CE uses all five references.

Take the existing TRAIN grounding manifest and deduplicate region supervision by
`(image_id, annotation_id, class_id)`, without another class quota or vocabulary filter.
Global-batch region loss averages over actual regions, with DDP sum/count correction;
zero-region ranks contribute zero sums and both-zero batches produce zero region loss.
The25%person cap on spans is not reapplied or assumed after deduplication.

For region r of class c, negative set N contains only observed weak-negative classes
for that image. Other present or masked classes are not region negatives:

`L_r = softplus(-s_r,c) + mean_{a in N} softplus(s_r,a)`;
if N is empty, use positive term only.

* **G:** s_r is logmeanexp of all196patches.
* **S:** s_r is fractional-bbox-overlap weighted logmeanexp, normalized by overlap sum.

G/S have identical architecture, parameters, initial state, region events, positives,
negatives, seed, batches and loss weights. Only region pooling differs.
No new token memory and no GT boxes at inference.

## Pilot and fixed interpretation

Two notebooks, each T4×2. Seed42; global batch8;2500updates/two train10k epochs;
AdamW head LR1e-4; checkpoint endpoint only; no coefficient sweep.
Keep inherited weight decay.01 and clip1 unless explicitly revised before implementation.
Use identical sampler schedule and actual-order logging in both arms.
No runtime benchmark has been performed; do not promise a GPU-hour estimate.

Primary evaluation: all Karpathy val5000, raw greedy, with A on the same5000images.
Old dev1000 is secondary. Train10k/val5000 intersection is empty.
Val5000 has been evaluated before: call this exploratory reused validation, not virgin
holdout. Do not use test5000. Do not select images based on current noun coverage or errors.

Numerical gates, locked before results:

1. S−A CIDEr≥.01 and paired image-bootstrap95% lower>0.
2. S−G95% lower>0 for any claim of bbox supervision benefit.
3. CHAIRs/i≤A, Recall≥A−.005; report noun count, caption length and changes outside
   the eligible vocabulary to detect evasive/incomplete descriptions.
4. If S passes1–3, S must also beat both inference controls below with paired95% lower>0.

No further trained arm: TRAIN-mean uses the mean **post-clip d** over train10k;
shuffled evidence uses a prelocked seed/derangement of evidence vectors while keeping
each image and all A inputs unchanged. Apply the same compound guard in all conditions.
Turning correction off must exactly recover A's captions because A is frozen.

If only G beats A, retain any practical gain with a presence-only claim, not bbox grounding.
If neither beats A, stop this configuration; no automatic prefix gate/extra-seed/fulltrain
extension to rescue it. If gates pass, next is seed confirmation and small blind caption
review, not an automatic full-scale run. Image bootstrap does not measure seed variance.

## Feasibility already checked locally

Pre-policy alias candidates cover16,679/17,904retained spans(93.16%), including
12,203/13,428nonperson spans(90.88%);50nonperson classes have≥50covered spans.
The frozen pre-implementation feasibility gates (≥50%nonperson;≥15classes with50)
pass. On saved A dev1000 captions, guard-aware candidates appear in852captions,
1,317mentions. This is coverage, not a correctness/performance estimate or final whitelist.

Dedup gives5,602regions on4,495images,71classes;person1,257regions(22.44%).
All classes remain in supervision even if some have no eligible output word.

The finite guard examples and metadata split checks pass in `audit_feasibility.py`.
`FEASIBILITY.json` stores input hashes and counts. Exact Kaggle A bytes, visual cache
coverage, final vocabulary policy and GPU integration remain unverified.
No model training/inference or training notebook has been executed or created locally.
