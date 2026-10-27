# Grounded probability gain: protocol C/H/T v0.3

2026-10-27. Protocol design reviewed by Astra, with the decisions in `ASTRA_PROTOCOL_DECISION_2026-10-27.md`. The v0.3 common-referent guard and group-level H ranking supersede the v0.2 multi-box/per-alternative-ranking policy. NOT a runnable notebook or authorization to launch. Supersedes the earlier joint cosine/mass pilot as the primary candidate. Novelty and caption improvement remain unestablished.

## 1. Question and claims

Does directional supervision of the source of incremental phrase probability gain improve free captions over frozen A, the same adaptation objective without that direction, and endpoint-importance ranking?

Claim scope: branch-conditional computational attribution of discriminative probability gain. No pixel causality, local recognition certificate, relation understanding, or proof that all caption errors are fixed. Suppression of competing words is permitted. Cancellation is controlled only BETWEEN support and complement, not inside either group.

## 2. Inputs, splits and targets

- Frozen A checkpoint SHA256: `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`. Exact original prompt, tokenizer, vocabulary10000, masks, decoding and CLIP preprocessing retained.
- Original dataset SHA: `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`; original prompt cache SHA: `d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d`.
- Training source: sealed noun/head/name labelbank v1.1, receipt SHA `bde873ba77124333488b263a052419dc80e412c1e089178cdf423d2836e1b146`; verify all six payloads. 4,495 TRAIN images, 5,602 regions, 17,904 spans. These are weak joins, not manual grounding gold.
- Select 2,000 images from this source intersected with locked TRAIN10k, ordered by SHA256 of UTF-8 `grounded-gain-v1:{coco_id}`; ascending integer ID breaks hash ties. Freeze IDs before eligibility handling. No silent replacement of skipped examples or filtering by model scores.
- Caption objective uses all five COCO references for each selected image and the SAME pinned full-training reward corpus/IDF as A; do not recompute IDF on the 2k subset.
- Exact A token-ID prefix before the first head/name token defines a group. Deduplicate identical target token sequences only when they identify the same instance/mask. Identical token sequences with different masks are ambiguous: skip the entire group for auxiliary supervision.
- COMMON-REFERENT GUARD: every alternative in an auxiliary group must have the SAME COCO annotation ID AND the SAME deterministically derived mask. If a group contains more than one instance ID or mask, skip the whole auxiliary group, even if different boxes happen to share an identical grid footprint. Retain multiple lexical alternatives referring to the same instance; do not claim that these are all exact synonyms. The source inventory counted75multi-box groups/14038groups (0.534%,74images), not the exclusion count for the selected2k after other guards. Report actual selected-pool counts. Caption membership is unchanged. No saturation threshold such as0.95 is introduced.
- Retain all disjoint observed phrase alternatives with their own boxes. If any alternative token sequence is a strict prefix of another, skip that entire auxiliary group. Do not sum overlapping prefix-event probabilities; no trie/frontier approximation in this pilot. This exclusion must be counted and is not a false label.
- If any retained alternative has an invalid/truncated/OOV/empty target or malformed mask, skip the ENTIRE auxiliary group by the sealed policy, without changing caption training membership. Targets must contain lexical A tokens only (no BOS/EOS/PAD/UNK); the prefix must be a valid A prefix before EOS/PAD. Masks must be finite, within [0,1], shape196, and have nonzero support after the exact coordinate transform. Do not repair malformed masks or discard a difficult alternative during optimization. Report exclusions by reason. No modifier expansion, parser, text encoder, semantic negative, action-zero negative, or new manual labeling.
- In each epoch, deterministically order groups per image by SHA256 of UTF-8 `grounded-gain-group-v1:{coco_id}:{prefix_ids}`, where `prefix_ids` is the comma-joined decimal integer sequence with no whitespace. Select group `epoch modulo group_count`, at most one group/image/update; include all its disjoint alternatives. Report selected and unseen group counts. Images with no eligible group still receive the caption objective. Freeze the resulting group schedule for C/H/T.
- Auxiliary loss: mean over eligible images in the GLOBAL batch, not rank-local means. If none are eligible, auxiliary losses are zero; retain DDP graph consistency. No duplicate supervision by repeated references.
- Evaluation: the existing locked development1000, disjoint from selected TRAIN IDs. It is repeatedly used, exploratory development, not fresh confirmation. Atlas20 is excluded from tuning, selection and numerical gates.

## 3. Shared architecture and initialization

All arms have 262,144 trainable weights; A and its vocabulary head are frozen/eval. Save one initialization file and load it in C/H/T.

`Z = ReLU(A.vis_projection(raw_CLIP[:,1:]))`, shape `[B,196,512]`, in A eval preprocessing. Use hA immediately before A's vocabulary head, already image-conditioned.

`q=LN0(hA)Wq; k=LN0(Z)Wk; v=normalize(LN0(Z)Wv, eps=1e-6)`.

Wq/Wk/Wv:512x128, Wo:128x512, all bias-free; LN0 has no affine parameters. Initialize Wq/k/v using one seeded Xavier-uniform draw and Wo=0. No dropout, gamma, topK, learned gate, additional semantic head or norm projection.

`a=softmax(q k.T / sqrt(128))`.

`b_tj(w)=a_tj * [W_vocab_A(v_j Wo)]_w`.

`logits_student=logits_A + sum_j b_tj`.

Do not add the frozen vocabulary bias twice. hA/logitsA are detached per prefix; attention/new head remain trainable. Zero Wo must reproduce A on identical shapes/runtime before training. It does not guarantee preservation after adaptation.

## 4. Attribution and numerical policy

For fixed reference prefix and alternative phi, scale only the new branch by alpha in [0,1]; hold the prefix, hA and computed attention fixed ALONG that path. Training differentiation still propagates through all new-head parameters, including attention.

`F(alpha)=log sum_phi P_alpha(phi | prefix)` for the disjoint lexical phrase-prefix events; use stable sequence log-probabilities and logsumexp. Do not append BOS/EOS/PAD/UNK or an artificial end-of-phrase token. Such tokens invalidate a semantic target under section2. First-EOS handling belongs to the separate main caption objective.

`Delta=F(1)-F(0)`.

`c_phi,j = integral pi_alpha(phi) * sum_t [b_phi,t,j(y_t) - dot(p_phi,t,alpha,b_phi,t,j)] d alpha`, where pi is the normalized probability over the retained alternatives.

Use 64-point Gauss–Legendre quadrature, float32 outside autocast for logits/probability/loss arithmetic; reductions for the numerical completeness check in float64. Keep gradients through pi and probabilities during optimization. No detached quadrature weights/probabilities except the fixed numerical node weights themselves, and no straight-through approximation.

After the common-referent guard, all alternatives share one mask M. FIRST sum signed contributions over alternatives: `c_j=sum_phi c_phi,j`. THEN calculate `CG=sum_j M_j*c_j`, with fractional box/cell overlap from the exact CLIP coordinate transform. `CU=Delta-CG`. No thresholded union box, separate-mask assignment, per-alternative absolute value or per-alternative cancellation/ranking penalty. Per-alternative components may be retained as trace, not separate relevance objectives.

This eliminates the demonstrated artificial cross-mask cancellation when probability is redistributed among valid aliases of the same referent and the total event mass is identically1. It does not establish universal invariance to every coordinated redistribution or every cancellation path. K still controls only the two computational groups. No multi-box identity/binding or multi-intent contribution claim is made for this pilot.

Avoid materializing `[196,10000]` corrections: use frozen vocabulary matrix products to contract target/expected logit directions into hidden dimensions. Chunk path nodes deterministically if needed, without dropping terms or changing normalization.

Completeness check: `abs(sum(c)-Delta) <= 1e-5 + 1e-4*abs(Delta)` for each auxiliary group. Pin before GPU use. Check every group in preflight and the first ten training updates, then every25updates on the current auxiliary batch.

Compare64vs128nodes in preflight on the SAME finite/nonzero synthetic corrections and actual preflight input tensors. Check individual `c_phi,j` and `CG` (not only their sum) with `abs(x64-x128) <= 1e-5 + 1e-4*abs(x128)`. Also compare the gradients of the smooth quantity `CG` over all new-head parameters: `norm(grad64-grad128,2) <= 1e-5 + 1e-4*norm(grad128,2)`. Do not compare only gradients of Delta, which does not depend on quadrature, or gradients at nonsmooth hinge/absolute-value boundaries. Use at least one deterministic nonzero shadow-head fixture on actual inputs; a zero-Wo fixture alone is insufficient. The fixture is not an optimizer update: preserve/restore the shared training initialization and RNG states, record fixture parameters/logit ranges, and verify the original initialization/state hashes afterwards.

If a check fails, stop and archive; no silent clipping, tolerance inflation or higher-node integration only on the reporting path. All train arms keep64nodes;128 is preflight verification only. FP32/neural feasibility is UNVERIFIED; the current NumPy test is not a substitute.

## 5. Losses and matched controls

Main shared objective:

`Lmain = L_SCST + 0.1 * KL(p_A || p_student)`.

SCST samples actual corrected student policy at temperature1, with current student greedy baseline. Sequence-sum logprob includes first EOS and masks later PAD/EOS positions. Reward is the existing pinned `CIDEr-D + 0.5*object_count_v2`; copy and verify exact evaluator, reward options, tokenizer and full training IDF hashes from the audited A protocol before packaging. If those hashes/options cannot be resolved, stop rather than reconstructing a similar reward. KL is a GLOBAL valid-token mean on the SAME detached sampled prefixes, with A distribution detached.

For each eligible group:

`m=min(0.05, 0.1*max(-F(0),0))`. For every evaluated alpha including0/1 and quadrature nodes, require finite `F(alpha) <= 1e-6`; this fixed log-mass tolerance only permits FP32 rounding. Any value above it is an invalid event/probability computation: STOP, not an adaptive group exclusion. Do not clamp the probability or F itself; only the tiny nonnegative headroom correction in m is allowed. Masks/eligibility are data-only and locked before training.

`Lgain=relu(m-Delta)`.

`Lcancel=relu(abs(CG)+abs(CU)-2*relu(Delta))`.

`Ldirection=relu(0.8*relu(Delta)-CG)`.

| Arm | Fixed objective |
| --- | --- |
| C | Lmain +0.1*(Lgain+Lcancel) |
| H | Lmain +0.1*(Lgain+Lcancel+Lrank) |
| T | Lmain +0.1*(Lgain+Lcancel+Ldirection) |

These coefficients are declared design constants, not empirically tuned or gradient-matched. All arms see the same reference groups. C already uses the region partition through symmetric cancellation and must not be called region-blind. No CE warm-up or sweep.

### Endpoint-importance H control (adapted HINT-style, NOT full HINT reproduction)

Introduce scalar eta_j AFTER attention multiplying patch j's logit correction, SHARED across all alternative computations and token positions for this image/group. A repeated common prefix must use the same eta and the same conditional distribution, rather than separate interventions per alternative. At eta=1, compute the endpoint derivative of the SAME normalized set-mass F:

`g_phi,j = pi_1(phi) * sum_t [b_phi,t,j(y_t)-dot(p_phi,t,1,b_phi,t,j)]`; `g_j=sum_phi g_phi,j`.

This derivative has an analytic form; retain its parameter gradients, do not detach g. Do not take gradients with respect to projected Z or silently detach the attention parameters. No gradients through A's query/base state are required for either H or T.

Using the group's common mask, define `d_jk=relu(M_j-M_k)`. If `sum d=0`, set the group's ranking term to0 and count it. Otherwise:

`Lrank = sum_jk d_jk * relu(196*(g_k-g_j)) / sum_jk d_jk`.

Signed alternatives MUST be aggregated before ranking/ReLU, since the objective being explained is F for the whole valid set. Ranking each alternative and summing would still penalize pure redistribution even after the common-mask guard. pi is included exactly once in g_phi,j; no additional pi or alternative-count averaging. Average group Lrank over eligible images in the global batch like other auxiliary terms. The fixed196 factor converts per-patch derivative scale to a patch-count scale; it is a design choice, not a proven calibration to Ldirection. Zero ranking margin. No multi-instance union mask. T-vs-H compares two constraint packages, not integration alone or strength-matched regularization. No post-result rescaling of H.

At Wo=0, branch scores and most directional/ranking penalties may be zero; common Lgain supplies the initial update. Verify first-step Wo gradient, subsequent q/k/v gradients and optimizer/DDP state before launch; do not claim every parameter has a nonzero first-step gradient.

## 6. Updates, execution plan and resource bounds

- Seed42, same initial tensors and image/group order. Two epochs of2,000images; global batch8, T4x2,4images/rank; exactly500updates/arm. No gradient accumulation or mixed per-arm batch sizes.
- AdamW lr1e-4,weight_decay.01,betas(.9,.999),eps1e-8; constantLR,globalgradclip1. FP32 new head and auxiliary arithmetic. Save optimizer/source/runtime receipts and end checkpoint; only end checkpoint evaluated. No dev checkpoint selection.
- Two user-run notebooks: notebook1 runs C then H sequentially; notebook2 runs T. Reuse the SAME verified preparation dataset/initialization; package it once. Different accounts must resolve identical payload hashes, not just similar dataset names.
- Budget CAP, not ETA: common preparation<=30wallminutes; preflight<=10minutes/arm; train+end-evaluation<=45minutes/arm. Up to3h15 allocated two-GPU walltime across jobs, conservatively6.5T4-GPUhours if preparation also uses2GPUs. Reserve20% =>7.8GPUhours. Current account quotas and maximum session lengths are NOT confirmed; no launch until verified.
- Keep these caps fixed for this protocol. Authorized preflight may measure throughput; it does not authorize increasing a cap. Reserve20% is for execution/setup overhead, not extra updates, retries, sweeps or relaxed assertions. Record each account's quota units and its own C/H or T allocation before any user-approved GPU run; do not assume two accounts' balances are interchangeable.
- First10updates are part of the500, not a discarded warm-up. Measure synchronized time and peakmemory; forecast remaining updates plus evaluation conservatively. If the45minute arm cap cannot be met, stop and preserve the partial artifact. Do not silently reduce500updates or change quadrature/batch/eval IDs. Staging can be reused after verifying its receipt; never restart expensive preparation needlessly.
- Log stage/progress/time every25updates; always archive failure traceback and incomplete status. A partial checkpoint/archive is not a completed experiment.
- A budget/preflight failure is `INCOMPLETE_COMPUTATIONAL_NO_GO`, not a completed negative efficacy result. Only a fully completed matched endpoint comparison may be labeled `COMPLETED_QUALITY_NO_GO` or pass its research gate. Preserve any completed arms without imputing missing endpoints for a stopped arm.

## 7. Preflight and evaluation

Before training: source hashes and train/dev disjointness; exact tokenizer/position/mask binding; disjoint-event mass<=1; paired masks; actual262144parameter count; frozenA/state parity; zeroWo teacherforced logits AND greedy parity; finite/nonzero synthetic completeness;64vs128integration; gradient flow; two-rank loss normalization and identical optimizer update to the corresponding single global batch within declared floating tolerance. Preserve all assertions, archive on failure.

Mandatory synthetic regressions include the full-valid-mass two-word redistribution case: different referents/masks must fail the data guard; a shared referent/mask must yield Delta, aggregated c_j/g_j, CG/CU, Lgain/Lcancel/Ldirection/Lrank all zero within tolerance. Test both orders of the lexical alternatives. Keep per-alternative nonzero trace values to confirm that the implementation cancels signed contributions BEFORE its nonlinear penalties. These regressions are required implementation checks, not already-passed neural tests.

Endpoint evaluation: fixed greedy free captions on all1,000development IDs using pinned A decoding/evaluator/IDF. A predictions must be byte/caption-ID matched to the existing baseline or recomputed with verified parity; baseline CIDEr expected1.205512599 under that exact dev evaluator, not val5000IDF. No reranking, prompt changes or test evaluation.

Report CIDEr, CHAIRs, CHAIRi, existing object recall and mean length. Paired image bootstrap10,000resamples,seed42, same sampled IDs for T-A,T-C,T-H. Record95%percentile CIs and point differences. These describe image variability conditional on one seed and reused dev, not training-seed variability.

Report auxiliary components, completeness residuals, CG/CU, support and exclusion counts on training and held-out eligible reference groups separately. Apply the SAME conservative weak-label policy to held-out COCO annotations if those inputs are available and verified; report coverage rather than treating unmapped mentions as false. On actual generated noun/name mentions with unambiguous instance mappings, report same-prefix computational attribution separately from reference-prefix scores. These weak-label computational measures are NOT independent factuality gold or a relation test. If inputs are missing, mark this evaluation unavailable and withhold grounding claims; do not invent an automatic factuality certificate.

Predeclared exploratory continuation gate: T-A>=+0.01CIDEr;95%pairedCI lower(T-A)>0 and lower(T-C)>0; T CHAIRs/i no higher than A; objectrecall>=A-.005. For the proposed-accounting advantage, also require positive T-H pointdifference and95%pairedCI lower(T-H)>0. This is a conservative operational gate, not a power calculation or multiple-comparison-corrected paper significance claim. A CI crossing0 means advantage not established, not equality.

If any quality guard fails, or attribution improves without free-generation benefit/advantage over H, do not scale/sweep/addseed to rescue this recipe. Report the failure and keepA. Failure does not prove a universal captioning/alignment ceiling. Passing permits a separately planned confirmation/ablation stage, not immediate novelty/publication claims.

## 8. Astra resolution for the four open items

1. Accept auxiliary0.1 coefficients, zero margin and the fixed196 scaling as heuristic adapted-control choices. Under the v0.3 common-mask guard, FIRST sum signed g_phi,j over alternatives, THEN apply group-level ranking; no per-alternative ReLU/ranking, additional pi or alternative-count normalization. No claim of matched loss strength or pure integration effect.
2. Accept whole-group exclusion for multiple instance IDs/masks, overlapping events/ambiguous masks/invalid targets, with counts and unchanged2kcaption membership. Retain same-referent lexical alternatives. No saturation threshold or trie in this pilot. Stop on invalid computed event mass rather than dynamically discarding it.
3. Accept the strict positive T-H point difference and CI-lower>0 continuation gate. It is an operational research gate, not a power estimate, corrected significance claim or equivalence test. Do not add runs to rescue an inconclusive comparison.
4. Accept the stated caps as caps, NOT an ETA. Input/reward hashes and per-account quota must be resolved before any user-approved GPU run. Keep caps and numerical checks fixed; fail computational feasibility if preflight/throughput cannot meet them. Research-lead protocol approval does not authorize spending GPU.

No runnable artifact has been produced from this draft. Research-lead approval is not a user authorization to spend GPU.
