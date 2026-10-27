# Review: grounded incremental probability gain

Date: 2026-10-27. Design review, not an experiment result. Research lead: Astra; independent checks: implementation thread.

## Decision at this stage

Keep this as a candidate for a narrowly scoped constrained-adaptation experiment. Do not claim a new grounding theorem, causal visual evidence, or publication readiness. No neural implementation, notebook, inference or training has been authorized by this review.

The previous joint semantic-cosine/spatial-mass objective is not the main contribution. Exact head/name metadata contains 14,038 prefix groups: 95.86% singleton, 4.14% multiple phrase–instance pairs, and 0.534% multiple boxes. These counts precede modifier expansion and are not language-wide ambiguity rates.

## Closest-prior comparison

| Prior | Method section checked | Existing contribution | Remaining candidate distinction, not novelty clearance |
| --- | --- | --- | --- |
| [HINT, ICCV 2019](https://arxiv.org/pdf/1902.03751) | 4.2–4.3 | Channel-summed output-score gradients define proposal importance; pairwise ranking aligns network and human importance; task objective retained. Caption word-to-COCO segmentation mapping also exists. | Align incremental normalized phrase probability relative to frozen A, with signed accounting rather than just importance ordering. |
| [LRP-IFT](https://arxiv.org/pdf/2001.01037) | V.C.2, Eqs. 28–32 / Algorithm 2 | Signed relevance reweights visual context and hidden features; fine-tuning can use CE or SCST. | Direct constraints on attributed probability gain from a paired region; signed relevance plus SCST itself is inherited. |
| [CDEP, ICML 2020](https://proceedings.mlr.press/v119/rieger20a/rieger20a.pdf) | 3.2–3.3 | Decomposes logits into feature-group and remainder contributions; explanation penalty plus task objective. Other differentiable explanations can be substituted. | A specific frozen-captioner adaptation problem and probability-gain constraints. This remains an instance of explanation regularization; a new notation is not an independent contribution. |
| [Integrated Gradients, ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html) | Primary record checked here; path completeness inherited in the derivation | Attribution via a baseline path; completeness is not new. | Baseline path scales only the additive adaptation branch while keeping the autoregressive prefix fixed. |

Scope: targeted comparison with these primary sources, not an exhaustive novelty search. HINT/LRP-IFT/CDEP method passages independently read in this review; no claim to reproduce their full experimental protocols.

## Formula checked

For fixed prefix, baseline logits lA and patch corrections bj:

`p(alpha) = softmax(lA + alpha * sum_j bj)`

`cj(w) = bj(w) - dot(mean_alpha p(alpha), bj)`

The sum equals `log p_student(w) - log p_A(w)` for exact integration. This is a computational attribution conditional on the chosen branch decomposition. It does not trace all of A's evidence or prove pixel-level causality.

Synthetic checks use NumPy and 64-point Gauss–Legendre quadrature. The random finite-logit example has completeness residual 2.78e-16, agrees with 128 points, and is invariant to per-patch vocabulary-constant offsets. This is not a general numerical error bound for arbitrary learned logits.

## Confirmed limitations and counterexamples

1. **Competitor suppression:** base `[0,0]`, target word 0, support correction `[0,-1]`, rest zero. Target raw logit does not change, yet log-probability gain and support contribution are both +0.379885. This is legitimate discriminative probability support, not proof that the region recognizes the target object.
2. **Within-group cancellation is unbounded by the proposed two-group penalty:** support patches `[10,0]` and `[-9,0]`, rest zero yield contributions +3.798855 and -3.418969. Their sum is +0.379885, and all three proposed penalties are zero. Limit the cancellation claim to cancellation BETWEEN support and complement. Do not automatically add another regularizer to rescue a broader claim.
3. **Attribution depends on allocation:** identical total logits can be written as all correction in support or all in complement. The prediction is unchanged but support attribution differs. The architecture restricts allowed decompositions; completeness alone does not establish identifiable visual evidence.
4. **Contextualized tokens:** patch coordinates do not isolate local pixel information after CLIP self-attention; query hA is also image-conditioned. A successful region attribution objective could still use globally conveyed context.
5. **Reference-prefix objective is not free generation:** increased reference phrase mass can leave greedy decisions unchanged or reinforce frequent phrases. SCST/KL and independent free-caption evaluation remain necessary; repeated dev data is not a new held-out test.

## Claims that remain testable

Hypothesis: with identical additive branch, reference exposure, main caption objective and training budget, directional supervision of probability-gain attribution improves free caption quality over A and a reference-gain control, and adds value beyond adapted HINT-style ranking.

This is more concrete than attention visualization but not yet established originality. No improvement is inferred from these arithmetic checks.

## Open decisions sent to Astra

- Retain the candidate with these narrow claims, or stop because the remaining difference is too incremental?
- Which C/H/T controls are necessary in the smallest pilot? C already uses region partition through the symmetric cancellation budget and must not be called region-blind.
- Precisely define adapted HINT output quantity and differentiation boundary. Using the full contextual Z dependence versus a detached A state changes the question; silently detaching could remove the supervised dependency.
- Choose a falsification condition before implementing. Do not build a new generic audit or consume GPU to answer a question that would not change the decision.

Artifacts: `check_grounded_gain_math.py`, `GROUNDED_GAIN_MATH.json`, `audit_prefix_sets.py`, `PREFIX_SET_INVENTORY.json`. No external account execution or model downloads.

## Protocol review follow-up: valid alternatives with different boxes

The subsequent protocol check found another restriction of the original candidate. Consider two valid one-token alternatives that exhaust a two-word vocabulary, A logits `[0,0]`, patch0 correction `[1,0]`, patch1 correction `[0,0]`. Alternative0 is paired with patch0 and alternative1 with patch1. Valid-set probability stays1 throughout the path, so Delta=0 and m=0. Nevertheless paired CG=+0.23105858, CU=-0.23105858 and Lcancel=0.46211716. The loss penalizes redistribution among valid alternatives, despite no change in total valid mass. This is not a completeness error.

With the SAME mask for both alternatives, their signed contributions cancel within each patch in this example, CG=0 and the extra penalty vanishes. Astra accepted the same-instance AND identical-mask group restriction in protocol v0.3, retaining caption membership and excluding only affected auxiliary groups. No saturation threshold or new loss was added, and no multi-box binding claim remains.

There is a corresponding H-control correction: aggregate signed endpoint contributions across alternatives BEFORE ranking/ReLU. Ranking them separately still gives a nonzero penalty (38.535939 in this toy with scale196), while group-level ranking is zero. Exact values and assertions are in the updated math artifact. This verifies the arithmetic counterexample and repair, not the unimplemented group guards or neural gradients.
