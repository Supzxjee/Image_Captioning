# Semantic caption panel: mixed AI/human screening

2026-10-29. All32 frozen pairs/64 images retained. 175 unique ratings cover192 arm-captions. 17 unique ratings confirmed by the user;158 remain AI-only. This is not an independent human evaluation. Three caption-label records changed relative to the proposal; two unresolved target-identity cases remain uncertain after explicit chat clarification.

| Arm | Target assertion on positive /32 (RC) | On negative /32 (FR) | Y=RC-FR | Target relation mentioned /64 | Correct / wrong among mentions | Unknown target /64 |
|---|---:|---:|---:|---:|---:|---:|
| A | 15/32 (46.88%) | 1/32 (3.12%) | 43.75% | 28/64 | 26 / 2 | 5 |
| I | 13/32 (40.62%) | 1/32 (3.12%) | 37.50% | 28/64 | 24 / 4 | 1 |
| J | 13/32 (40.62%) | 1/32 (3.12%) | 37.50% | 28/64 | 25 / 3 | 2 |

J expresses the true target assertion on fewer positive images than A (13 versus15), without reducing definite false assertions (1 each). Definite target-relation coverage is equal (28/64), while correct/wrong counts do not favor J (25/3 versus26/2). This is not a semantic gain over A.

Unknown q claims: A1 positive+1 negative; I0+0; J0+1. Conservative Y bounds:
- A: [40.62%, 46.88%].
- I: [37.50%, 37.50%].
- J: [34.38%, 37.50%].

## Paired effects, percentage points

| Contrast | Metric | Difference | Pair bootstrap95% interval |
|---|---|---:|---|
| J_minus_A | RC | -6.25 | [-21.88, +6.25] |
| J_minus_A | FR | +0.00 | [+0.00, +0.00] |
| J_minus_A | coverage | +0.00 | [-9.38, +9.38] |
| J_minus_A | correct_per_image | -1.56 | [-10.94, +7.81] |
| J_minus_A | wrong_per_image | +1.56 | [+0.00, +4.69] |
| J_minus_A | Y | -6.25 | [-21.88, +6.25] |
| J_minus_I | RC | +0.00 | [-12.50, +12.50] |
| J_minus_I | FR | +0.00 | [+0.00, +0.00] |
| J_minus_I | coverage | +0.00 | [-9.38, +9.38] |
| J_minus_I | correct_per_image | +1.56 | [-7.81, +10.94] |
| J_minus_I | wrong_per_image | -1.56 | [-6.25, +3.12] |
| J_minus_I | Y | +0.00 | [-12.50, +12.50] |

Intervals use10000 stratified whole-pair bootstrap draws, seed20261029. They are conditional on the mixed annotations and do not establish annotation accuracy or population significance. Unknown claims are treated as not-definite-yes in point estimates, with separate bounds. All strata and denominators are retained in the JSON.

## Decision

Keep A. Consistent with Astra's post-result rule, stop the current I/J configuration rather than automatically running seed2, full training or J_bypass. This screening result does not rule out the general shared-readout idea; it does not justify more GPU on this recipe. A different experiment needs a separately stated hypothesis.

## Human clarification

#6X and#49X retain uncertain target identity at the user's explicit request. The original uploaded judgments correct/wrong are preserved in the input cache; for analysis their correctness remains unresolved, not definite. No user file was overwritten.

## Limits

- 158 of175 unique ratings remain AI-only; 17 human confirmations are targeted, not a random agreement sample.
- Source VSR labels define positive/negative strata; pair labels were not re-audited in this caption panel.
- 64 images/32 reused pairs, one training seed. No independent test or broad causal/novelty claim.
- RC/FR measure target assertion expression, not full-caption accuracy; wrong nouns/other relations can fall outside the target pair.
