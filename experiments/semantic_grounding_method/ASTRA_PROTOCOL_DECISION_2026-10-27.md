# Astra — final protocol design decision, C/H/T v0.3

2026-10-27. Reviewed GROUNDED_GAIN_PROTOCOL.md directly and patched the specification only. No ML code/notebook/model execution or GPU launch. Status: APPROVED PROTOCOL DESIGN with the exact v0.3 changes below; not implementation validation, novelty validation or user execution authorization.

## Latest decision: common-referent groups only

Root's redistribution counterexample is valid. The earlier statement that summing signed contributions before absolute value avoids valid-alternative penalties was too broad when each alternative has a different mask. Choose the minimal common-referent policy: retain a group only if all alternatives have the same annotation ID AND derived mask. Exclude whole multi-instance/multi-mask groups, retain same-referent lexical alternatives, and keep caption membership/2kIDs fixed. The source75/14038 count is not a promised selected-pool exclusion count. No0.95 saturation threshold, new loss or trie.

CG now uses common M AFTER summing signed c_phi,j over alternatives. H must also aggregate signed g_phi,j BEFORE ranking/ReLU. Merely summing per-alternative ranking losses would retain a penalty for pure redistribution. Define shared eta_j across alternative computations and token positions so H is the endpoint derivative of the same valid-set F, with common-prefix distributions consistent. This supersedes the v0.2 sum-of-per-alternative-ranking definition.

Required regression: for the supplied two-word full-valid-mass example, different masks/IDs trigger group exclusion; a shared mask/ID yields zero netgain, aggregated contributions, budget/direction/ranking penalties within tolerance in either alternative order. Do not claim zero per-alternative traces. This repairs the demonstrated restriction, not every possible path/cancellation phenomenon. No multi-box binding claim remains in the pilot.

## Open items resolved

1. Keep auxiliary coefficient0.1, H zero margin and196 scaling as explicitly heuristic choices. They are not gradient/strength matched. g_phi already includes pi_1(phi); sum signed g_phi FIRST, then apply a SINGLE common-mask group ranking loss. No alternative-count division or extra pi. Global eligible-image averaging still applies. No post-result retuning or claim that T-H isolates integration alone.
2. Skip the whole auxiliary group for multiple instance IDs/masks, prefix-overlap, identicaltokens/differentmask, or any empty/nonlexical/OOV/truncated/invalid target/mask. Count exclusions; keep the2kcaption training membership fixed. Freeze exact prefix serialization and group schedule. No trie/saturation threshold. Computed logeventmass>1e-6 or nonfinite is a STOP, not an online eligibility change.
3. Keep strict T-H positive difference and paired95CI lower>0 as the operational continuation gate. Failure to pass means advantage not established; it is not equivalence. Other quality/recall guards remain. Reuseddev and one-seed limitations remain; no multiple-comparison significance claim.
4. Keep preparation30m/preflight10m-per-arm/train+end45m-per-arm,6.5deviceGPUh allocated+20%overheadreserve=7.8deviceGPUh cap. Current quota is unverified and each account's allocation must be confirmed before user-approved execution. A computational stop is incomplete feasibility, not evidence of negative caption efficacy. No automatic cap/quadrature/batch/update changes or retries.

## Numerical correction

Completeness of the SUM can hide quadrature errors in individual contributions or CG. Added64vs128preflight checks on c_phi,j andCG, plus gradientCG in the new-head parameter space. Same value tolerance1e-5+1e-4abs(x128); gradientL2 tolerance1e-5+1e-4norm(grad128). Compare smoothCG, not nondifferentiable penalty boundaries or Delta gradients. Require a deterministic NONZERO shadow-head fixture on actual preflight tensors, and restore initialization/RNG hashes. ZeroWo alone is a trivial check. All arms still train with64nodes;128verification does not create a different training/scoring path.

## Claims retained

Suppression of rivals is allowed discriminative support. Within-support cancellation remains uncontrolled. Attribution is conditional on the fixed additive branch decomposition and contextualCLIPtokens. C is not region-blind because its cancellation term uses the region partition. H is an endpoint-ranking adapted control, not full HINT reproduction. Passing the pilot does not establish originality, factual correctness or causal pixel grounding.

Main falsifier: attribution/budget scores improve while actual free-generation outcomes fail to improve overA/C/H or lack useful generated-phrase grounding. Do not rescue the method with extra loss terms or selected examples after results.

No additional method brainstorm, archive reread or neural test was performed in this review. Runtime, gradients/DDP and exact numerical tolerances remain implementation preflight obligations, not already-passed checks.
