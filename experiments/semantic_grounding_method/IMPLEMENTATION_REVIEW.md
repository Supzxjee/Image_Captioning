# Static implementation review and disposition

2026-10-27. Astra review turn `01a17d84-2160-7573-9b8f-a975d46d12e1`; implementation thread applied the fixes below. Astra did not run code or edit this implementation. This is not a claim of neural/GPU preflight PASS.

| Finding | Applied change | Verification status |
| --- | --- | --- |
| P1: external COCOJSON could differ on the second account, corrupting reward/evaluation | Check DATA_SHA and CHAIR_SHA before worker/preflight work AND before endpoint evaluation; sealed preparation verifies all original inputs | Static inspection; user-run hashes still required |
| P2: sample/rescore and unequal-rank counts were not sufficiently checked | Add actual nonzero-shadow-head sample logprob versus full-prefix rescore; synthetic nonzero policy with firstEOS masking; two-rank unequal-token and rank-without-aux gradient fixture | Torch checks written and syntax-checked, NOT executed locally |
| P2: grad128 could be nonfinite while an inf-vs-inf comparison failed to stop | Explicit finite checks for BOTH64/128gradient vectors, then fixed norm tolerance | Static inspection; user-run numerical check required |
| P2: each rank repeatedly hashes the same multi-GB HDF5 | Parent does receipt/schema-only validation; rank0 verifies full payload once per preflight/worker phase and broadcasts success/error to peer | Static inspection; no concurrent duplicate hashing, still independent checks per phase |
| P2: bootstrap failure artifacts and source-guarded archive could lose or mask failures | Persistent `/kaggle/working` bootstrap log/error JSON; stdlib archive fallback bypasses runtime source guards, records original traceback and preserves failed output | Generated-cell AST checks; not executed against Kaggle services |

Astra's static reading found the sample/rescore policy structure, analytic contribution contraction, pi autograd, sum-before-ranking and distributed RL/KL/aux scaling consistent with the protocol. This does not validate their actual numerical execution.

Additional implementation safeguards: exact cached baseline captions before training, runtime manifest comparison across C/H/T, full-cache input IDs, second-step q/k/v gradient check, preserved completed C arm on rerun, no partial-archive-as-completed-result, and strict matched-review comparison refusing mixed preparations or incomplete arms.

Locally executed: four NumPy/metadata/saved-artifact guard tests, source/notebook AST and unexecuted-output checks. Earlier NumPy formula checks remain separate. No torch import, neural forward, model download, optimizer or GPU run was performed locally. A sandbox temporary-directory permission error during the saved-artifact tests was corrected by placing fixtures under the workspace; the rerun passed.

Independent held-out grounding is UNAVAILABLE in the current package; no claim that training attribution or CIDEr establishes semantic alignment/novelty. The package is ready for USER-RUN preflight, not a validated experimental result.
