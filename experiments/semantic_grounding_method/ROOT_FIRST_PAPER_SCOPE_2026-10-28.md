# Scope for a first domestic university-journal paper

28/10/2026, Asia/Bangkok. User clarified the intended output: a first paper for a university journal in Vietnam, not an assumed Q1/top-conference submission. Exact venue is still unknown. This changes the intended size of the contribution, not standards for evidence, citation or honest reporting. No acceptance or deadline outcome is promised.

## Root recommendation sent to Astra

Keep one small, interpretable improvement to caption evidence readout. Inherit ordered subject/object/union features and direct decoder supervision with attribution. Prefer a closed-form coupled reader over a new iterative OT solver for the first pilot. This reduces implementation/numerical risk and does not require an unsupported claim of a novel optimization principle.

Use at most two trained arms and the fixed A baseline. Both trained arms must share input features, parameters, initialization, data/order, updated decoder layers, losses and update budget. The intended comparison is independent versus coupled reading of entity and pair evidence. Differences in prior temperature/softening must not become an unreported second intervention. Astra is choosing the exact control and protocol; this note is not a run authorization.

The defensible question is: does this specific coupling improve caption quality/semantic faithfulness relative to matched readout? It is not: have we invented attention/OT, guaranteed referent binding, or proved physical causal grounding? New role-prior errors cannot automatically be called measured errors of the original A, which has no such reader.

## Evidence needed to support the limited claim

- If proposed > matched control but proposed < A, the stated goal of improving the strong caption baseline is not met.
- If both trained variants > A but proposed is indistinguishable from matched control, benefit cannot be assigned specifically to coupling.
- If scores on teacher-forced assertions improve but free captions do not, language transfer is not established.
- If apparent semantic precision improves by omitting relation descriptions, report coverage; do not present omission as better grounding.
- Keep full paired results, uncertain labels, cost and failure cases. Do not change original NO_GO or describe reused development/heldout sets as untouched confirmation. A further seed does not remove adaptive reuse of evaluation data.

One modest improvement plus fair ablation and bounded conclusions is the target. Existing mechanisms are legitimate foundations, not automatic grounds to discard every application-specific improvement. Conversely, changing terminology or venue does not make an inherited mechanism an original contribution.

## Current status

Root sent these arguments to Astra. Astra agrees with closed-form joint attention and selected the narrower control: both arms have the same joint pair distribution and pair context; the control reads entities using separate priors, whereas the proposal reads entities using the joint distribution's marginals. Endpoint loss is removed. The revised decision is saved in ASTRA_FIRST_PAPER_DECISION_2026-10-28.md and is the selected research plan, superseding the unrun iterative-solver comparison. No code, training or GPU launch is authorized by this note. Model implementation/preflight and current GPU budget must be established separately.

For linear context fusion, the proposed readout can be rewritten as one attention-weighted sum over augmented pair values: sum(T_ij [W_s e_i + W_o e_j + W_r r_ij]). Therefore, do not describe marginal pooling as a new optimization or feedback algorithm. The modest tested contribution is the shared-weight readout design in A and its measured value relative to the matched control, not an alleged new mathematical primitive. This equivalence is a claim boundary, not a reason to reopen endless search or discard every application-specific improvement.
