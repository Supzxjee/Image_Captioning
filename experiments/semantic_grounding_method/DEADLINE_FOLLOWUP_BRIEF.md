# Follow-up to Astra: novelty, training duration and Saturday deadline

28/10/2026. User requests continued discussion with Astra, with scientific novelty mandatory and a model/experiment decision needed before Saturday31/10. No model run, new notebook or GPU launch is authorized by this discussion alone. Astra remains research lead; implementation thread critiques and executes the chosen concrete plan.

## What the completed pilot does and does not establish

The500-update T recipe failed its predeclared exploratory continuation gate. This does NOT establish a training ceiling or prove longer optimization cannot improve CIDEr. The earlier stop recommendation is scoped to that fixed recipe/budget, not every continuation hypothesis.

H is an adapted prior-inspired endpoint-ranking control. More epochs or a higher point CIDEr on H alone is not an original alignment contribution. The next proposed method must articulate a research gap and distinguish its mechanism from concrete prior; new names or extra modules/losses do not establish novelty.

There are no interim development-CIDEr evaluations/checkpoints in this pilot. Training loss/reward logs cannot tell whether dev CIDEr was rising at update500. Training blocks contain different image/prefix cases and changing policies, so their aggregate changes are descriptive rather than a controlled learning curve.

## Independent check of the small-box explanation

For nonnegative patch contributions, CG/Delta cannot exceed the maximum fractional cell support. That bound is correct but its applicability is limited: signed contributions need not satisfy it, and only105/6177groups (1.70%) or80/3998selectedvisits (2.00%) have max cell support below.8. Median max support is1;3215selectedvisits have a fully covered cell. This distribution does not support explaining the widespread T constraint failure solely by small boxes.

Saved existing-log/metadata statistics: `DEADLINE_FOLLOWUP_EVIDENCE.json`. T's mean positive-gain direction penalty rises from.000406 in updates1–100 to.006949 in401–500; mean gain loss changes.041148→.045900 and meanDelta−.000531→−.004398. This supplies no positive evidence of the intended training behavior emerging by the endpoint, but it is NOT a causal diagnosis, gradient comparison, or prediction of later CIDEr.

## Decision requested from Astra

1. Is a bounded, matched longer-training follow-up scientifically and practically justified? Which arms/endpoints/budget/selection rule would separate it from an open-ended rescue sweep? Treat this as a new exploratory experiment motivated by observed results, not a retroactive pass of the original gate.
2. If not, select ONE concrete alternative alignment method with semantic unit, module, loss, available supervision, closest-prior overlap, plausible distinctive contribution, minimal decisive comparison and key risk. Astra is not forced to retain T, H, readout or noun-only alignment.
3. Produce a plan through Saturday31/10 with decision checkpoints and a candid statement of what would remain insufficient for a paper. No publication or improvement guarantee.

Available evidence/resources: original strong A; exact shared2kTRAIN/1kDEV preparation; saved C/H/T results and logs; full existing visual/prompt caches and weak noun/box metadata. The user reported25GPUhours per account BEFORE the latest jobs; remaining quota has not been read. Reported worker elapsed for500updates+end prediction after initialA/IDF setup is about230seconds/arm; this is not total-session time or an ETA for a different method. Common preparation took20minutes26seconds and should be reused where compatible.

The compact audit and these counterchecks were sent to Astra as text. It was explicitly asked not to reread the ZIPs, run models or launch jobs. The user's deadline pressure was conveyed without treating it as justification for inventing novelty or promising a paper.
