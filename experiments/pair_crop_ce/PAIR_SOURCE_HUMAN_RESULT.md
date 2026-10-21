# Human review result — 21 October 2026

Decision: **NO_GO_FOR_FULL_SCALE_PAIRCROP** under the prespecified criteria. Stop this crop-only variant rather than increasing training scale, changing crop margins/seeds/checkpoints, or adding SCST to rescue this result. This does not disprove visual–semantic alignment as a research direction.

The received human export contains all32candidate judgments, matches the locked review fingerprint, and has no unresolved labels. Candidate IDs match the hidden X/Y key exactly. Within each event, identical outputs have consistent labels. No human label was changed or inferred from template/verb matching.

| Human outcome | FULLFRAME | PAIRCROP |
|---|---:|---:|
| Correct and specific phrases | 10/16 | 11/16 |
| Wrong relation or additional fact | 5/16 | 4/16 |
| Wrong instance | 1/16 | 1/16 |
| Groups with both selected pairs correct | 2/8 | 3/8 |
| Skateboard joint groups | 2/4 | 3/4 |
| Surfboard joint groups | 0/4 | 0/4 |

One group improves, none regress: netgain+1. The gain is group`crop_probe_2ccc00fada5ec769` (COCO153249): for the carrying target, FULLFRAME generates `a person on a skateboard on a ramp.` (human: wrong relation/fact), while PAIRCROP generates `a person carrying a skateboard` (human: correct and specific). The other target is correct in both arms.

The criterion requires PAIRCROP>=6/8, netgain>=3, regressions<=1, and gains in both families. Observed:3/8, +1,0, skateboard-only. Only the regression condition passes. Ranking results1/8versus3/8 from the earlier audit must not replace these human generation results2/8versus3/8.

## Label consistency note, without changing the primary result

In COCO355593/group`crop_probe_11a6923598414ac0`, the second selected pair had previously been assigned `a person standing on a skateboard` as a positive target. PAIRCROP generates that exact phrase, but this later human review marks it wrong relation/fact. FULLFRAME's carrying phrase for that event is also marked wrong. This is a disagreement between the earlier reference and the later output review, not justification to override either record. It may concern the relation definition or original target verification; it needs inspection only before reusing that sample as training/evaluation truth.

Even granting PAIRCROP success on that disputed event would yield at most4/8 groups, netgain+2 and still no surfboard gain: the NO_GO decision is unchanged. No additional labeling is required to settle this experiment's decision. Keep both annotation versions and their provenance.

## Interpretation and limits

Re-encoding a pair crop helps one observed skateboard group. It does not provide a sufficient general mechanism for relation-sensitive phrase generation across both families. The surfboard contrast and instance failures remain. This is a small, one-reviewer, AI-assisted study on8groups new to adaptation, not an independent test of the whole caption system or a new CIDEr/CHAIR result.

Next research work should inspect the remaining surfboard and instance failures, including whether local features or the learned readout distinguish the selected pair. This result alone cannot identify the exact cause, prove CLIP incapable, or justify a particular replacement architecture. Preserve the best full-caption baseline and both pilot adapters; do not promote this pilot as an improved caption model.

All aggregation ran locally using JSON/HTML only; no model, notebook, or training was run. Synthetic summary tests cover paired grouping, incomplete/wrong-fingerprint/missing exports, and preservation of unclear/user labels. The original human export is retained in ignored local review storage, with its SHA256 in the result JSON.
