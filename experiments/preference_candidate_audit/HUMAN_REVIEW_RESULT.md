# Human-assisted review rejects the current automatic pair recipe

2026-10-25. Original export SHA256 `7e509c235c42c0623c7947a1a6e306a07765493caed8481cfc280b6acb078fd6`. Review fingerprint matches all30 locked candidate pairs. All30 IDs, explicit confirmation flags, allowed labels and AI-assisted provenance validate. The original export is retained unchanged in ignored local storage. Empty human notes are preserved; AI rationale is not relabeled as human rationale.

| Judgment against the automatically selected preference | Count |
|---|---:|
| Human prefers selected winner | 13 |
| Human prefers the other caption | 9 |
| Tie | 4 |
| Unclear | 4 |
| Total verified | 30 |

The user changed COCO256875 from AI `unclear` to `X`. All other exported choices match the preannotations. This is human verification of AI-prefilled labels, not an independent unassisted replication. The assistant's rationale must remain separate from the user's own empty notes.

## Decision under the prelocked screen

The rule required at least27/30 favoring the selected direction and no systematic parser/compound errors. Agreement is13/30 (43.33%); the agreement condition fails, so the joint gate fails without requiring any claim about absence of systematic errors. Among the22 decisive X/Y choices,13 agree (59.09%); this alternate denominator does not replace the locked denominator30. Even assigning all8ties/unclear hypothetically to the selected winner would yield21/30, still below27. No further labeling is needed to decide this experiment.

**Stop the current automatic candidate-pair construction recipe. Do not expand it to TRAIN10k or train C/T/U from these labels. Do not reverse the9 pairs, drop inconvenient records, relax thresholds or rerun the same30 to manufacture a pass.** If a different data policy is proposed, it needs an explicit new hypothesis and separate validation.

The preceding numeric result remains valid in its own scope: many candidates differ in CIDEr and an oracle can choose higher-scoring captions. This review shows that the current CIDEr/object/recall/length rule does not reliably select the human-preferred direction in this small screen. It does not demonstrate a trained model gain, a failure of all preference learning, or failure of every automatic preference policy. It also does not establish a new alignment mechanism.

## Next research decision

Keep A and the existing candidate bank, raw user labels and provenance. Report this failed label-quality screen to Astra before another GPU run. The current blocker is preference-label validity and the unresolved methodological contribution, rather than generation throughput or absence of numeric candidate headroom. Root performed only JSON validation/aggregation; no model or training was run.

Three synthetic checks pass: exact threshold/tie treatment, rejection of wrong fingerprint/missing IDs/unverified or invalid labels, and preservation of human changes.
