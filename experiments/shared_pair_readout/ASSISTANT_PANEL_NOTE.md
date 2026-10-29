# Assistant-proposed semantic panel review

2026-10-29. The user asked Codex to propose ratings and will check them. All64 images and175 unique captions were visually inspected in displayed order, without opening the arm key or positive/negative source labels. Original JPEGs45,49,53 were additionally inspected. This is model-assisted annotation, not independent human gold, and no arm-level success claim is made before verification.

The draft is `ASSISTANT_PANEL_REVIEW.json`:175 proposals, zero human confirmations. `make_assisted_panel.py` embeds it into ignored `review/assistant_review.html`. The original blank page remains available. Assisted and blank browser drafts have separate storage keys. Changing a field or note revokes that caption's confirmation. Export becomes `human_verified_assistant_proposal` only after all175 valid annotations are explicitly confirmed and the reviewer enters a name. Partial exports retain assistant-pending provenance.

Provisional interpretation rules, offered for human adjudication:

- Absence of the target relationship is not a false relationship. A relation involving a different object does not count as the target pair merely because the target box overlaps it.
- Spatial role inverses count as paraphrases, such as person on bed versus bed beneath person. Ordinary physical on/holding/milking is taken to entail contact for touching assertions; cutting via a tool does not necessarily entail direct person-object contact.
- Generic co-occurrence with does not automatically specify a spatial relation; uncertain target-relation readings are retained rather than forced to yes/no.
- Evaluate the relation actually asserted, not just its broad entailment. For image45, on the elephant's back implies touching the elephant, but the specific back relation is visibly wrong; q_claim=yes can coexist with relation_correctness=wrong.
- Unknown tokens are not reconstructed. Wrong noun labels are not automatically repaired from an assumed intended object. Cases with ambiguous endpoint identity remain uncertain.

Priority checks: image6 (with/under a book),20 (scope of next to/on a bike),45 (elephant back/trunk versus side),47 (on top versus in an open bag),49 (which scissors instance),53 (bowl versus plate). Notes for every image are visible in the assisted page. These choices are provisional and should be revised by the human reviewer, not defended as ground truth.

No new training, inference or metric run. Script syntax, ID coverage, annotation consistency and human-confirmation gating are checked locally; browser rendering/interaction is unverified because the browser tool rejected local file URLs. No alternate browser surface was used to bypass that restriction.
