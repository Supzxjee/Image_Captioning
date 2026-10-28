# VSR32-record human-assisted audit — complete

28/10/2026. **HUMAN_REVIEW_COMPLETE**. Reviewer: Duc Anh. No model/GPU execution or manifest relabeling.

## Effective result

- **32/32** records explicitly confirmed; S/O boxes clear for all32.
- **16supported /16contradicted /0uncertain**.
- The32definite judgments agree with the corresponding released VSR labels. This is agreement on a fixed small sample, not proof of whole-cohort truth or a model result.
- Human answers override three assistant suggestions: #16/#22 initially uncertain, now contradicted; #32 initially supported, now contradicted. No records were dropped or replaced.

## Provenance preserved

The uploaded file has31confirmation flags and retains `assistant_proposed_pending_human_verification` because #12was unticked. Its bytes are preserved, SHA256 `0daa710272b2a403889f8c3eecbd34d5da09d25e2533c7b0561fe5d8db3f1d09`; it was not rewritten as a fully human file.

Direct user chat supplies the missing evidence:

1. #12horse next-to bus, COCO172406: user answered **“Đã kiểm tra, giữ đánh giá đó”**, explicitly confirming contradicted and both clear boxes.
2. #16person next-to horse, COCO497579, and #22suitcase below cat, COCO324483: user then wrote **“16 22 false nhe”**. These are explicit assertion revisions from uncertain to contradicted, not automatic promotion of the prior AI draft.

`HUMAN_AUDIT32_CHAT_SUPPLEMENT.json` records these messages separately, bound to the uploaded-file hash. `HUMAN_AUDIT32_EFFECTIVE.json` combines the raw31confirmations with those direct replies. Notes inherited from the earlier assistant draft are preserved as historical notes; where they mention uncertainty, the explicit human revision field takes precedence. #32was already changed and confirmed in the uploaded file; no explanation was invented for that revision.

The frozen V2manifest SHA remains `6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7`. All source VSRlabels remain unchanged.

## Checks and next step

Five metadata-only tests passed: retain valid per-record confirmations in a mixed export; apply explicit revisions; reject wrong audit identity/unknown records; prohibit silent changes during mere confirmation; require the correct prior value and trusted-chat origin for revisions. Raw input and effective output remain separate. These tests perform no neural computation.

The fixed sample review has no remaining incomplete/uncertain entries or definite source-label disagreement. This is one **AI-assisted, self-reported human review**, not independent double annotation, an inter-rater result or a gold-label guarantee for320pairs. The assistant's former concerns remain documented without overriding the user's final choices.

Next prerequisite is the user-run CPUcache-copy reviewZIP and its pinned original-prompt coverage/array checks, followed by model preflight. No cache-copy receipt was supplied with this review, no training has run, and `training_approved` remains false. Astra received the final human update; the earlier message about two unresolved cases is superseded.
