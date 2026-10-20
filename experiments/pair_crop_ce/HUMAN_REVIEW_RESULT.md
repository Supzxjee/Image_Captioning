# Human-assisted review audit — 20 October 2026

**The label gate passes: four skateboard and four surfboard probe groups are locked.** This does not approve training or establish model performance. Actual HDF5 metadata and pinned feature compatibility remain pending.

The user export contains21groups,19explicitly confirmed by the human reviewer. Groups15and17 remain unconfirmed; they are not promoted to human labels or used in the probe. Its whole-file type correctly remains `assistant_proposed_pending_human_verification`. Selection checks each group's explicit confirmation and validated choices instead of converting the entire export into gold labels. Original reviewer identity/export are retained only in ignored local storage; public receipts contain hashes and deidentified decisions.

## Locked probe

| Original review index | Family | COCO ID |
| --- | --- | ---: |
| 2 | skateboard |355593|
| 4 | skateboard |153249|
| 6 | skateboard |461543|
| 8 | skateboard |563271|
|11 | surfboard |176009|
|12 | surfboard |225682|
|16 | surfboard |300206|
|18 | surfboard |567486|

These are the first four eligible confirmed natural-scene groups per family in the original order. All have a clear human-confirmed2x2semantic contrast and clear person/object boxes. The224visibility field is not a filter: indices8,11and16 remain eligible despite unclear visibility for one event.

Groups5and9 have human-confirmed relation contrasts but are observed photo montages; they are conservatively excluded from natural same-time scene comparisons. This structural exclusion was made during image review before any new model outputs, not by caption performance. The inherited montage flag alone is not proof for arbitrary images; the two known images were visually inspected. Other old assistant ambiguity flags and notes do not override revised, explicitly confirmed human truth/box choices. Such notes may now be stale and remain part of the original audit trail.

The fit reconfirmation235799 has the original positive indices[1,0], confirmed by the human, with clear boxes. Retain14fit groups. No fit targets, sample schedule or geometry statistics are silently relabeled or changed. The7previously inspected development groups remain exposed development data.

`LOCKED_DATA.json` contains the data lock and human-confirmed targets for the8new probe groups; `HUMAN_REVIEW_RESULT.json` contains per-candidate inclusion/exclusion reasons and source hashes. The new probe image IDs do not overlap the old fit/development groups or548known excluded IDs. Previously established duplicate screening is limited to available local images, not all historical datasets.

## Validation and limits

Three pure review-lock tests PASS: unconfirmed/unclear box labels are rejected,224visibility/old AI flags cannot veto confirmed choices, and montage/noncontrast groups are excluded. Manifest/proposal identity and source hash are checked. No model or notebook was run.

One AI-assisted self-reported human reviewer, no independent adjudicator, and potentially backbone-seen train/restval images. This is a small adaptation probe, not an independent whole-system test. Raw notes may disagree with final chosen fields; final confirmed choices are authoritative for this protocol, not proof of universal semantic ground truth.
