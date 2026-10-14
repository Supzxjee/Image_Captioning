# Caption/relation lexical audit: limited coverage, no semantic truth measurement

CPU audit completed for the same512 training images and12 inspected evaluation images in each arm. Returned counts were recomputed from all records and agree with the summary. ZIP SHA256: `3f19a11c7673754a629e82dcd9600445f5f8d3f807829b1381d0a36cd8f31946`.

| Across590 train claims | Count | Rate |
|---|---:|---:|
| Object alias in at least one reference | 379 | 64.24% |
| Predicate word in at least one reference | 306 | 51.86% |
| Predicate word and object alias in same reference | 217 | 36.78% |

| Predicate | Claims | Joint lexical matches |
|---|---:|---:|
| holding | 255 | 75 |
| riding | 226 | 118 |
| wearing | 75 | 23 |
| carrying | 34 | 1 |

These are literal/limited-alias cooccurrence counts, NOT supported relation counts. Low carrying coverage may partly reflect synonym gaps (e.g., bag/luggage versus detector categories), not absent visual carrying. Missing mention does not make a relation false; reading a book need not use holding. A matching word may refer to a different actor or object, so joint matching can overcount too.

## Evaluation candidates and a concrete attachment failure

Both arms have27 selected claims across12 images. References have13 joint lexical matches; top candidates have3, and the union of the three candidates also has3. Duplicate same-class selected claims count as separate slots and can match the same sentence; these are not unique facts or relation recall.

Image348636 demonstrates a limitation: the ledger already records the ball as airborne and the bat as held. The audit marks both holding-ball and holding-bat as reference joint matches, because it searches verb/object presence anywhere in a sentence rather than attaching the verb to the correct object. It does not establish that a reference asserts holding-ball. Similarly, holding an umbrella while talking on a phone can match holding-phone without asserting that link explicitly.

Thus neither the217/590 training number nor13/27 reference matches should be used as semantic supervision, positive/negative labels, caption quality, or a reranking score. The exported ZIP contains flags rather than raw reference sentences, so this analysis does not independently inspect the exact reference sentence or resolve all attachment cases.

## Decision

The audit does not prove that caption supervision caused R95's lack of gains, nor that more training would help. It confirms the need to distinguish caption-relevant semantic content from predicted scene relations and to preserve verb-object/instance attachment when verifying claims. The selected-prompt visual failures from R96 remain stronger evidence of bad input relations than lexical absence.

Stop additional variants that merely increase relation influence or force all selected predicates into captions. A useful next method should verify the caption's actual object/action claims against their specific image evidence and abstain when localization or evidence is weak. Candidate generation and selection should be evaluated separately: under this limited literal diagnostic, the current three candidates do not add predicate coverage beyond the top caption, but semantically equivalent wording is not measured. No new verifier/reranker is implemented or validated by this result.

Keep R95's adapted checkpoints and baseline without promotion. No BLEU/CIDEr/CHAIR improvement, new training labels, independent benchmark result or novelty claim is made. All pools remain reused development, and source checkpoint may have seen these COCO images.

Local preserved ZIP/analysis: `C:/KhoaLuanTotNghiep/Code/output/caption_relation_supervision_audit_result`. Public aggregate: `audit/caption_relation_supervision_audit_result.json`. No raw caption corpus or images are published here.
