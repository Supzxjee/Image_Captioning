# Frozen routing utility: A/T/W on the locked138 TRAIN images

Implements Astra's decision after the positive frozen-A phrase-grounding diagnostic.
**User runs Kaggle. No local ML execution, no training or teacher-bank construction.**

Open `ROUTING_UTILITY_ATW_138_2GPU.ipynb`, attach the four input datasets listed in its
first code cell, enable Internet and T4x2, then Save & Run All. The two GPU workers
split69 images each without DDP collectives. Send `routing_utility_ATW_138_review.zip`.
No checkpoint is created. Runtime has not been benchmarked; setup and scoring add
overhead beyond the previous138-image attention forward.

## Locked data and conditions

`cohort.json` is a hash-sealed metadata package derived from audited review
`b387d808…d20f`, saved raw Q maps and the existing sealed v1.1 noun spans. It embeds the
same138 IDs, box masks, unchanged original prompt positions and reference spans; it
does not replace original cached embeddings. Original text binding was verified in
the prior run. Thus no re-upload of the old review or original JSON is required.

For each image, hash-select one unique separated-box pair and a receiver direction,
without caption scores. Select q+ and q− by **raw** Q patch mass in the receiver and
donor box, retaining CLS outside spatial mass. Metadata shows137 distinguishable Q
choices and one indistinguishable image. Do not find an alternate pair to rescue it.

Build factual A memory. At the existing prompt MHA output `attended_prompt`, before
gate and layernorm, move only the receiver's CLIP subword rows toward the actual learned
one-hot visual read for q+ (T) or q− (W). Both use the original MHA value/output
projections, including biases. With d+/d− the Frobenius displacement norms, use
h=.1*min(d+,d−), lambda+=h/d+, lambda−=h/d−. Every lambda<=.1. Preserve original text,
all other prompt slots and Q memory exactly. The gate/LN remain original and frozen.

Skip indistinguishable/degenerate/no-positive-raw-headroom cases with explicit reasons;
keep their original A predictions in both T and W on the full138 cohort. Never select
alternate Q, pairs, strengths or favorable captions. Fresh Q choices must reproduce the
prior audited metadata. Log pre-fusion matched norms and unequal post-fusion norms.

## Outcomes and screening rule

Teacher forcing uses the same five fixed reference prefixes across A/T/W. Noun scores
average subwords of the first eligible mention per reference, then references per
image. Validate against actual first alias mention and checkpoint vocabulary. Missing
or ambiguous targets are not negatives. No reference text is injected into prompts.

Benefit = logp_i(T)−logp_i(A); contrast = logp_i(T)−logp_i(W);
selectivity = contrast−[logp_j(T)−logp_j(W)]. Report the same valid-intervention image
population for benefit/selectivity, image-bootstrap10000 CI, median and positive count.
This asks whether i benefits more than j, not whether a true donor noun should be false.

Generate raw greedy A/T/W, max30 including BOS, no reranking. Official PTB/CIDEr uses
fixed locked TRAIN10k reference IDF (blank hypotheses outside138 only preserve IDF);
do not compare magnitudes to val/dev IDF. Run pinned CHAIR, reproduce its numerators
and denominators, record recall, mean length and all138/secondary-intervened contrasts.

Screening Go-to-design only requires benefit and selectivity lower95CI>0, >=50 common
primary images, no more than50% nonintervened, changed captions, positive full138 mean
CIDEr T−A AND T−W, and no increase CHAIRs/i, decrease recall or mean caption length
against A. The explicit50-image/50%/no-length-decrease guards are conservative budget
choices, **not power calculations**. CIDEr need not have a positive lower CI for this
small screen; a positive mean is not confirmed generalization gain. No automatic train
follows a Go. Failure does not reject all learnable alignment/co-adaptation.

## Validation

Local:6 model-free tests of selection/raw CLS-aware Q choice/magnitude/utility/coverage;
AST checks; embedded imports and metadata hash consistency; unexecuted notebook <1MB.
Kaggle only: synthetic MHA one-hot/convex equivalence, matched norm and untouched rows;
zero-rho exact same-shape memory/BOS-logit parity; complete frozen A digest before/after;
real cache/target/scorer checks and caption generation. A ZIP alone is not completion:
require `receipt.json` with `ATW_COMPLETE` and both hashed worker outputs.

GT boxes make T an oracle diagnostic, not a deployable method on new images. Q may
carry global context; the intervention does not remove a real object from an image.
Norm matching before fusion does not control all downstream perturbation properties.
The138-image set is a restricted, repeatedly used TRAIN subset with automatic weak
phrase labels; no claim of relation understanding, independent testing or novelty.

Rebuild metadata: `python prepare_cohort.py` (needs ignored saved review/NPZ files).
Package notebook: `python make_notebook.py`. Test locally: `python test_utility_data.py`.
