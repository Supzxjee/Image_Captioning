# A/T/W frozen decoder-utility diagnostic: completed, No-Go

Audited26 October2026. User review ZIP SHA256:
`34c3070226cb88c82be4a4ecbd5332ed7a4b05cb5ec83f208bec2139468ff7f0`.

**Technical saved-artifact audit PASS. Research screening decision: NOT_YET_GO.**
Do not build the proposed routing teacher bank or train its distillation recipe from
this result. The GT-box-max-Q correction has not demonstrated useful decoder alignment.

## What ran and what was checked

Exactly138 locked TRAIN images,69 per GPU, with unchanged A checkpoint.137 received
the predeclared norm-matched T/W correction; one indistinguishable-Q image retained A
in all conditions. All138 had reference mentions for both selected noun classes; the
same137 intervention images form the primary benefit/selectivity population.

Both workers reported zero same-shape memory/BOS-logit difference on their parity
image and the original frozen A tensor digest at completion. Each worker section took
about73seconds, excluding setup/preparation/scoring. Synthetic tensor and scorer
preflights completed in the saved log. No optimizer, training or held-out evaluation.

All receipt payload and embedded source hashes were verified. Saved noun contrasts,
image-bootstrap10000 intervals, caption identities and CHAIR numerators/denominators
were independently reaggregated with NumPy. All137 corrections had increased raw
receiver box mass, convex coefficients<=.10 and matched pre-fusion norms. Mean raw
box-mass increase was.022328 (range.002280–.057212).

Post-fusion T/W norm ratio was not exact: median1.0241, range.7836–1.4118. Thus norm
matching controls one nuisance before gate/LN, not every downstream difference. This
limits a causal interpretation of the sign of T−W. Actual memory tensors/checkpoint
bytes are not in the ZIP; frozen-state/parity/untouched-slot checks remain supported
by verified source and on-Kaggle receipts. Local audit did not execute a model, rerun
Java/PTB scoring or reparse CHAIR word annotations from captions.

## Teacher-forced noun utility: primary outcome failed

Units are nats per target noun token, averaged within reference then image.

| Contrast on137 images | Mean | Image-bootstrap95% CI | Positive images |
|---|---:|---:|---:|
| Benefit: noun_i(T)−noun_i(A) |−.002612|[−.004846,−.000450]|57/137|
| Correct-vs-donor: noun_i(T)−noun_i(W) |−.003606|[−.006507,−.000749]|44/137|
| Selectivity: preceding contrast minus noun_j(T)−noun_j(W) |−.003868|[−.007653,−.000152]|49/137|

Effects are small, but all three intervals are below zero rather than supporting the
required positive benefit/selectivity. Increasing this attention-mass proxy did not
make the intended noun more probable under the frozen decoder in this recipe.

## Free captions on the full138-image cohort

Official PTB/CIDEr uses the fixed TRAIN10k reference IDF; do not compare these absolute
values directly with old dev1000 or val/test scores. CHAIR is automatic object-level
annotation-based hallucination measurement, not human verification or predicate truth.

| Metric | A original | T correct-box Q | W donor-box Q |
|---|---:|---:|---:|
| CIDEr |1.485343|1.487353|1.483181|
| CHAIRs |6.5217% (9/138)|7.2464% (10/138)|6.5217% (9/138)|
| CHAIRi |3.4130% (10/293)|3.7543% (11/293)|3.4364% (10/291)|
| Object recall |45.7711% (276/603)|45.6053% (275/603)|45.4395% (274/603)|
| Mean caption words |10.2319|10.2174|10.2246|

T−A CIDEr=.002011, paired95CI[−.003923,.008889]: no established improvement. T−W=.004172,
CI[0,.012517], also not a strictly positive lower bound. T changes only3/138 captions
(2.17%). It adds one CHAIR-flagged caption/mention and loses one recalled object versus
A, so the content/hallucination guard also fails. Do not advertise the positive CIDEr
point estimate as success.

The three changed examples are preserved in `RESULT.json`:

- COCO448409, couch←tv: A lists couch/desk/television; T and W both omit couch and say
  “a living room with a desk and a tv”. Both score higher CIDEr despite deleting the
  receiver noun. This is not evidence of improved target grounding.
- COCO398225, couch←bed: T changes “dresser” to “chair”; W produces a different bed/window
  caption. This one image accounts for the entire T−W CIDEr difference on the cohort.
  Metric changes are not a visual correctness judgment of those objects.
- COCO131485, person←kite: T and W both reorder “a boy on the beach flying a kite.” to
  “a boy flying a kite on the beach”, lowering CIDEr. Neither condition is a new model.

## Decision and scope

The preceding diagnostic established noun-specific spatial signal and a GT-informed
Q-mass headroom, not semantic correctness of the selected Q value read. This utility
test shows that **choosing a Q by maximum box mass and applying this frozen<=10%
correction is not a demonstrated beneficial routing teacher**. It blocks the current
distillation proposal under its predeclared screen.

It does not prove every better-grounded representation harms captioning, that decoder
co-adaptation cannot help, or that visual-semantic alignment is impossible. The Q
features remain globally contextual, the oracle proxy is imperfect, perturbations
differ after fusion, and only a narrow repeatedly used TRAIN subset was tested.

Retain A; do not sweep strengths, replace pairs, select favorable images or relabel this
as a positive model result. Report the compact audited evidence to Astra and reconsider
the next research mechanism before spending GPU on a new training chain. A new method
and caption gain remain unresolved.

Reproduce the saved-data audit with `python audit_result.py` after caching the review
text payloads in ignored `review/loaded.json`. Full summary and audit limits are in
`RESULT.json`.
