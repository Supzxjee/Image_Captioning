# Frozen-A prompt–patch diagnostic: positive signal on a restricted TRAIN subset

User-run review audited 25 October 2026. Review ZIP SHA256:
`b387d808172f272612e61793bb9e5c3fd608e40a4fa0e663e931eadcc473d20f`.

**Saved-artifact audit PASS; predeclared premise gate PASS. This is not a caption-quality
improvement, a new trained method, or approval to train AISK-Full.**

## Coverage and technical checks

The original text JSON was found at the historical model path and contains123287 entries
(SHA `df88b7cafac854b98b1609e5899beab681aa60cd155e5dda5cb25bc320f3f88d`).
Selected text encodings passed the existing comparison to the original cached vectors
on Kaggle. The combined rtol=.002/atol=.002 criterion permits maximum absolute errors
.007143/.005011 at large-magnitude elements; it is not an absolute .002 bound or bitwise
text-embedding identity. Original .pt tensors remained model inputs.

Only138/10000 TRAIN images (1.38%) met all locked conditions, so the nominal200 maximum
was not reached. All138 eligible images were retained without changing filters.
9161 images lacked two eligible distinct prompt noun classes;701 further images lacked
a sufficiently separated box pair. These are restrictions of this diagnostic, not
counts of all incorrect/missing prompts. There were298 phrase-class instances,
54 categories and156 separated-axis constraints. Labels are automatic conservative
joins to unique COCO instances, not a new human gold grounding set.

Both workers completed69 images, reporting about62.2/62.6seconds each for their worker
section, excluding preparation/install/model-download overhead. Their receipts confirm
exact A checkpoint/tensor digests, unchanged frozen parameters and zero memory/BOS-logit
difference on the one parity image per worker. Raw checkpoint and original cache bytes
are not included in the review; those checks are receipt-supported, not independently
rerun on this machine. No optimizer, backward or caption generation was used.

All receipt payload hashes and embedded implementation hashes were verified. NumPy
reconstructed phrase maps from saved prompt→Q and Q→CLS+patch weights (max abs error
1.84e-8), reproduced metrics and10000 image-level paired bootstrap intervals. One
entropy-correlation difference below4e-16 reflects numeric rounding. No local ML run.

## Results

All figures below are means of per-image summaries on the same138 restricted images.
Mass is conditional on the196 patch tokens; CLS path mass is reported separately.

| Map | Mass inside noun GT box | Centroid inside GT box | Separated-axis order accuracy |
|---|---:|---:|---:|
| Correct phrase map |32.33%|28.26%|74.82%|
| Uniform |14.38%|20.05%|50.00%|
| Mean eligible noun map in the same image |26.82%|21.26%|50.00%|
| Whole maps deranged among noun classes in the same image |21.64%|17.15%|24.28%|

Uniform/image-mean centroids tie on ordering and receive .5 by the locked rule; this
is not measured random-classification accuracy. With two nouns, the derangement
reverses their maps, making it a strong association control but not independent data.

| Paired contrast | Mean difference | Image-bootstrap95% CI |
|---|---:|---:|
| Correct−image-mean excess box mass |+.055025|[+.044112,+.066379]|
| Correct−deranged excess box mass |+.106849|[+.084949,+.129598]|
| Correct−image-mean axis order |+.248188|[+.177536,+.315217]|
| Correct−deranged axis order |+.505435|[+.369565,+.637681]|

All four required lower bounds are positive and n>=100: the predeclared premise gate
passes. The maps carry noun-specific spatial signal beyond shared image saliency on
this subset. They do **not** consistently localize sharply: only28.26% centroid hits.
Direction summaries are uneven: x+84.62%, x−64.86%, y+53.85%, y−76.09% (different
small image subsets). Do not claim reliable vertical/predicate understanding.

## Q-capacity analysis from the existing saved tensors

For each noun mask, max over the32 conditional Q maps gives a diagnostic GT-informed
single-Q oracle. The mean per-image oracle-minus-actual conditional mass gap is
**.251275**, CI95[.237499,.265191]. Phrase-weighted oracle conditional mass is.573434;
do not subtract the image-weighted.323264 from this differently weighted statistic.

Without dropping CLS mass, phrase-weighted actual box mass is.316081 and the single-Q
oracle is.566661. Phrase-weighted prompt-path CLS mass averages.018797. Both raw and
conditional proxy upper bounds were checked for every phrase.

This suggests room to improve prompt selection among the existing Q maps; it does not
establish a learned selector can attain an oracle selected with GT. Maximizing32 maps
inflates the oracle, and no Q slot is assigned a semantic identity. The current
diagnostic does not separate object appearance from noun-position priors completely.

## Research decision

Send this compact audited evidence to Astra to choose the smallest next pilot. The
evidence favors inspecting/learning phrase→Q selection over assuming that no Q can
represent a suitable region. It does not mandate AISK moments or a new memory branch.
Before training, fix the matched continuation control, trainable parameters, source of
noun–box supervision, decoder-utility measurement and free-caption endpoint. Existing
automatic phrase labels and narrow coverage need to remain explicit.

Do not infer CIDEr/CHAIR improvement from these localization numbers. A positive TRAIN
diagnostic and a GT oracle are neither a held-out result nor proof of paper novelty.

Reproduce saved-data audit: `python audit_result.py` after caching the review payloads.
See `RESULT.json` for complete numeric results and limitations.
