# Frozen-A attention geometry diagnostic

User-approved 25 October 2026 following Astra's REVISE decision. **No AISK training**.

Open `ATTENTION_GEOMETRY_DIAGNOSTIC_200_2GPU.ipynb` on Kaggle, enable Internet and T4x2,
attach the inputs in its first code cell, then Save & Run All. The checkpoint is
the trained hidden-caption **A**, SHA630a4b6d…, not the earlier legacy SCST checkpoint.
The noun-box dataset is the same sealed v1.1 input used for routing B/R.

### Observed preparation blocker and correction

The user-run review proves that the pinned legacy cache has only `tokens`/`mask`:
all10000 TRAIN entries were rejected for missing original prompt text. This was a
metadata blocker, **not** an attention/grounding result. No A forward was performed.
Use the **original** text JSON consumed by the embedding-cache producer, never guessed
YOLO templates or UPT prompts. The historical notebook `kltn-mscoco (1).ipynb` records
`/kaggle/input/models/ducanh2403/promt-cache/maxtext/default/1/prompt_cache.json` as a
text-cache candidate; this path is provenance evidence, not proof of file availability
or matching embeddings. The notebook accepts that file or a sibling `prompt_cache.json`
beside the legacy .pt. Correct PROMPT_TEXT for the real mount. It records the JSON hash
and verifies each selected prompt's encoded vectors against the unchanged cached tokens
before any A forward. A wrong JSON must fail binding. If the original JSON is unavailable,
the phrase-grounding diagnostic remains blocked; do not loosen eligibility to bypass it.

## Locked experiment

Select up to200 hash-first eligible images from locked A TRAIN10k using metadata only.
Two distinct conservative prompt noun classes must each have a unique GT instance in
the sealed noun-box manifest; at least one pair must have an edge gap >= one patch
on x or y. This is a deliberately restricted subset, not representative COCO accuracy.
Coverage includes all10000 candidate images, missing text/mask failures and eligibility.
Multiple instances/missing mentions are not negatives. Existing noun labels are weak,
automatically joined via references; they are not new human-verified gold labels.

Use original cached legacy tokens as input. Extract original prompt strings and use
pinned CLIP fast-tokenizer character offsets, excluding special/padding/truncated
subwords. Verify every selected text encoding against the original cache using a
pinned frozen CLIP text encoder (rtol/atol .002, accommodating FP16 storage). This
verification does not replace any cached tokens. Missing text or failed binding is
a data blocker, not a reason to inject UPT text or reference captions into A.

Read last-layer Q→197-token and prompt→32Q attention by passive pre-hooks, preserving
the actual model calls/output. Average heads, compose through **CLS+patch**, record
CLS mass, then condition each prompt token on its patch mass and average noun subwords.
This is an attention proxy, not full attribution through residuals/self-attention/CLIP.
Q slots are not assigned instance identities. Verify exact same-shape memory and
BOS-logit parity with/without hooks on one image per GPU; verify frozen tensor digest
before/after. No gradient/optimizer, no caption generation, no DDP collectives.

Controls on the same images: uniform; mean of eligible noun maps within image;
hash-fixed no-fixed-point cyclic permutation of **whole** noun maps within image.
The permutation preserves maps/geometry/entropy but breaks phrase association.

Primary: per-image mean noun GT fractional-patch mass minus GT area. Also centroid
inside GT, entropy, sigma, CLS mass, sizes and separated-axis order accuracy. Ties
score .5 and are counted separately (image-mean/uniform always tie). Fractional box
masks assume the verified A cache's square224 bilinear resize; no crop inversion.
Bootstrap10000 resamples **images**, preserving all phrases and pairs in each image.
Localisation contrasts condition on the restricted metadata-selected cohort.

Go-to-design only: >=100 images AND paired95 lower bounds >0 for true−image_mean and
true−deranged on BOTH excess-mass and axis-order accuracy. Otherwise NOT_YET_GO,
not a universal rejection of geometry. No automatic threshold relaxation or rescue
with beta/R. No automatic implementation of a learned module after the gate.

Send `attention_geometry_diagnostic_200_review.zip`. Receipt DIAGNOSTIC_COMPLETE is
required; error ZIPs intentionally preserve logs/coverage/source but are not results.
The ZIP includes maps and noun-box joins; selected legacy tensor subset is excluded.
Map archives also preserve `q_full_<COCO_ID>` (32×197 including CLS) and
`p_to_q_<COCO_ID>` (all legacy slots×32), allowing an offline Q-capacity/proxy-bound
analysis later. These are observations, not semantic identities for Q slots.

## Validation and limits

Local checks are NumPy/CPU synthetic policy/metric tests, syntax, notebook size and
embedded source hashes only. Torch observer synthetic checks, text binding, actual A
parity, cache coverage and model forwarding are performed **only by the user on Kaggle**.
Runtime has not been measured. No improvement/novelty/predicate grounding claim.

Generate notebook: `python make_notebook.py`; local tests: `python test_diagnostic.py`.
The notebook uses a pinned public project commit plus hash-checked embedded overlays.
