# Native concept-mass grounding C/G pilot

This is a user-run candidate method, not a demonstrated improvement or novelty claim. It implements Astra's 2026-10-29 proposal with the explicit [negative-label amendment](../semantic_grounding_method/CG_NEGATIVE_POLICY_AMENDMENT_2026-10-29.md). I/J remains stopped; A remains the best model. No local model/forward/autograd was run.

## Two notebooks

- `CONCEPT_MASS_C_2GPU.ipynb`: control, native A continuation with SCST + .1 rollout KL + .1 positive concept preservation.
- `CONCEPT_MASS_G_2GPU.ipynb`: same, plus .1 negative concept-mass correction.

Use one account per notebook, **T4x2 + Internet ON + Save & Run All**. Both use the existing A checkpoint, original prompt tokens, visual cache, detector JSON and Karpathy COCO JSON. No RGB, UPT, VSR, I/J checkpoints, extra image cache or API token. Preserve previous outputs; new names are `/kaggle/working/concept_mass_C` and `_G`.

Inputs already filled:

| Dataset | Mount/file |
|---|---|
| pair-source-fullframe-ce | `upt_hidden_caption_A/checkpoint.bin` |
| prompt-cache | `prompt_clip_tokens_cache.pt` |
| visual-cache | directory with original CLIP HDF5 shards |
| objectdetectionecache | `objectdetectioncache.json` |
| vuthetam/mscoco-2014 | `dataset_coco.json`, references only |

A SHA630a4b6d...; data SHA2fd99922...; prompt SHAd7910e9b...; detector SHA498825da... . Paths and complete SHA values are pinned in code/notebooks. Exact checkpoint tensor identity and vocabulary are checked on the user runtime. The shared source commit and data-content digest must match between C/G; the baseline captions must match the historical A exactly before training proceeds.

## Locked data and model

Population is the same TRAIN10k used previously, with full official COCO TRAIN/restval instance annotations for those10000 images, including crowd/small objects (71950 annotations,28858 image/class presences). It is not the retained routing-span subset. The frozen seed42 schedule uses4000 distinct images over500updates, global batch8 (4/rank), without repeating the source population's first epoch. Each step cycles reference index `(step-1)%5`. Fixeddev1000 is disjoint from the adaptation train images. This is a pilot, not full training.

`TRAIN_PRESENCE.json.gz` contains only TRAIN labels. `DEV_DIAGNOSTIC_GT.json.gz` is separate and read only by the post-training diagnostic, never the training-event selector. All model inputs stay original CLIP features and original A prompt features. No GT, detector classes, flags or reference text are inserted into the inference memory.

Only decoder layers1/2 (the last two) learn; everything else and a frozen A decoder teacher remain fixed. Dropout stays off for sampling/rescoring. PAD/BOS remain possible policy actions as in A, and EOS counts as an action. Event positions preserve the original token sequence, including intervening PAD/BOS, rather than indexing into a collapsed decoded string.

## Probability event and loss

A class event is the union of prefix-free token paths for the fixed, unambiguous aliases available in A's whitespace vocabulary. There are72 nonempty classes,510 disjoint-frontier paths,97 multiword paths. This is a curated alias set, not every synonym or all80 class meanings. Tokenization retains punctuation; punctuation variants are admitted at phrase ends only. Ambiguous aliases, depiction/compound cases and unknown tokens abstain. Longest matches prevent hot dog from becoming a dog mention. No EOS is appended to noun events. Only surfaces actually represented by a lexical event are eligible. Branches that cannot finish within the remaining generation horizon have zero event probability; they are not length-normalized.

Single-token branches reuse the policy/reference logits. Multiword branches are rescored through the real decoder in chunks64 without dropping branches/events. Exact sequence probability products are summed over disjoint events in log space; no geometric means or length normalization. Student and teacher use the same event definitions.

- Negative event: a sampled, safe mention class unrecorded in TRAIN GT and all five references, without an available detector veto. Existing A annotation/reference-union support is also a conservative veto. Reduce the entire curated alias mass with `-log(1-q)`.
- Positive event: a safe reference noun mention whose class is in the full GT set. Keep loss is `relu(log(q_A)-log(q_student))` at that reference prefix, not at arbitrary prefixes.
- KL: A versus student on the exact on-policy sampled prefixes, token-mean per image over active actions.
- SCST: unchanged full TRAIN/restval IDF CIDEr-D reward + .5 original object-count-v2 reward.

Positive/negative losses use global event-count reduction over both ranks. C monitors the same selection rule and class mass, but receives no negative gradient. C/G samples can diverge as their policies learn; equal starting seeds do not require identical later samples/counts. Selected events have no gradient through selection.

## Negative supervision amendment before any run

The original extractor of the pinned detector JSON is unavailable. It has123287 records and max87 objects/image, but neither fact certifies completeness or export thresholds. Its provenance remains **unknown**.

Therefore, negative labels are defined only as **annotation-conditioned weak negatives**, using full official GT + references, not detector absence. Every valid stored detector hint vetoes a negative (no .5 threshold and no default confidence). A detector does not create a positive preservation anchor. Missing/empty detector records abstain conservatively; malformed labels/scores stop preparation. No absence is certified. Raw GT/ref misses can still produce false negatives, especially if a missing detector veto is unavailable.

Do not change this policy after seeing dev metrics to obtain more negative events. Zero negative exposure means the correction objective was untested, not a completed fair C/G improvement experiment. Counts, class coverage, weighted auxiliary gradient norms, and exact image exposures are exported. These labels can optimize the COCO/CHAIR ontology and may still encourage generic captions; they do not prove physical grounding or correct instance binding.

## Checks, metrics and budget

Local checks are metadata/NumPy/AST/mock only. User GPU preflight checks actual A identity, trainable scope, native teacher/student logit+greedy parity, sample-rescore policy parity, real decoder event mass against a separate probability-tree oracle, finite nonzero negative gradients, serial versus two-rank unequal-event gradient reduction, zero-event reduction, frozen states and no state mutation.

User pipeline: CPU metadata test/metric dependency preflight -> CPU copy of the scheduled4000+dev1000 cached rows (no feature extraction) -> actual2GPU preflight -> A captions ->500updates -> endpoint captions -> neutral-prefix diagnostic -> CPU official PTB/CIDEr/BLEU/METEOR/ROUGE/CHAIR evaluation -> review ZIP. Persistent source HDF5 dataset handles avoid the prior per-row metadata lookup overhead.

Caps per arm: CPU prepare1800s, GPU phase3600s, CPU evaluation900s; dependencies have separate900s command caps. These are limits, not measured ETAs. The user reported24remaining GPU hours; no account distribution or billing conversion is assumed. Enabled GPU during preparation/CPUmetrics can still consume Kaggle quota outside the measured GPU phase.

The diagnostic uses the fixed neutral prefix `a photo of a`, all representable classes with at least5 GT-supported and5 weak-negative dev images, at most32 hash-selected images per class/state, chosen before scores. It reports each support-state mean as well as macro separation. Lowering both states is not claimed as better visual conditioning. This weak-label sensitivity diagnostic is not a causal or independent test.

Send `concept_mass_C_review.zip` and `_G_review.zip`; keep `checkpoint.bin`. Failure archives are marked incomplete, even if a checkpoint exists. One seed and reuseddev1000 are exploratory. A new best is not automatically promoted.

Continue only if G has a useful signal versus both fixedA and matchedC: higher CIDEr, lower CHAIRs/i with absolute counts, and recall no more than .005 below A. Report paired intervals for CIDEr and CHAIRs, recall/length/object counts, and uncertainty; a point-estimate signal is not confirmed robustness. If G only degrades less than C, changes nothing, has no useful negative exposure, or reduces hallucination by omitting content, stop this recipe. Native interface, SCST, UL, aliases and preservation are inherited ideas, not new primitives. No claim of being first or of publication acceptance.
