# I/J shared-weight caption readout pilot

This implements `ASTRA_FIRST_PAPER_DECISION_2026-10-28.md`: one modest inherited-method/readout change, not a claim of a novel attention algorithm. No local neural execution. Users run the two notebooks on Kaggle; the assistant never submits or starts them.

## User run

Import `SHARED_PAIR_READOUT_I_2GPU.ipynb` and `SHARED_PAIR_READOUT_J_2GPU.ipynb` on separate accounts. Each needs **T4×2, Internet ON, Save & Run All**. Add these six inputs; exact mounts are prefilled:

| Dataset | Used for |
|---|---|
| ducanh2403/relation-frames-v2-cache-v105 | Verified612image cache and frozen VSR pairs |
| ducanh2403/pair-source-fullframe-ce | Strong A checkpoint, SHA630a4b6d… |
| vuthetam/mscoco-2014 | dataset_coco.json references only; no RGB access |
| ducanh2403/prompt-cache | Original A prompts for fixeddev1000 |
| ducanh2403/visual-cache | Existing raw CLIP features for fixeddev1000 |
| ducanh2403/objectdetectionecache | Predicted-box banks; no new detector |

No UPT, learnedframe200 checkpoint or API token. CPU preparation copies existing values only. Dev image dimensions are already packaged from a verified official COCO annotation member, without copying GT boxes into the model.

Each arm has fixed caps: dependencies900s per command, CPU prepare1800s, two-GPU phase3600s including preflight/baseline/train/predictions, CPU evaluation900s. Caps are not measured ETAs. User reports25remaining hours; account allocation and Kaggle billing units are not inferred or doubled. Two GPU phases together are capped at4device-GPU-hours; enabled-GPU notebook preparation/dependencies/CPUevaluation add usage outside this measurement. Check the account's displayed balance before launching.

Fresh output/log/review required; previous files are preserved. Version2 defaults to `shared_pair_readout_I_v2` / `shared_pair_readout_J_v2`. Send their `_v2_review.zip` files. Keep both `checkpoint.bin` files. Failures still archive logs/config; a checkpoint may exist after a later evaluation failure, but the run is incomplete until summary status confirms full evaluation. No automatic dataset upload, seed2, reranking or promotion.

If an old `shared_pair_readout_I`/`J` folder with `preparation.json` remains in the same session, v2 detects it and verifies/reuses its three prepared payloads in a fresh output directory. No old checkpoint/history is resumed. Set PREPARED to a saved notebook-output/dataset directory if needed, or None for fresh preparation. Reuse requires matching preparation source hashes, fixed policy/input identities and full payload hashes; it hardlinks when possible, otherwise copies and verifies. Original data/results are unchanged, and lineage distinguishes the old data-preparation commit from the corrected model commit.

## One controlled difference

Both use the same object values e, ordered-pair values r, prefix queries, parameters and joint distribution

`T_ij = softmax_ij(log alpha_i + log beta_j + s_ij)`.

Both pair contexts are `sum T_ij r_ij`. I reads entities with alpha/beta. J reads entities with row/column sums of T. A single branch in `PairReader.read` selects entity weights; the other model/data/training code is shared. Linear fusion makes J equivalent to attention over pair values augmented with endpoints; it is not a new solver or guaranteed semantic binding. Negative/same-text supervision does not make two images pixel counterfactuals.

Features: raw197×768CLIPcache; discardCLS, apply original frozen A projection+ReLU to196patches; fractional predicted-box ROI/union pooling; enclosing union includes intervening context. Top12confidence predictions, ordered pairs, 8Dsigned geometry. Null object with zero value and neutral self/null pair keys; interaction values are zero for self/null pairs. Seven fixed-dev detector records are empty according to local metadata: retain these images with null-only readout, do not drop them to improve scores.

Original A memory `[original_prompt;32QFormerQueries]` unchanged. New residual1536→512projection is zero initialized. The implementation computes `fc_out(h) + linear(delta, fc_out.weight)` with the original full-prefix projection, algebraically identical to `fc_out(h+delta)` with its bias added once. This preserves A's original projection path instead of changing layout/kernel choice by adding zero to hidden states first. Zero residual is checked exactly; teacher/student logits use atol1e-5/rtol1e-4 with measured maximum error; zero-init greedy sequences must still match exactly. New reader and last two decoder layers train; original projection/QFormer/prompt/embedding/vocabulary/firstdecoderlayer and a separate original-A decoder teacher stay frozen. Dropout remains disabled for the SCST policy and teacher-forced rescoring. PAD/BOS remain valid sampling actions as in A, so PAD is not automatically treated as a missing prefix token; trailing reference padding is excluded by loss weights.

## Data and loss

Frozen256VSRtraining pairs,484unique caption-training images;128heldout images/64pairs; fixed1000dev images. Disjoint at adaptation-image level. All1612source features/prompts staged unchanged. Only predicted boxes enter the forward. Other unlabelled assertions remain unknown. Labels/references/GT endpoint bindings are verification/supervision metadata, never input pair filtering.

500updates, seed42, global8caption images and4VSRpairs per update, split equally between2ranks. Caption references cycle deterministically through the five training references. AdamW newreader1e-4/lastdecoderlayers5e-6, wd.01, clip1. No warmup or endpoint loss. Same padded-permutation schedules between arms.

`loss = SCST + .1 KL(A || student) + .2[-ell_positive + softplus(ell_negative - ell_positive)]`.

SCST reward exactly reuses full TRAIN/restval fixed-IDF CIDErD plus .5object-count-v2reward (aliases_v2); no heldout reward labels. KL uses common COCO reference prefixes, token mean per image, stop-gradient originalAteacher. Assertion ell averages the entire predicate+object tail including intervening article, excludesEOS; predicate-only likelihood is exported as diagnosis. No negative statementCE, classifier or auxiliaryGRU. GPU gradient synchronization averages rank losses correctly; unused parameters keep grad=None.

## Checks and outcomes

Local checks: CPU/NumPy probability oracle, null/padding, fractional boxes/union, frozen320assertion spans, disjoint cohorts, deterministic schedules and fixed32pairpanel; original1000devdimensions and detectorbanks verified. No Torch/model tests run locally.

User-run preflight: full input/source SHA; actualAstate identity; zero-init logit/greedy parity; both reader arms against an independent FP64NumPy oracle; null-only/self/padding and zero-interaction cases; no-dropout sample/rescore parity; first output-only and subsequent upstream gradients; frozen scope/state; mean two-rank gradient versus serial global batch; synchronized update and exact restoration before a fresh optimizer.

Report fresh same-runtime A/end/off caption scores on the same1000images: CIDEr/BLEU/METEOR/ROUGE, CHAIRs/i/objectrecall/length, per-imageCIDEr and pairedCI; actual timings/peakVRAM; held128A/endcaptions and64tail/predicate likelihood gaps; null and prior/posterior readout diagnostics. HistoricalAcaption drift is explicit. Never insert its historical1.205512599 score instead of currentA. If identical historicalA captions yield a different evaluator score, stop evaluation. I/J analysis must also require identical currentA predictions and input content identities.

Timing/VRAM are measured within a common resident student+teacher wrapper; A timing is not a standalone-A memory benchmark. Parameter receipts separate baseline, reader and deployment totals excluding the frozen teacher. Do not use wrapper measurements to claim precise standalone inference-memory overhead; a production benchmark is separate. Detector/CLIP cache creation cost is also outside these measurements.

The semantic panel is hash-selected16pairs/stratum before training, with32pairs/64images/192A-I-Jcaptions once combined; human review remains pending until both outputs arrive. No automatic success threshold derived after seeing results. Caption superiority, semantic coverage and cost must be assessed together. A positive teacher-forced ranking or attention change alone is not caption success.

One seed and reused development/adaptation-heldout data are exploratory. Model inference on cached CLIP patches is not region-pure evidence; off-reader changes are sensitivity, not causal proof. If J>A but J≈I, no specific shared-readout benefit is established. The failedframe200gate remains unchanged; this is a new recipe from A.

## Locked semantic review after v2

Astra's [post-result decision](ASTRA_RESULT_DECISION_2026-10-28.md) keeps A and stops expansion of the current recipe. Finish the existing 32-pair panel before any new GPU diagnostic. The prior `off` condition disables pair scores/values, not entity contexts or the adapted decoder.

`make_panel.py` reads the already-decoded I/J review caches, verifies the same metadata and frozen selection, and builds `review/index.html`. The page uses 64 locally saved original COCO JPEGs with frozen predicted S/O boxes overlaid in SVG. Images and captions are ordered by fixed hashes. Arm names and positive/negative labels stay in ignored `review/private_key.json`, not the public HTML. Same-image identical captions are scored once then mapped to all originating arms: 192 source captions become 175 ratings. No lowercasing, punctuation removal or cross-image deduplication.

Open `review/index.html` manually in a browser. Fill reviewer name and the three fields per caption: assertion/paraphrase claim; mention of a relation between the target instances; and visual correctness of that relation. A relation different from the assertion is not automatically wrong. Keep uncertain cases. The page saves a local draft and exports `shared_pair_readout_blind_review.json`, including incomplete reviews; the import button restores an exported draft. Download an export before clearing browser storage or moving to another browser. No AI ratings have been inserted.

Keep the private key hidden during review. Analysis after annotation must report RC/FR, target-relation coverage, correct/wrong/uncertain counts, paired differences and uncertainty as specified in the decision, retaining the original 32 pairs. Do not silently replace the frozen panel with model-favorable examples.

Metadata test: `python experiments/shared_pair_readout/test_panel.py` (requires decoded local caches). Script syntax/13 annotation-consistency cases: `node experiments/shared_pair_readout/test_panel.cjs`. All 64 JPEGs were checked locally. Browser UI was not exercised because the browser tool rejected the file protocol; no alternate surface was used to bypass that block.

### User-requested assisted ratings

All64 images/175 unique captions have AI-proposed visual ratings in `ASSISTANT_PANEL_REVIEW.json`; see `ASSISTANT_PANEL_NOTE.md` for interpretation rules and priority adjudications. `python experiments/shared_pair_readout/make_assisted_panel.py` creates `review/assistant_review.html` with these labels prefilled. This page has175 per-caption confirmation boxes, separate browser draft storage, and assistant-pending export provenance until a named reviewer confirms all captions. Edits revoke the affected confirmation. No arm-level semantic result has been computed from these provisional ratings.
