# Stale semantic prompt: preparation, frozen scores and disposable cost profiling

Authorized preparation26 October2026, following Astra's post-A/T/W proposal. This
folder does **not** implement or launch the1250-update C/T training pilot.

## Deliverables

- `STALE_PROMPT_PREPARATION_2GPU.ipynb`: user-run Kaggle preparation. Internet/T4x2.
- `ARGON_PHAN_BIEN_HEP_2026-10-26.md`: compact standalone critical-review prompt for
  the user to send manually to external Argon. No Gemini message was sent automatically.
- Source/scripts/tests and hash-checked packaged runtime. CPU tests are synthetic;
  actual data coverage, mask quality, CLIP parity, gradients and runtime remain pending.

Inputs: exact A checkpoint, original prompt .pt and text JSON, global visual cache,
MSCOCO JSON **and image pixels**. The official COCO annotation archive is downloaded
if necessary and verified by the previously audited full SHA/size. No image downloads.

## Stages and restrictions

1. Metadata selection from locked TRAIN10k, hash-first <=2000images/oneclassperimage.
   Class must have unambiguous one-caption-token aliases, a first reference mention,
   a prompt mention retained under original20token budget, and valid masks for every
   annotated instance. Reject the whole target for any crowd/invalid segmentation.
   Union mask area1%–40% is a fixed preparation choice, not an empirically optimal range.
2. Neutralize all target mentions with `object` (repair immediate a/an); number/predicate
   cues may remain. Re-encode whole P0; original P1 remains cached and binding-checked.
   Restrict P0 to P1's actually retained character window, require that window exactly
   replays P1 token IDs/mask, and reject any P0 overflow. Never reveal previously truncated
   tail content. Store original/neutral token IDs and offsets. First noun positions are
   checked against complete whitespace tokens; reject comma/hyphen compound ambiguities.
3. Fill every target-instance mask in original pixels with RGB123/117/104 and re-encode
   CLIP197x768. No old-feature zeroing/inpainting/absence labels. Save fixed16original/
   occluded panels and text. Original-image pixel/cache parity is checked on that panel.
4. Store both I1/I0 and P1/P0 per rank, FP32. About2.6GB for2000 (factual duplicates are
   intentional for a reusable shared cache), plus~300MB benchmark inputs and metadata.
5. FrozenA class scores on fixed first128hash-selected examples, same prefix before
   target. Report factual/occluded prompt boost, visual contrast and stale hinge;
   report fraction above fixed.1stale margin, quantiles and class/mask-area strata;
   no post-score selection, no held-out/novelty/accuracy claim.
6. Fifty forward/backward batches per C/T objective with DDP/globalbatch8/auxevery5,
   reset A and RNG between arms. **No optimizer step**, verify A tensor digest is
   unchanged. AdamW state memory is estimated separately, optimizer latency unmeasured.
   This is a compute profile, not a pilot result; no weights/checkpoint are saved. Include
   sample64greedy generation timing solely to project dev1000 evaluation cost.

The30minute metadata/cache ceiling and2hour/arm projected ceiling are budget limits,
not measured ETA. Projection is1.5*(50-update time*25 + greedy64time*1000/64)+120seconds
scoring reserve. Optimizer cost, I/O, cold starts and long-tail decoding can invalidate this estimate.
Profile must remain separate from a confirmed1250-update cost. No full pilot starts
when cost gates pass. Panel inspection and Argon/Astra critique remain required.

Main CE+outputKL uses five captions/image and globally valid-token-normalized loss.
Auxiliary uses one fixed first-mention reference per image, never a synthetic caption
for I0. C andT share all four forwards/augmentations; only T adds one-sided stale loss.
CLIP image/text/vis_projection frozen; lastQFormer, promptprojection/MHA/gate/LN and
decoder have gradients enabled during the no-step profile. No SCST or new inference module.

## Output

Keep entire `stale_prompt_preparation_2000` folder for possible later dataset transport.
Send `stale_prompt_preparation_2000_review.zip`, containing summaries, manifest, panel,
logs/source but excluding HDF5 and .pt. Require receipt PREPARATION_COMPLETE; an error
ZIP is not completion. Successful preparation still waits for scientific/user review.

## Validation scope

Local model-free tests: whole-phrase matching, all target occurrences, alias ambiguity,
union of every instance, reject-any-crowd/invalid masks, four-condition score indexing.
Additional tests cover exact whitespace noun positions and excluding truncated tail context.
Notebook AST, embedded imports and full40SHA bootstrap are checked locally. User-run
synthetics verify actual COCO polygon/RLE decoding, alias probability summation and
stop-gradient direction. Actual CLIP/checkpoint forwarding occurs only on Kaggle.

Known risks: masking exposes shape or removes context, annotations miss instances,
neutral text changes grammar/sequence length or retains indirect clues, model learns
mask-color detection/global noun suppression/prompt abandonment. This preparation
does not establish a new visual-semantic alignment method or improved captions.

Review Astra: PROCEED preparation/profile only. `four_condition_interpretation.json` reports per-image factual/occluded prompt boost, their interaction, and clipped absolute alias probabilities; summary.json aggregates these. No additional model forward is required. Masks may preserve recognizable silhouette and scores miss words outside the locked alias set.
