# Root review — new information from native-image tiles

2026-10-23. **Candidate worth a resource/preprocessing feasibility check; not a demonstrated
gain or a new alignment method. No cache extraction, training code or model run started.**

## What Astra changed

Keep the negative G/S conclusion and frozen A as reference. The new question is whether
native-image detail, lost before the existing224×224 encoding, can help caption generation.
This does not assume that resolution is the proven bottleneck.

One shared adapter sees both conditions during TRAIN. H crops four quadrants from the
native image, then resizes each to224; L first resizes the entire image to224, then crops
the same quadrants and enlarges each to224. Both use the same frozen CLIP, four views,
positions,196pooled tokens and adapter weights. The A global image and legacy prompt stay.

Frozen A queries32×512 retrieve tile evidence196×512 through128D/four-head attention;
the residual updates the32queries. Recompute the existing grounded legacy prompt from
the updated queries before decoding. Output projection starts at zero.

TRAIN adapter only: five natural captions/image, global-token-normalized CE with EOS,
no new SCST/object/presence/bbox/contrastive loss. Four epochs/5000updates,batch8,seed42;
each TRAIN image gets H twice/L twice on a prelocked caption-independent balanced schedule.
Endpoint rawgreedy A/L/H on reusedval5000. Keep original G/S stopped.

## Critical assessment

* **A real change of information source:** H has access to native image detail that
  cannot be reconstructed by routing the old patch cache. H/L under one trained adapter
  avoids confounding two separate optimization histories. The full-scene A path remains,
  unlike replacing the entire input with a selected pair crop.
* **Positive results have a limited interpretation:** H beating L supports the utility
  of this native-detail processing path, not verified word/box or relation grounding.
  H and L share geometry/capacity but their blur/resampling and CLIP feature statistics
  differ; do not attribute individual caption fixes to resolution without further evidence.
* **Novelty is unresolved:** multi-view features, cross-attention and zero-initialized
  residual adapters are inherited ideas. Full-caption CE provides weak generation-linked
  supervision, not explicit lexical/relation alignment labels. A gain could justify
  further research; it cannot alone establish a new scientific alignment contribution.
* **A negative result is also bounded:** average-pooling14×14 to7×7 can discard useful
  detail, fixed quadrants can cut objects, and shared H/L training may favor their common
  information. Failure would stop this configuration/budget, not disprove high-resolution
  encoding or show that A's remaining errors are inherently linguistic.

## Implementation boundaries checked against the current model source

The current encoder really uses ReLU(vis_projection(CLIP tokens)), QFormer, then legacy
prompt-to-visual attention/gate/norm. Astra's injection point is compatible with that order.
Important preflight requirements if implementation is requested:

1. Keep the original global cache and prompt input identical; at zero output projection,
   A/L/H must recover A on the same batch. Add the residual **before** recomputing the
   grounded legacy tokens, not after reusing cached grounded prompt outputs.
2. Freeze A weights but keep autograd through its downstream legacy attention and decoder
   to the adapter. The `no_grad` decoder wrapper used for G/S is **not reusable here**.
3. Explicit bias-free q/k/v/output matrices have262656parameters:
   512×128 +2×514×128 +128×512. A default MultiheadAttention with additional internal
   projections would silently exceed this architecture/count; use the specified projected
   four-head attention, or explicitly revise/count the design before running.
4. Pool CLIP patches before frozen projection/ReLU. Match CLIP revision, RGB conversion,
   normalization, square bilinear/antialias preprocessing and the H/L crop coordinates.
   Do not silently switch to a library's default center crop. Verify low-resolution identity
   controls and odd image dimensions before encoding120000views.
5. Check head gradients, absence of A weight updates, global token CE normalization across
   both GPU ranks and the exact per-image H/L exposure schedule. Dataset split/image IDs
   must remain auditable; no content/error/GT-dependent tile selection or inference inputs.

## Resource and decision limits

120000tile encodes for train10k+val5000, two conditions/four tiles. FP16 cache payload is
15000×196×512×2bytes=3010560000bytes/condition,6021120000bytes total, before HDF5/metadata.
Check actual source-image availability, cache/stage disk peak and remaining Kaggle GPU
quota. Existing staging has sometimes dominated runtime; do not promise total GPU-hours
from update count alone or duplicate full FP32 copies unnecessarily.

Future gates proposed by Astra (not applied retrospectively to G/S): H−A≥.01 and positive
paired95lower; H−L≥.005 and positive lower; CHAIRs/i≤A; Recall≥A−.005. H>L without H>A is
not a successful captioner; H/L>A without H>L is not native-detail attribution. No test,
no post-hoc coefficient/tile sweep. This remains exploratory reused validation/one seed.

Recommended next decision: establish resource/preprocessing feasibility before committing
to this pilot. No new notebook or automatic run follows merely from receiving the proposal.
