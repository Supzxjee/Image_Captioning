# CSA10 + word–region loss: matched continuation pilot complete

04/11/2026 UTC+7. Both arms completed fixed2,000 updates/64,000 caption events,
then all5,000 Karpathy VALID images. **Small favorable point estimates, below the
predeclared improvement gates; no statistically established gain. Not TEST.**

| Metric | XE continuation | XE +0.1 grounding | GROUND−XE |
|---|---:|---:|---:|
| CIDEr ↑ |1.029875|1.035297|+0.005422|
| BLEU-4 ↑ |0.342524|0.342573|+0.000049|
| ROUGE-L ↑ |0.541193|0.541112|−0.000081|
| CHAIRs ↓ |5.96%|5.84%|−0.12 percentage points|
| CHAIRi ↓ |4.317959%|4.217288%|−0.100671 percentage points|
| Object recall ↑ |40.224761%|40.369161%|+0.144400 percentage points|
| VALID NLL ↓ |2.397916|2.399525|+0.001610|
| TRAIN online NLL ↓ |2.322722|2.324563|+0.001841|
| Online mean grounding NLL |0.600057|0.526067|−0.073990|
| Hallucinated captions |298|292|−6|
| Hallucinated mentions |308|302|−6|
| Generated object mentions |7,133|7,161|+28|
| Recalled class–image pairs |6,407|6,430|+23|
| Reference class–image pairs |15,928|15,928|0|
| Mean words |8.110|8.099|−0.011|
| Caption length <=2words |427|428|+1|

CHAIR measures object mentions against annotations, not full caption/relation
correctness. These counts are automatic, not human judgments. Online grounding
loss is averaged over training steps, not an independent held-out localization
score; its decline does not establish better causal use of visual evidence.

## Paired comparison and locked decision

10,000 image-bootstrap resamples,seed20261104. CHAIRi and recall are ratios
recomputed from summed numerators/denominators for every draw. Intervals measure
image-sampling uncertainty, not variation between training seeds.

| Difference | Paired95% interval |
|---|---|
| CIDEr |[−0.000676,+0.011299]|
| CHAIRs |[−0.400,+0.160] percentage points|
| CHAIRi |[−0.299,+0.094] percentage points|
| Recall |[−0.038,+0.320] percentage points|

All contain zero. Of838 changed captions,29 XE-hallucinated cases no longer have
a CHAIR flag, while23 previously unflagged cases acquire one.269 flagged images
are common. Net improvement is6, not29. No images were visually reviewed here.

Predeclared CIDEr gain>=.01 raw **fails** (+.005422); CHAIRs reduction>=.3pp
**fails** (−.12pp). CHAIRi/recall/short-caption safeguards pass point estimates.
This is not evidence that all grounding methods are ineffective, nor a reason to
claim success or automatically extend updates/sweep lambda/run TEST to rescue it.
Keep both endpoints. Recommend inspecting changed-case semantics and label quality
using saved text/photos before discussing another bounded experiment. No new run
or wider budget is initiated by this report.

## What was verified

81 metadata/source/count/replay assertions PASS. Both use exact source1f4a7eb
(18source hashes per arm), same CSA10 original139 checkpoint272f843a, same
ROI/vocabulary/references, label file hash and full 2,000×32 batch schedule hash.
Normalized identities are identical except arm/lambda0 versus.1. Source/receipt
evidence supports matched training; actual weights/stage binaries are not in ZIP
and were not independently loaded locally.

Logged user-run neural preflights:3 tiny tests and DDP unequal/zero-eligible-rank
gradient parity PASS; actual same-shape instrumentation maxabs6.67572e-6. Final
workers cover2,500 images/rank,286,871 global target tokens; saved NLL and CHAIR
counts/lengths/CIDEr-score means replay from text. No local model, Torch/autograd,
official scorer or fresh caption generation ran. One seed and weak labels without
manual verification remain limitations. Comparisons with prior TEST5000 metrics
must not be presented as same-cohort gains.

Worker times: XE452/449s; GROUND424/436s, including training+validation but excluding
parent setup/staging/scoring. This is not a controlled speed benchmark or quota
billing estimate. Timing does not establish that grounding is faster.

## Preserve

- XE ZIP: `d1b1bc58b6153130c31de9e914eacbee164dce5e7a523d529ff777cf68c2872a`.
- GROUND ZIP: `ce7c0d274bdf5ec7cb9866890213d4c5175acdd7dd40acc85a9285734488999e`.
- XE checkpoint: `eb5c09cce7d78f68499420a295aca3f01e752282509d62d750e5743a5fd8dbd8`.
- GROUND checkpoint: `b2b9b9928fef31cc1927ce94fbef8087d1f83d5c4b9ba4d4512d2909f928ef23`.

User reports Save Versions XE154 and GROUND51. Exact notebook-owner mapping was
not provided in that reply; confirm before any version-pinned download rather
than guessing. Preserve entire `csa_grounding_pilot_XE_v2` and
`csa_grounding_pilot_GROUND_v2` output folders; checkpoint binaries are outside
the review ZIPs. Machine-readable comparison/examples: GROUNDING_PILOT_RESULT.json.
