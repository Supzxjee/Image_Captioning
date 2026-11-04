# TRAIN30k matched XE/GROUND continuation — complete, small unconfirmed gain

04/11/2026 UTC+7. Both fixed2,000updates/64,000events from original CSA10; all30k
images exposed,26k twice/4k three times. Same round-robin schedule, labels, inputs,
initial weights, optimizer and attention instrumentation; lambda0 versus.1.
Evaluation: same5,000 Karpathy VALID images, beam5/max20. **Not TEST.**

| Metric | XE | XE+grounding | GROUND−XE |
|---|---:|---:|---:|
| CIDEr ↑ |1.038046|1.042326|+0.004280|
| BLEU-4 ↑ |0.343666|0.342629|−0.001037|
| ROUGE-L ↑ |0.540325|0.540771|+0.000446|
| CHAIRs ↓ |5.66%|5.56%|−0.10 percentage points|
| CHAIRi ↓ |4.079910%|4.016854%|−0.063056 percentage points|
| Object recall ↑ |40.187092%|40.249874%|+0.062783 percentage points|
| VALID NLL ↓ |2.396445|2.397578|+0.001134|
| TRAIN online NLL ↓ |2.327211|2.329002|+0.001791|
| Online mean grounding NLL |0.596709|0.526470|−0.070240|
| Hallucinated captions |283|278|−5|
| Hallucinated mentions |290|286|−4|
| Generated object mentions |7,108|7,120|+12|
| Recalled class–image pairs |6,401|6,411|+10|
| Reference class–image pairs |15,928|15,928|0|
| Mean words |8.1200|8.1042|−0.0158|
| Captions <=2words |431|429|−2|

Grounding changes866captions. It clears CHAIR flags on21images but adds flags
on16others, a net−5;262 flagged images are shared. These are automatic object
annotation counts, not visual/human judgments or relation accuracy. The online
training grounding loss is not a held-out alignment score or proof of causal
visual grounding. No saved images were visually judged in this comparison.

## Paired uncertainty and predeclared screening gates

10,000 paired image-bootstrap resamples,seed20261104; ratio numerators and
denominators reaggregated for each draw.95% intervals for GROUND−XE:

- CIDEr: [−0.001908,+0.010306].
- CHAIRs: [−0.340,+0.140] percentage points.
- CHAIRi: [−0.233,+0.107] percentage points.
- Recall: [−0.119,+0.239] percentage points.

All include zero. CIDEr+.01 and CHAIRs−.3pp minimum gains fail; CHAIRi/recall/
short-caption safeguards pass point estimates. One seed/reusedVALID/unreviewed
weak labels remain limitations. This is not evidence of zero effect, but does
not justify calling grounding effective or automatically scaling the same recipe.

## What did expansion change?

Same-reference VALID comparison, descriptive only:

| Study | XE CIDEr | GROUND CIDEr | Grounding advantage | XE/GROUND flagged captions |
|---|---:|---:|---:|---|
|10k grouped captions |1.029875|1.035297|+0.005422|298/292|
|30k round-robin |1.038046|1.042326|+0.004280|283/278|

Both arms have higher CIDEr and fewer flags than their10k counterparts. Grounding's
increment over its matched XE control is not larger in point estimates. Because
BOTH pool size and sampler changed, this is not a controlled causal claim that
more images alone caused the improvement, and not replication across seeds.

**Recommendation:** preserve endpoints and stop automatic image/update expansion
of this loss configuration. Before another GPU experiment, inspect the weak-label
semantics and changed cases/decoder dependence. Do not use TEST or lambda sweeps
to rescue a missed pilot gate. A modified mechanism requires a new locked protocol.

## Verification/provenance

100 metadata/source/count/sample-score assertions PASS. Both use source85004b2
(23source pins per arm), original139latest272f843a checkpoint, equal normalized
identities except arm/weight, labelSHA2d488980 and batchSHA652c47da, same stage
feature/offset/ID/sequence hashes and identical VALID references/vocabulary.
Stage reports30k unique exposures with26k2x/4k3x,150076reference pool,
158197eligible labels. Actual exposure contributes733666 target tokens and67380
eligible mention occurrences. User-run neural/DDP preflights and same-shape
instrumentation pass. Each rank scores2500VALIDimages; NLL totals286871targets.
Counts, lengths and CIDEr per-image score means replay from stored text.

Full stage/weights binaries are not in review ZIPs; source/receipts support identity
but were not independently loaded locally. No agent Torch/model/autograd, metric
scorer or new caption generation ran. Worker times XE468/459s and GROUND466/466s
include train+validation, exclude parent setup/staging/scoring; not quota estimates
or a controlled speed comparison.

- XE ZIP: `a5c7e4e2e42460b0d82c8e1af0cd81a2dc56a2a534d8276d5b623c135f9db16e`.
- GROUND ZIP: `c96af03c9a3913abc510578f931ee3f2209f7d6c53d371c30dddf940d850a9cb`.
- XE checkpoint: `728a71d598512df13829b483af719e2bbf215fd3a1cb76142ba8575516e9852f`.
- GROUND checkpoint: `b1000f1f3fd5511fc20f4ddcbe3c92881813f630b03a5a925c6b6e37aeed9db6`.

Preserve entire `csa_grounding_30k_XE_v1`/`csa_grounding_30k_GROUND_v1` outputs;
Save Versions/notebook sources requested, pending. Do not infer exact source
versions from earlier154/51; those belong to the10k study.
