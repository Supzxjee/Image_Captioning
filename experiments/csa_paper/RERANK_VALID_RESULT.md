# Fixed reranking improves the CSA161 pipeline on VALID5000

Reviewed05/11/2026. User run COMPLETE; actual inference-only selection, no oracle in selector. [Protocol](RERANK_VALID_PROTOCOL.md), [numeric audit](RERANK_VALID_RESULT.json), [audit script](audit_rerank_result.py). User sourcef72d54d; candidate pool exactSaveVersion163, captioner weights161 unchanged/not loaded in this run. ZIP SHA883954a53f5aeef78d096ba08dcec02cdac5c8996abc1c0bf1600300c3ca07b0; CRC checked once, text cached. User confirms reranking output SaveVersion164: `ducanh2403/2-vi-cai-kia-qua-nhieu/164`, folder `/kaggle/working/csa_ce161_rerank_valid5000`. This user-reported source is distinct from candidates163 and weights161; API transport has not been independently replayed.

**The old fixed CLIP+object selector is the preferred current main policy.** It raises CIDEr106.31→115.24 and reduces CHAIR flags260→206 on the same full5000VALID set and same25000candidates, while recall increases. BLIP ITM also improves over decoder selection, but its CIDEr is lower than the fixed mixture and its recall higher. Keep BLIP as a comparator, not a mandatory extra stage in the chosen pipeline. No finalTEST result or new-alignment/novelty claim.

## Same frozen captioner and candidates

| Metric | Decoder rank0 | Fixed CLIP+object | BLIP ITM |
|---|---:|---:|---:|
| CIDEr ×100 ↑ |106.3103|**115.2416**|113.0610|
| BLEU4 ↑ |.348294|**.366623**|.357266|
| ROUGE-L ↑ |.544929|**.566730**|.562042|
| CHAIRs ↓ |5.20%|**4.12%**|4.36%|
| CHAIRi ↓ |3.7369%|**2.7366%**|2.7970%|
| Object recall ↑ |40.5387%|43.9227%|**44.5379%**|
| Hallucinated captions ↓ |260|**206**|218|
| False object mentions ↓ |267|**216**|226|
| Recovered class/image pairs ↑ |6457|6996|**7094**|
| Mean words |8.4164|9.2676|9.8736|
| Captions<=2words ↓ |407|**60**|111|

Fixed mix uses exactly .3minmax(raw decoder logprob)+.4minmax(CLIPScore)+.3normalized detector consistency, historical class parser/conf>=.5/penalty1; no coefficient search. BLIP uses positive-minus-negative ITM margin argmax, earlier rank on ties. References/CHAIR labels are used only after selection. No teacher-forced NLL comparison: captioner weights are identical.

## Paired evidence

Fixed minus rank0: **+8.9313CIDEr points**, CHAIRs−1.08percentage points, CHAIRi−1.0003pp, recall+3.3840pp.3027captions change;89flags repaired and35new introduced, net54fewer. Correct object coverage rises, so lower hallucination is not explained by suppressing all object mentions. Still not human confirmation or proof of all relation/attribute correctness.

BLIP minus rank0: +6.7507CIDErpoints, CHAIRs−.84pp, CHAIRi−.9398pp, recall+3.9992pp.4207captions change;102repairs/60new/net42. Fixed policy retains rank0 on1973images; BLIP only793, often choosing lower beams. Fixed ranks0/1/2/3/4 =1973/1357/885/585/200; BLIP793/919/941/1060/1287.

Paired image bootstrap10000resamples, seed20261105; recompute aggregate count ratios each sample. Conditional95% intervals, unadjusted for repeatedVALID use/policy choice and training seeds:

| Difference | CIDEr points | CHAIRs pp | CHAIRi pp | Recall pp |
|---|---|---|---|---|
| Fixed − decoder |+8.9313 [7.6509,10.2239]|−1.08 [−1.52,−.64]|−1.0003 [−1.3001,−.7055]|+3.3840 [3.0218,3.7653]|
| BLIP − decoder |+6.7507 [5.2810,8.1812]|−.84 [−1.34,−.34]|−.9398 [−1.2850,−.6047]|+3.9992 [3.5748,4.4199]|
| Fixed − BLIP |+2.1806 [.9948,3.3760]|−.24 [−.64,.14]|−.0604 [−.3240,.1985]|−.6153 [−.9576,−.2564]|

Both selectors improve all four main measures versus decoder under this conditional comparison. Fixed has supported CIDEr advantage versus BLIP and a recall disadvantage; its apparent CHAIR advantage versus BLIP remains uncertain. Do not say fixed wins every metric or that one-seed VALID proves generalization.

## Whole recipe context

SameVALID reference set: CSA15 XE CIDEr100.9453/293CHAIR flags → CSA15+SCST+CE1epoch+fixedrerank CIDEr115.2416/206flags, **+14.2963CIDEr points and87fewer flags**. This includes BOTH SCST training and output selection. The isolated reranking gain remains+8.9313points; neither increment is a new CSA alignment module. Original CSA alignment is inherited. No comparison against external121.95 is established because its protocol/split is unspecified.

Reference-dependent best-of-five oracle141.4344 from R246 is not a deployable policy; the current actual115.2416 is separate. No target-reaching guarantee by further optimizing the remaining oracle gap.

## Verification, preservation and limits

78metadata/text/count/selector checksPASS, including13Git source pins, exact163candidate hashes/full25000token-text-logprob validation,5000raw-imagefilename/ID/size/SHA metadata joins, identical references/vocabulary/top1metric/per-imageCIDEr replay, inferencepayload field boundary(no refs/oracle/CIDEr), modelrevision/filehash receipts, fulluntruncated text coverage(CLIP27/77tokens;BLIP38/512), worker2500/12500pair coverage/fileSHA, all5000selector replays, ITM probability replay(max8.97e-8), three prediction membership/CHAIRcount/length/CIDErmean/rankhistogram/pairedrepair checks. User logs6safe tests/exact163APIrouting/mock/baseline5000metricpreflight/isolatedHFdependenciesPASS; no traceback. Models frozenFP32/Torch2.11+cu128; workers936.2/929.0s(~15.6min), excluding setup/modeldownloads/staging/CPUmetrics, not billed GPU hours.

No author Torch/model/scorer execution. Official BLEU/ROUGE and per-imageCIDEr are stored user-run values; means/identities/counts/selector logic replayed, not official scorers rerun. Images, original detector source and model binaries are absent ZIP; recorded SHA/forward results are user-run evidence. BLIP is externally COCO-finetuned with split provenance unverified here. Detector absence is not true absence; detector topK/threshold provenance remains unknown. New CLIP scores use official JPEG preprocessing rather than historical cached CLS; reused formula is not exact old-system score reproduction.

Preserve entire `/kaggle/working/csa_ce161_rerank_valid5000` including rawscoreJSONL/selectedranks/inferenceinputs/modelsreceipt and captionmetrics. No new checkpoint was trained. Next recommendation: freeze the fixed mixture and evaluate the SAME captioner with/without fixed reranking on KarpathyTEST5000 under a separately approved protocol; retain existing CSA/NoCSA10 test comparison as a different training-budget experiment. No automatic TEST, new coefficients, additional training or BLIP fusion authorized by this result.
