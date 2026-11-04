# CSA15 paired SCST pilot — VALID5000

Reviewed04/11/2026. Both user-run arms COMPLETE at fixed500updates/global8=4000uniqueTRAIN image events (from10kpool), then fullVALID5000. Same epoch15 Save147 weights, freshAdam5e-6, same stage/schedule/reward labels/vocab/references/source, FP32/dropoutOFF, beam5/max20. Only object reward coefficient differs0/.5. CSA alignment architecture unchanged; this is post-XE sequence training, not new alignment. [Protocol](SCST_PILOT_PROTOCOL.md), [numeric audit](SCST_PILOT_RESULT.json).

## Result

| Metric | CSA15 before SCST | SCST+CE | SCST+CE+object |
|---|---:|---:|---:|
| CIDEr ×100 ↑ |100.9453|**102.3465**|102.1517|
| BLEU4 ↑ |.332274|**.337689**|.337169|
| ROUGE-L ↑ |.535130|**.536761**|.536433|
| CHAIRs ↓ |5.86%|5.82%|**5.48%**|
| Hallucinated captions ↓ |293|291|**274**|
| CHAIRi ↓ |4.4410%|4.3238%|**4.0431%**|
| False object mentions ↓ |315|305|**285**|
| Object recall ↑ |**40.1180%**|39.9987%|39.9799%|
| VALID NLL ↓ |2.405520|**2.388992**|2.389700|
| Mean words |8.2212|8.2618|8.2870|
| Captions≤2words |425|424|419|

Full recipe versus CSA15: CIDEr+1.2064 publication-scale points, CHAIRs−.38percentage points/net19 fewer flagged captions, CHAIRi−.3979pp, recall−.1381pp. These are modest positive point estimates, not ten CIDEr points. Words/short captions do not indicate avoiding hallucination simply by shortening every caption. SCST+CE alone increases CIDEr slightly more but hardly changes flagged-caption count (net2).

Matched OBJECT minus CE: CIDEr−.0019483 raw (−.1948points); CHAIRs−.0034 (−.34pp), CHAIRi−.0028066 (−.2807pp), recall−.00018835 (−.0188pp). 598captions change;25CE-flagged captions repaired,8new flags introduced, net17fewer. This is the direct object-term comparison absent from the historical QFormer final test; it supports reduced COCO object hallucination for this pilot, not correctness of every caption/relation.

## Paired uncertainty

10000 image-bootstrap resamples, seed20261104, ratios recomputed from sampled numerators/denominators:

| OBJECT minus CE | Point | Conditional95% interval |
|---|---:|---:|
| CIDEr raw |−.001948|[−.007153,+.003237]|
| CHAIRs pp |−.34|[−.58,−.12]|
| CHAIRi pp |−.2807|[−.4576,−.1158]|
| Recall pp |−.0188|[−.1766,+.1377]|

On this fixed validation sample, the hallucination reduction has stronger support than a change in CIDEr/recall. A CIDEr interval crossing0 does NOT prove equivalence/noninferiority. These conditional intervals do not incorporate seed variance or repeated validation selection; no multiplicity correction or finalTEST claim.

Versus CSA15, OBJECT changes1556captions, repairs61flags and introduces42. CHAIRsdelta95CI[−.78,0]pp touches0; CHAIRideltaCI[−.7039,−.1021]pp. Baseline per-imageCIDEr unavailable, so no paired baselineCIDEr interval was fabricated. Before-SCST reference bytes/IDs match currentVALID; its predictions are historical saved outputs, not regenerated here.

## Verification and preservation

118metadata/source/count checksPASS,20Git source pins per arm at ff9a3b64705fdafb10897762b3013b36c337cee0. ZIPCRC PASS; noZIP reread after cached text. Normalized complete identities equal except arm/object_weight; reconstructed scheduleCOCOIDNPY hash matches. Same full5000references, vocab, trainingtargets/refs/events/stage hashes; noTRAINpool/VALID overlap. Token-to-caption mapping, CHAIRcount arithmetic, NLL weighted totals and per-imageCIDEr mean verified. Official metrics NOT rerun locally; binaries/targets absent from reviewZIP, so receipt claims do not imply independent binary inspection.

User-run logs:5data checks+3neural checks and twoGPU DDP gradient parityPASS for both arms. Exact loaded-model policy replay maxabs6.20e−6 and2.62e−6; no dropout. Both Torch2.11.0+cu128. Workers458/465sec CE and563/558sec OBJECT include reward initialization/training/VALID, exclude notebooksetup/CPUmetrics; not GPU quota/billing estimates. Full notebook contains successful pipeline; assistant ran no model/scorer/Torch.

Preserve entire outputs, particularly checkpoint.bin + optimizer/RNG/identity + receipt:

- CE: `/kaggle/working/csa15_scst_ce_pilot500/checkpoint.bin`, SHA daef7f333ff6ace7eb46ee0baf20e15fa96554a60d7b6eac7b20f0c08d43077b.
- OBJECT: `/kaggle/working/csa15_scst_ce_object_pilot500/checkpoint.bin`, SHA b2795aa97e882bd340a31596f57c29607008a93ab7268644cac7cbc542f7b707.
- Source notebook SaveVersions pending user reply. These SHAvalues are from runtime receipts; checkpoints are not in smallZIPs.

## Recommendation

Treat CSA+SCST+CE+object as the preferred joint-quality candidate; keep SCST+CE as matched control. Preserve both500 endpoints before changing notebooks. Discuss one bounded extension with matched budget/sampler and measured runtime; do not silently change500endpoint, discard the control or launch30RLepochs. Current sampler covers4000images only (~3.53% of full113287-image RL epoch), so this pilot does not establish convergence. No further training/TEST initiated or authorized by receiving the ZIPs. The original CSA alignment remains; no new grounding module is needed to interpret this result.
