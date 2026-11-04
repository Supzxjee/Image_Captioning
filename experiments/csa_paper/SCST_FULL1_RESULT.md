# CSA15 full-TRAIN SCST one epoch — COMPLETE

Reviewed05/11/2026. [Protocol](SCST_FULL1_PROTOCOL.md), [numeric audit](SCST_FULL1_RESULT.json). Both user runs completed14161updates/113287uniqueTRAINimages once, no duplicate padding; final7images split4/3. Same originalCSA15 weights/freshAdam5e-6/global8/FP32/dropoutOFF, objectweight0/.5 only. ORIGINAL/MID/END all evaluated on fullVALID5000; noTEST. Both chooseEND under the predeclared CIDEr/CHAIR/recall rule; all3points eligible, nofallback. User reports SaveVersions CE161/OBJECT55; notebook-owner mapping not repeated, confirm before API transport.

## Final comparison, same VALID5000

| Metric | FreshCSA15 original | SCST+CE END | SCST+CE+object END |
|---|---:|---:|---:|
| CIDEr ×100 ↑ |100.9453|**106.3103**|105.6063|
| BLEU4 ↑ |.332274|**.348294**|.344926|
| ROUGE-L ↑ |.535130|**.544929**|.542346|
| CHAIRs ↓ |5.86%|5.20%|**5.12%**|
| Hallucinated captions ↓ |293|260|**256**|
| CHAIRi ↓ |4.4410%|3.7369%|**3.7162%**|
| False object mentions ↓ |315|267|**265**|
| Object recall ↑ |40.1180%|40.5387%|**40.6768%**|
| Recovered class/image pairs ↑ |6390|6457|**6479**|
| VALID NLL ↓ |2.405520|2.365264|**2.364907**|
| Mean words |8.2212|8.4164|8.3886|
| Captions≤2words |425|407|419|

CE versus own ORIGINAL: CIDEr+5.3650publicationpoints, CHAIRs−.66pp/net33flags, recall+.4206pp. OBJECT versus own ORIGINAL: CIDEr+4.6611points, CHAIRs−.74pp/net37flags, recall+.5588pp. Both improve quality/hallucination/coverage on thisVALIDsample; no newclaim of causal alignment correctness or generalization. Neither shows a10pointCIDEr gain over CSA15. Modelarchitecture remains inheritedCSA; this experiment modifies training only.

## Object reward: not the preferred main recipe now

OBJECT minus CE: CIDEr−.00703948raw (−.70395publicationpoints), CHAIRs−.08pp/net4flags, CHAIRi−.0207pp, recall+.1381pp.1064captions change;25CEflags repaired and21new introduced.

Conditional paired10000image-bootstrap95% intervals (seed20261105): CIDEr[−.013619,−.000531]raw, CHAIRs[−.34,+.18]pp, CHAIRi[−.2053,+.1681]pp, recall[−.0625,+.3457]pp. CIDEr difference favors CE in this conditional comparison; the small hallucination/recall differences remain uncertain. These unadjusted intervals omit training-seed variance, repeatedvalidation use and checkpoint-selection variability.

**Recommendation changes with evidence:** chooseSCST+CE as the main current candidate, retain OBJECT as an ablation. The500pilot's stronger object-specificCHAIRsignal weakened at2500 and full1. Do not persist with OBJECT solely because the research originally expected it to help; no demonstrated object-term benefit justifies a blind penalty increase or committing5epochs to it.

Both fullrecipes versus ORIGINAL have conditional intervals excluding0 on all4measures: CE CIDEr[+.041480,+.066077],CHAIRs[−1.20,−.14]pp,CHAIRi[−1.1059,−.3116]pp,recall[+.0758,+.7594]pp; OBJECT CIDEr[+.034285,+.059060],CHAIRs[−1.26,−.22]pp,CHAIRi[−1.1161,−.3372]pp,recall[+.2181,+.9034]pp. This supports joint improvement for this fixed sample/seed, not finalTEST performance.

## Midpoint and earlier pilot

| Arm | MID7081 CIDEr / CHAIRs / recall | END14161 CIDEr / CHAIRs / recall |
|---|---:|---:|
| CE |105.3645 /5.42% /39.9611%|106.3103 /5.20% /40.5387%|
| OBJECT |105.4699 /5.14% /40.1745%|105.6063 /5.12% /40.6768%|

CE MID→END CIDEr+.009457raw,95CI[+.000591,+.018503]; OBJECT+.001364,CI[−.007447,+.010065]. OBJECT flags257→256 only. Both recall increases have positive conditional intervals. A guaranteed plateau or continued monotonic gains beyond1epoch is not established.

Relative to2500study, CE CIDEr103.4778→106.3103 butflags259→260; OBJECT103.2307→105.6063 butflags251→256. Both recover more objects. FullTRAIN training is not better on every metric than the small-pool checkpoint; updates AND imagecoverage change. Preserve2500as history, do not silently choose it under the full1original/MID/END rule. Comparison to oldQFormer requires commoncohort/scoring; historicalQFormerobject fullVALIDCHAIR3.88%/recall43.53% still stronger objectmetrics than eithercurrentCSA. No directCIDEr subtraction across differentcohorts.

## Verification and preserved outputs

197metadata/source/count checksPASS,25Gitpins/arm atc4b83f185a1bcd4facfc75f7d5bde6dc139a7b0e. ZIPCRC verified once and textcache used thereafter. Identical normalized fullidentities exceptarm/objectweight, fullID schedulehash reproduced,113287×1coverage/last7/heldoutdisjoint confirmed bymetadata. PhasefileSHAs, rootreference/vocabSHAs,567trainingrecords(last14161/113287), sealedtrainingloghash and checkpointselection arithmetic checked. Six complete5000caption sets have token-captionmapping/CHAIRaggregates/NLLdenominator286871/per-imageCIDErmean checks. Fresh ORIGINAL metrics AND generatedcaptions identical across arms. MID/ENDreceipts/selectedSHA consistent; binaries/actualobjecttargets absent reviewZIP, so local independentweight/label verification is not claimed.

Userlogs:5+4+6safechecks,3neuralchecks, standardDDP andnewunequal4/3 CEtoken+SCSTimage gradientparityPASS. No agentTorch/model/scorer execution. Maxworkers CE5802.2sec(~96.7min),OBJECT5997.1sec(~100min), includestraining/rewardinit/threeVALIDdecodes, excludesnotebooksetup/CPUmetrics; not quota/billing estimate.

- SaveVersion161 CE END latest/best: `/kaggle/working/csa15_scst_ce_full1/best.bin`, SHA58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b.
- SaveVersion55 OBJECT END latest/best: `/kaggle/working/csa15_scst_ce_object_full1/best.bin`, SHAa98487d77a6476f782565e53cea4660c3bf9c85034047a8bef6680d80e42fb0b.
- Keep wholeoutputs includingmid.bin/checkpoint.bin/optimizer/RNG/receipts/locked_selection and phasecaptions. No newDataset is required merely to recordversions; verifybinarySHA on any later transport.

## Next decision

Freeze both1epochendpoints and selection records. Maincandidate CE, OBJECT comparator; propose matchedKarpathyTEST evaluation againstCSA15 before claiming a finalpipelinegain, or a separately approved bounded CEextension if moretraining is desired. No automaticTEST/extraepoch/lambdatuning/architecturechange is authorized by receiving theseZIPs. Research scope remains effective visual-semantic alignment viaCSA, with no user requirement for method novelty.
