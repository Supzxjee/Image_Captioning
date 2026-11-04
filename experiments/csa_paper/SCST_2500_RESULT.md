# CSA15 paired SCST2500 — COMPLETE on VALID5000

Reviewed04/11/2026. [Protocol](SCST_2500_PROTOCOL.md), [numeric audit](SCST_2500_RESULT.json). Both fresh from original CSA15 Save147e02593ea, NOT resuming pilot500 optimizer/weights. Same initialization/data/source/schedule/reward labels except object coefficient0/.5; freshAdam5e-6, global8/T4x2, FP32/dropoutOFF. Actual2500updates=20000TRAINimage events, same10kpool2visits/image; fullVALID5000 beam5/max20, noTEST or mid-checkpoint selection. Architecture still inheritedCSA; no new grounding loss.

## Same-VALID quality

| Metric | CSA15 before SCST | SCST+CE2500 | SCST+CE+object2500 |
|---|---:|---:|---:|
| CIDEr ×100 ↑ |100.9453|**103.4778**|103.2307|
| BLEU4 ↑ |.332274|.340888|**.341330**|
| ROUGE-L ↑ |.535130|.540300|**.540322**|
| CHAIRs ↓ |5.86%|5.18%|**5.02%**|
| Hallucinated captions ↓ |293|259|**251**|
| CHAIRi ↓ |4.4410%|3.8274%|**3.7080%**|
| False object mentions ↓ |315|267|**258**|
| Object recall ↑ |**40.1180%**|39.8669%|39.8104%|
| VALID NLL ↓ |2.405520|2.375194|**2.374963**|
| Mean words |8.2212|8.2784|8.2840|
| Captions≤2words |425|418|414|

OBJECT versus CSA15: CIDEr+2.2855publicationpoints, CHAIRs−.84pp/net42fewer flaggedcaptions (~14.3% relative), CHAIRi−.7330pp/57fewerfalsementions; recall−.3076pp (6390→6341correctclass/image pairs). Meanlength does not fall, so reduction is not simply global caption shortening, but small loss of objectcoverage remains a tradeoff. CE also improves CIDEr+2.5326points and CHAIRs−.68pp/net34flags.

## Does the object term help beyond SCST+CE?

OBJECT minus matched CE: CIDEr−.00247086raw (−.2471points), CHAIRs−.16pp/net8flags, CHAIRi−.1194pp, recall−.0565pp. 865captions change;26CEflags repaired/18newintroduced. Conditional paired10000image-bootstrap95% intervals:

| Difference OBJECT−CE |95% interval |
|---|---:|
| CIDEr raw |[−.008565,+.003622]|
| CHAIRs pp |[−.42,+.10]|
| CHAIRi pp |[−.3103,+.0712]|
| Recall pp |[−.2358,+.1239]|

All four intervals contain0. Do not claim isolated object-reward benefit established at2500. Earlier500study had a stronger conditional CHAIRsignal; the narrowed gap changes our confidence, not the direction of the point estimate. Both runs share initialseed/pool/prefix, so they are NOT independent-seed replications.

Versus CSA15, OBJECT CHAIRsdeltaCI[−1.30,−.38]pp and CHAIRideltaCI[−1.0865,−.3846]pp exclude0; CE also improves CHAIRs withCI[−1.14,−.22]pp. Thus fullrecipe dehallucination has evidence on thisVALID sample, while attributing that improvement entirely to objectreward is unsupported. Baseline per-imageCIDEr absent: no baselineCIDErCI invented.

## Longer learning versus pilot500

| Arm | CIDEr ×100,500→2500 | CHAIRflags,500→2500 | Recall%,500→2500 |
|---|---:|---:|---:|
| SCST+CE |102.3465→103.4778|291→259|39.9987→39.8669|
| SCST+CE+object |102.1517→103.2307|274→251|39.9799→39.8104|

OBJECT500→2500: CIDEr+.01079045raw, conditionalCI[+.001631,+.019802]; CHAIRs−.46pp,CI[−.82,−.10];53repairs/30newflags. CE500→2500: CIDEr+.01131302raw,CI[+.002110,+.020724];CHAIRs−.64pp,CI[−1.02,−.26];64repairs/32newflags. Both improve. Totalupdates AND exposurecoverage change (4000unique→10000unique,2passes); not a pure time-only ablation. No evidence yet of convergence or monotonic improvement beyond2500.

## Verification and preserved artifacts

134metadata/source/count checksPASS;22Gitpins/arm sourcefb75bc6accompanying runtime receipts; ZIPCRC verified once then cachedtext used. Full identities normalized equal exceptarm/objectweight; stages/refs/vocab/events/objecttargets hashes equal; reconstructed20000IDschedule hash matches and exact10000×2 exposurehistogram. Full5000ID/rankpartition, token-captionmapping, CHAIRcount aggregation, NLL denominator286871 and per-imageCIDEr means checked. Traininglog100entries/25stepintervals/20kevents finite. Userlogs show5data+4newdata+3neural tests andDDPgradient parityPASS; model/policy-replay checkspassed. Agent did not runTorch/model/scorer/officialmetric recomputation. Actualcheckpoint/label/stagebinaries absent reviewZIP; byteSHAvaluesbelow are runtime-reported, not independently rehashed locally.

- CE ZIP SHA294d47362f972c0a385fa7eb108e3ba7354f34bdc16acd28cead1f411e60f9c7.
- OBJECT ZIP SHAb095e24c62ad0cf706389f1f42fa709fae2cdbbdf5572c986e6a953de518f5dc.
- CE checkpoint `/kaggle/working/csa15_scst_ce_pilot2500/checkpoint.bin`, SHAbf429b5de9b2adf6d694e4edd9f39a5c1828a91854f0138bb2333a763bd62f35.
- OBJECT checkpoint `/kaggle/working/csa15_scst_ce_object_pilot2500/checkpoint.bin`, SHAe12ca68fa6761d17a1e6100521630052cc8421f978745d60106bfdb610ef0de9.
- SaveVersions requested, pending userreply; user prefers keeping version outputs without anotherDataset.
- Maxworker1265.9secCE/1325.9secOBJECT (~21.1/22.1min), includesrewardinit/training/VALID, excludesnotebooksetup/CPUmetrics; not quota/billing estimate.

## Recommendation and limits

Keep both2500endpoints. OBJECT remains the lowest-CHAIR pointestimate candidate of these three; CE has slightlyhigherCIDEr and needs to remain the control. The next discussion can consider one broader-TRAIN budgeted study with samecoefficients, not increasingpenalty blindly or droppingCEcontrol. No furthertraining/TEST is authorized by receipt of theseZIPs. Before choosing a finalmodel, weighCIDEr/CHAIR/recall jointly. One seed/reusedVALID/COCO80classCHAIR; no guarantee of relationcorrectness or testgeneralization. Research goal is an effective inheritedalignment pipeline, not a novelty claim.
