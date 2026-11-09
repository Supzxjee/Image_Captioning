# CLIP patch baseline full10XE — COMPLETE, version199

User confirmed SaveVersion199; owner/notebook inherited from current task `ducanh2403/2-vi-cai-kia-qua-nhieu`. Sourcepin9aaddf5, archiveSHA in LOCAL_V3_RESULT.json. Metadata/JSONseal/source/IDs/counts/timing/LR audit **1928 assertions PASS**. This is an artifact audit, not1928 independent experiments; no localTorch/checkpoint loading/officialscorer replay. Captionfeaturepayloads/checkpointbinary are not inside reviewZIP. Full sourcehashes and initialtensor provenance checked.

## Scope and architecture

Frozen cached CLIPViT-B/16 last_hidden_state197tokens → removeCLS →196patch768 →trainableLinear+LayerNorm512 →standard4layer/8head Transformerdecoder/FFN2048/dropout.1 →10000wordvocabulary. Freshinitialseed42,27.462Mtrainableparameters from unchangedsmoke architecture. No detector/ROI/bank/prompt/SCST/objectreward/retrieval/reranking. All113287KarpathyTRAIN(train+restval),566747captionevents/epoch,10epochs/177110updates/5667470captionevents. Global32/T4×2, finalglobal27(16/11), FP32Adam/lr warmup1epoch8e-6→1e-4 then cosine1e-5; globalnonPADtoken normalization. All10VALID5000 cohorts match historicalVALID5000IDs, disjoint2500/rank. NoTESTfeatures/inference.

## Best=end epoch10, VALID5000, beam5/max20

Quality/overlap scores shown×100; CHAIR/Recall inpercent; raw NLL unscaled. Allmetrics from the SAME epoch10savedcaptions; no mixingcheckpointselection.

| Metric | Value |
|---|---:|
| BLEU-1 | 77.21 |
| BLEU-4 | 37.07 |
| ROUGE-L | 57.37 |
| CIDEr | 119.09 |
| CHAIRs | 5.38% |
| CHAIRi | 3.78% |
| Recall | 43.05% |
| VALID teacher-forced NLL | 2.1016 |
| TRAIN online NLL epoch10 | 1.8389 |

269/5000captionflags,277hallucinatedmentions/7320objectmentions;6857/15928uniqueobjectrecall. Mean9.4154words, noempty/≤2wordcaption. METEOR/SPICE/TEST notmeasured. BLEU/ROUGE runtime-reported; CIDErperimagesmean/sealedpoint andCHAIRcounts/captionidentity replayed fromcachedJSON, officialscorers notrerun locally.

## Epoch trajectory

| Epoch | CIDEr×100 | TRAIN online NLL | VALID NLL |
|---|---:|---:|---:|
| 1 | 106.01 | 3.1885 | 2.4661 |
| 2 | 111.62 | 2.3789 | 2.2646 |
| 3 | 115.86 | 2.2248 | 2.1904 |
| 4 | 117.19 | 2.1352 | 2.1489 |
| 5 | 117.11 | 2.0647 | 2.1203 |
| 6 | 116.57 | 2.0025 | 2.1058 |
| 7 | 118.34 | 1.9467 | 2.0987 |
| 8 | 118.28 | 1.8987 | 2.0966 |
| 9 | 118.50 | 1.8625 | 2.0992 |
| 10 | 119.09 | 1.8389 | 2.1016 |

Epoch10highestCIDEr119.09 winsfixedVALIDselection. MinVALIDNLL isepoch8(2.0966), notepoch10(2.1016); thesecriteria neednotpickthesamecheckpoint. Last3epochsCIDEr118.28→118.50→119.09 withsmallNLLincrease; no proof fullconvergence orjustificationforautomaticmoreepochs. This successfulnewbaseline differsfromoldROI/dual-memoryplain inarchitecture/inputs/LR; no isolatedcausalclaimaboutbank/prompt andno directrankingagainstoldTESTscores.

## Cache repair and timing

All118287TRAIN+VALIDlocalpatchFP16 exactcopy33.17GiB, smoke256rowsSHA replayed, allcopyblocks byteequal. Stage+sealing2689.28s (**44min49s**); worker15923.06s (**4h25min23s**) includestrain/VALID/checkpoints/localstageverification, excludesearlierprepare/setup/preflightandfinalqualityscoring. Sum of those2measuredcomponents **5h10min12s**, notentirenotebook/billedGPUhours. Parent10.5hwasamaximum, notrequiredelapsedtime.

FullIOpreflightmedianmaxrank.07116s/update/P90.08439vsV1mountedmedian1.40264/P902.20960. Theseareseparateruns/operationalpipelineprobes, notisolatedstoragecausalbenchmark; actualfullworkercompletedunderbudget. No claim sustainedwholetrain20×speedup from preflight. GPUpeak.790GiB ispreflight-only, notfullrunpeakmeasurement.

## Preserve and next decision

`/kaggle/working/clip_patch_baseline_full_xe10_v3/best.bin` and `checkpoint.bin` sameSHA **5061e305d98edf54930a200090a41c983c354d1ba2dfcf700b715cf20a5b80b7**; preserve eithersealedweights plusreceipts/protocol/source/history/outputv199. ExistingSaveVersionisthesource; noDatasetcreationrequiredthisturn. BestincludesAdam/RNG/cursor fromcompleted10epochs. Noautoretrain/extraepochs/SCST.

Recommendednext: freeze thisbaseline andmeasureTEST5000(plusmissingMETEOR/SPICEfromsavedcaptions), then comparebank/prompt additionsonthissamebackbonewithmatchedbudget. PriorROIadapterscannotbeattachedtopatcheswithoutanexplicitmappingadaptation. Strongbaselinehelpsresearchbutdoesnotguaranteefuturepromptgain. One seed and reusedVALID; do not claim TEST119.09,SOTA,newalignmentbenefit or fairgainoverTâmwithoutmatchingprotocol.
