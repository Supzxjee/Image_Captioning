# Full-cache XE throughput benchmark, not full training

User confirms mount `/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134`. Semanticbanks130 confirmed earlier. Latestquota report30GPUhours before CPU-only ROI/save; not a promise of availablebudget after other jobs. User approved measuring throughput/memory before scheduling fullKarpathyXE. This notebook uses GPU0 only; secondGPU/AMP/DDP not silently assumed. Fulltraining and oldcheckpoint131diagnostic remain separate.

## Input and staging

Require BOTH fullROI134 andsemanticbank130 mounts beforemodel/download work. Verify247fullROIfileSHAs, receipt/manifest/vocab/source identities and all119per-image metadata shards via savedtransporthelper. Read NPZ `allow_pickle=False` and validate everyrawarray/offset/ID/box/score/54-token sequence against same metadata using existingNumPyvalidator. No H5/RGB/model encoding required.

Stage compact `.npy` arrays in temporarydirectory (notfinalreview): featuresFP16[833473,768], offsetsINT64[118288], IDsINT64[118287], captionsequencesINT32[591757,54], captionimageindexINT32[591757], groupUINT8[118287]. Records followmanifestorder, TRAINbeforeval; confirm566747TRAINcaptionevents,25010valevents from actualrecords. Validate uniqueIDs/excludedtest via transporthelper. Banks loadedwithcorrectedwordJSON/fullSHA fromrepo plus originalinputpackage/Output130receipt; noold320ROIinput is required. RecordstagefileSHAs andstartupCPUduration, mmapsread-only in worker.

## GPU work

Fresh unchangedpaper-profileCSA, FP32/noAMP/noTF32, frozenbanks, Adam(.9,.99)/eps1e-8/decay0, clipnorm1. NoC0/pilotcheckpoint reuse. Effectivebatch32=micro4×8. Numerator=sumtokenNLL per microbatch; denominator=totalvalidEOS-inclusive/PAD-excluded targets acrossfull32. Stressfirstmicrobatch includes amaximum-region image and amaximum-length target to measurepaddingmemory; onebackward/clip check, nooptimizerstep. Then resetseed/model for actualtiming.

10warmupupdates excluded fromtiming +30timedupdates usingdistinctseed42shuffledTRAINcaptionevents. Constant1e-4 forbenchmarkonly (not a600-step pilotschedule/fullwarmuprecipe). Finiteforward/loss/allgrads/bankbuffers checked. Timewall withCUDA synchronize includesbatchgather/H2D/forward/backward/optimizer, notinitialcachehash/staging/modelinit. Save30stepdurations,median/p90/mean,paddedshapes/tokenstats,allocated/reservedpeak. Warm-cache/mmap measurement is notcold-sessionepoch runtime guarantee. No optimizercheckpoint saved;40smallbenchmarkupdates notfulltraining/qualityresult.

Time teacher-forcedNLL on first32valimages/allrefs and actualbeam5/max20 on same32valimages afterbenchmark. Prefixes neverusereferences. Storegeneratedlength/timing only, explicitly freshmodel40updates; long-trainedcaptionlength/runtime can differ. NoCIDEr/CHAIR/grounding orqualityclaim. Forecasttraincompute-onlyhours for1/15epochs usingceil(566747/32)=17711updates/epoch andmediantiming; add separate projectedvalNLL/beamtime from32imageprobe, label officialscorers/checkpointI/O/coldcache excluded. Do not automatically launchfulltrain if estimatefitsquota.

Workerwallcap1800s includingstage+GPU, tempdisk6GiB, oneGPU. Preservefailurelog/partialtimings, INCOMPLETEonfailure. Agent onlymetadata/NumPy/mock/static tests; userrunsallactualarrayload/modelwork. Afterreview decide whether1GPUFP32fitsbudget or a separatelyvalidated2GPU/AMP path is needed. Newfulltrainer/evaluation/saving policy must be explicit before longrun.
