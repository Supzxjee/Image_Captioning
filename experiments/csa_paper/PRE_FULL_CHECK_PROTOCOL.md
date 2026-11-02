# Before full Karpathy XE: decode checkpoint and prepare raw ROI coverage

02/11/2026. User approved the two steps after discussion: same-checkpoint greedy/beam/EOS inspection; full Karpathy TRAIN/val ROI preparation. No full training, additional15epochs, new decoder scoring, RL or test evaluation is scheduled.

## A. Same-checkpoint decoder inspection

User confirmed source Output131 of `ducanh2403/2-vi-cai-kia-qua-nhieu`. Use endpoint15/update600, checkpointSHA `99b1f9936b864e549182f2307c369a221e4ca4f17b0f0cad9c60f3fb1cb7017e`. Exactversion download uses the previously pinned CLI/version-routing mock preflight and a KaggleAPI Secret, checkpoint file only, no latest fallback. Preserve original state and exact model/data/bank source. Load user-trusted self-produced checkpoint explicitly on user Kaggle only, strict model/config/source/input/frozenbank checks. No optimizer step. Same64val images, same BOS/vocab/beam5max20/rawlogscore/no lengthpenalty. Decode greedy (beam1 implementation) and beam5, confirm beam5 exactly reproduces reviewed caption IDs including PAD afterEOS. Record maxscoreerror with tolerance (rtol1e-5/atol1e-4); fail mismatch outside tolerance, not arbitrary tiny score rounding.

For every greedy prefix untilEOS/cap: EOSprobability/rank, chosenID/probability, best nonEOS alternative (BOS/PAD forbidden as in existing search). Also record unforced prefixEOS at BOS (step1). No minimumlength, forcedEOS suppression, lengthnormalization or label-dependent decoding. Val references may be exported for inspection but never fed to decoder. Exportgreedyvsbeam wordcounts,EOS,repetition and perimagecandidate scores; do not interpret longer as moreaccurate or claimgrounding. No teacher-forced prefix as inference.

Use cachedROI320+bank130, GPU0FP32, cap900s. Review excludes checkpoint; do not update its weights or overwrite source. This diagnoses earlytermination/search hypotheses, not anewmethod/metricbenchmark.

## B. Full ROI preparation, CPU only

Inputs: pinned `dataset_coco.json`, detectorJSON and full rawCLIP H5cache; existing correctedassets Dataset for verifiedvocabulary. Reuse ROI pooling/coremetadata helpers; no spaCy/POS/CLIP encoding/Torch/captionmodel. TRAIN=train+restval113287; val5000; test excluded. Each group sorted by rawH5path/row for sequential I/O, not assumed same as originaldataset order; exported explicitCOCOIDs/filenames remain authoritative.

Verify originaldata/detector SHA, exactsplitcounts/uniqueCOCOIDs, completeH5profiles/tables and row filenames, missing-vs-emptydetections, finite197×768 rows. Read actualJPEGheaders for width/height (notfullpixelhash), use fractional-area pool on196patches, FP32compute/FP16store, top150confidence stableties, genuineempty→fullframefallback. Boxproducerprovenance/visualrevisionunknown remain declared. Captionseq54 inclBOS/EOS and vocabulary unchanged. No perimageGT used tochooseROI.

Write1000images/shard:114TRAIN+5val=119NPZ and119metadataJSON, eachoriginalbyteSHA/shape/count inmanifest. Metadata contains IDs, originalsplit/group,dimensions,sourceboxcounts/clipping/fallback/rawrowSHA, originalreferences and encodedtargets. Copy onlyverifiedvocabulary; bank staysseparateexistingDataset. Featureprofile/source/config/selection/progress receipts explicit. ReviewZIP onlyglobals/logs, not allshardmetadata/features; fullfolderpreserved for Dataset later. Expected~245files below500transportlimit. Actualcompletion requires118287uniqueimages,113287TRAIN/5000val/0test, all119shards presentandhashed; nevermarkpartialcomplete.

CPU cap6h (notETA), internalstop at5h50min, flushprogress after everycompletedshard; a partialrun is nottrainingready. Reportfirst1000-image throughput/projection andactualwall/diskusage. If projectedremaining exceeds remaininginternalbudget afterfirstshard, stop early with thatshard retained, rather than silently consume6h on anunlikelycompletion. No automatic resume/parallelworkers/fullH5payloadhash claim; partialshards can be retained for separatelyprepared continuation. At least8GiBfree required; finalsize measured rather than promised. Input rawcache is nevermutated.

## Boundary and next decision

Agent onlymetadata/NumPy/mock/static checks. UserrunsCUDA/HDF5featurepooling/API. Afterdecoderreview andfullROIcomplete, prepare a measuredfew-dozen-update throughput check with full-data batching and calculate a fullXEbudget. Do not extrapolate331s smallpilot or commit full15XE+30RL now. Diagramwork staysatplanningstage asrequested.
