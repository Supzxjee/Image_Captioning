# Full Karpathy ROI preparation — COMPLETE

User-run review ZIP SHA256 `a7749fde0ee8ee2ccfb78661c8121d892179dcc81d842864f969b1569bb0ea41`, source `936cd17f1bc33de16ceccda8d88cfda77e1bf188`. [Audit facts](FULL_ROI_RESULT.json), [metadata audit](audit_full_roi.py), original reviewed text `full_roi_review_text/`.

| Item | Actual reported/consistent metadata |
|---|---:|
| TRAIN (train+restval) images |113287|
| Validation images |5000|
| Test images selected |0|
| Total images |118287|
| Raw768D regions |833473|
| Caption events, TRAIN+val combined |591757|
| Paired shards |119NPZ +119metadataJSON|
| Full-frame fallback images |1004|
| Maximum regions/image |87|
| Boxes requiring boundary clipping |0|
| Shard payload bytes |1749398178 (~1.75GB decimal /1.63GiB)|
| Worker time |3431.090s (~57min11s)|

The final run's first1000images took30.041processingseconds (previousattempt199.857s), and its projection was3523.462remainingseconds. This documents I/O/runtime variability; raising a timeout does not itself speed up pooling. Actualworker duration excludes notebooksetup and is not a futuretraining ETA.

## Checks and evidence boundary

862metadata assertions PASS:7sourceGitblob hashes; presentmanifest/inputreceipt fullbyteSHAs; progress andmanifest byte equality; exactdata/detector/vocabulary identities and retainedfeatureprofile;119sequentialshard names,114TRAIN+5val counts with287-image tail; region/count sums and bounds;238uniquepayloadnames and SHAformat; log119completedshards/finalcompletion; nofailureartifact. No featurepooling/model/HDF5/scorer run locally.

The238binary/per-imageshard files are excluded from review. Their hashes and per-shard ID-order digests are reported; full payloadSHA verification and independentper-image ID/sequence/shape checks must occur when transporting/loading the actualcache. Do not call all118287IDs independently reread locally. Worker used pinned select_full uniqueness/split checks and validated eachreadrawrow/filename/dimensions/ROI, as verified in source. JPEGheaders read fordimensions, not pixelhashes. Full H5 payloadwas nothashed; visualmodelrevision null and detectorproducerpolicy unknown are preserved. Data encoders remain declared CSA substitutions.

Manifest SHA `512bf34336be54f2854743bd06004dbc30ff506cda130c03378d77a6d3cf6130`. This is CPUROI preparation only: no CLIP reencoding, POS, captiontraining, qualitymetrics or semantic-grounding result. BankDataset130 remains separate and reusable.

## Next

Preserve entire `/kaggle/working/csa_full_roi_karpathy` folder; smallreviewZIP alone is insufficient. Obtain exactsourceversion and save Dataset by verifying all238payloadSHA plus originalglobalreceipts/vocabulary. User asked forsourceversion; actualmount pending. Then measuredfull-databatching trainingbenchmark before fullXEbudget. Decode131 still has only the incompleteinputroot attempt reviewed; do not mark greedy/beam questionresolved. No additional256imageepochs/fullXE/RL automatically authorized by this result.
