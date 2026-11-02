# Two-GPU ordinary/shared encoder benchmark — COMPLETE

ReviewZIP SHA256 `4a14542622cfb5f09131a1c6310a185f22de0e578375c0e7d385ec0b6ffb984c`, source `a92b7ea0cdc97391e3c196fda8e5b4e40eff3e5b`. [Facts](DDP_SHARING_RESULT.json), [metadata audit](audit_ddp_sharing.py), reviewed text `ddp_sharing_review_text/`.

## Matched results

TwoTeslaT4, PyTorch2.11.0+cu128, NumPy2.1.3, FP32. Both use16captions/rank/global32, accumulation1, SAMEimage-groupedcaptionorder/init/optimizer recipe;10warmup+30timedupdates each. Noqualitymetric/checkpoint/fulltraining.

| Measure | Ordinary | Sharedencoder |
|---|---:|---:|
| Medianseconds/update |0.105242|0.099481|
| p90seconds/update |0.107886|0.102131|
| Encoderimagerows/globalupdate |32|8|
| PeakallocatedMiB rank0/rank1 |1252.452/1187.370|1182.669/1184.337|
| PeakreservedMiB rank0/rank1 |1390/1320|1422/1256|
| Train-only hours/epoch projection |0.518|0.489|
| Train-only hours/15epochs projection |7.766|7.341|

Matchedsharingthroughput gain5.791% (elapsedtime reduction5.474%), approximately25.51min saved across15epochs usingmedianprojection. Reductionofencoderrows4× does not meanwholemodel4×faster: decoderstillprocesses32captionrows andgradientcommunicationremains. Reservedmemory is not uniformlylower; do notclaim universalmemorysavings.

Historical1GPUmicro4/accum8 benchmark was0.414184s/update. Newsharedconfiguration is~4.16×faster in these differentruns, but GPUcount,microbatching,captiongrouping/order andpaddingchanged; this is not a matchedcausalGPU-count comparison. No speedupfactor should be attributedentirelytoDDP or sharing.

## Preflight and audit

Allsixdropout0cases pass:batch32,actualtail27(split16/11),stress32(maxROI87/prefix50), eachordinary/shared. MaximumNLLabsoluteerrorvs1GPUexpandedreference4.6153e-7; maximumrelativegradientL2error1.22724e-4, maxabsolutegradienterror7.07805e-8 acrossfiveprobes. These are belowpredeclared5e-5NLL/.005relative/3e-5absolute thresholds. Allparametergradientsfiniteguard andfrozenbanks passeduserworker. Index_select sharing didnotdetachencoder onthesechecks. This is numerical/logicvalidation withdropout0, nottrainingtrajectory/semanticgroundingaccuracy orqualityequivalence.

573metadata assertions PASS locally:12GitblobSHAs, bothrankorders identical,1280distinctTRAINcaptionindices,tail27range, unchangedfullcache/bank/stagefiledigests vspriorbenchmark, sixloss/gradientreports, both40steps/timedmasks/globaltokens/paddings,32vs8encoderrows, timings/statistics/projections recomputed,perrankGPUmemory andfinalstatus. NoagentTorch/features/NPY/weights/scorerexecution; stagedarrays notincludedreview. StageSHAsagreewithprioruser-runfullbenchmark, notindependentlyrehash locally.

Actualtimingdropout.2 meanssharedencoderreusesdropoutnoise acrossreferences, ordinaryencoderusesindependentrows. Samplerimagegroupingalso differsfromoldrandom-captionpilot. Marginalobjective/globaltokenmean remain intended, but statisticaltrajectory/qualityequivalence has NOTbeenmeasured. Keep operationaldifferences explicit.

## Recommendation

TwoGPU execution is now technically validated and feasible toprepare for a fullXE run. For the FIRSTCSA baseline, recommend ordinary2GPU (~7.77h train-only) as the conservativecontinuation oftheexistingper-captionencoderbehavior. Sharing is valid as measuredexecutionvariant, but its~25.5min projected saving is modest relativeto15epochrun andintroducesencoder-dropoutcorrelations; no evidence thatqualityisworse orbetter. This recommendation is aboutbaselinecontinuity, not rejectionofsharing.

Beforelongrun: explicit15XEfullKarpathyrecipe(fromscratch,global32,3epochwarmup,knownconventions), officialvalscorer/checkpointselection, periodicresumablecheckpoints withbothranksRNG, andavailablewall/quota/sessionbudget. Timingabove excludesvalidation,scorers,checkpointI/O/startup;7.3–7.8h are elapsedtrainprojections, not guaranteedKagglebilledGPUhours. Noautomaticfulltrainingstarted. Old131greedy/beamdiagnosticstillunresolved; thisbenchmarkdoesnotdiagnoseit.
