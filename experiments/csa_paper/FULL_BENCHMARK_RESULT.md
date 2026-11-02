# Full-cache XE benchmark — COMPLETE; current15epoch budget too tight

Review ZIP SHA256 `ae4bb5a8b371729be1f26e95d27f84ed7a5fe8bcca9fb929bcee595609f95501`, source `eeda5cdf2943bf1407e371ad1485b282325eb4c9`. [Facts](FULL_BENCHMARK_RESULT.json), [metadata audit](audit_full_benchmark.py), reviewed text `full_benchmark_review_text/`.

## Measured configuration and result

One TeslaT4 GPU0, FP32/noAMP/noTF32; twoGPUs visible but onlyone used. FreshCSA, frozenbanks, effectivebatch32=micro4×accumulation8. Stressbackward(nooptimizer),10warmup+30timedupdates;40actualoptimizerupdates total. No checkpoint, full epoch, officialmetric or oldcheckpoint131 evaluation.

| Measure | Value |
|---|---:|
| Median train update |0.414184s|
| Mean /p90 update |0.416837 /0.429995s|
| Normal-training peakallocated /peakreserved |997.968 /1064MiB|
| Stress firstmicrobatch |4examples,87regions,maxprefix50|
| Stress peakallocated (noAdamstep/state allocation) |521.399MiB|
| Fullcache validation/staging |118.295s|
| Worker total |152.221s|
| FullTRAINupdates/epoch |ceil(566747/32)=17711|
| Train-only projection/epoch |2.038h|
| Train-only projection/15epochs |30.565h (p90scenario31.732h)|

Timing includesgather/H2D/forward/backward/Adam; it excludesinitialstaging,checkpointI/O, officialscorers/cold-session costs. Fifteenepochs consume265665updates. The one-GPUFP32configuration does not leave margin in the latestreported30GPUhours even beforevalidation. This is measured-sample projection, not guaranteedruntime or a livequota reading.

Validation timing probe:32valimages/160references teacherforced0.680s;32beam5images4.918s. Linear scenarios:fullvalteacher106.319s andbeam768.396s. Adding these onceperepoch gives34.210h for15epochs, still excluding officialmetrics/checkpointI/O. This is NOT a mature-model ETA: all32selectedbeamcaptions were empty (EOSatfirsttoken) afteronly40benchmarkupdates. Actualbeamdecoderloopcounts acrossall5beams were notstored, so cannot infer executeddecodersteps or guarantee how much longer maturegeneration willtake from bestcaptionlength alone. These are not qualityresults or a regression of endpoint131.

## Data/runtime validation

Userworker verified247inputfileSHAs, independently inspected all119metadataJSONs, loaded all119NPZarrays withallow_pickle=False, checkedarraydtype/shape/offsets/IDs/boxes/seq againstmetadata, and stagedreadonlytemporaryNPYfiles. Counts:118287uniqueimages/833473regions/591757captionevents;566747TRAIN,25010val;0test. Frozenbank equality passed. Thus actualfull-cache load/staging has now succeeded onKaggle, beyond the earliermanifest-onlyreview.

247independentmetadataassertions PASS locally:11GitblobSHAs; sourcepackages/manifest/bank identities; stagecounts/shapes/SHAformats;40steps,1280distinctTRAINcaptionindices;token/microbatchshape bounds;finiteNLL/gradnorm;timingmedian/p90/mean recomputed;warmupmask;stress/runstatus;projectionarithmetic;32uniqueemptybestbeam outputs. ActualNPY/NPZfeaturecontents/stagedfileSHAs were NOT reloaded/rehash locally; no agentTorch/model/scorer execution.

## Recommendation — proposed next optimization

Measuremicrobatch16×accumulation2 andmicrobatch32×accumulation1, preservingeffectivebatch32, FP32, tokenmeanXE and architecture. LowmeasuredallocatedVRAM gives a reason totrylargerphysicalbatches, not proof ofspeedup or memoryfitness. Resetinit/order comparably and recordactualpeaks/timings; padding/RNGdifferences must not be calledbitwiseequivalent. This changesexecutiongranularity, not the researchmethod. Prioritize this shortmeasurement beforeintroducingAMP/DDP or schedulingfull15epochs.

No newbenchmark/fulltraining was launched by this review. Fullrunrecipe/checkpoint/scoring/schedule remains to be selected withuser afterbudget. Olddecode131inputfailure remains unresolved; noautomaticminlength/lengthpenalty introduced.
