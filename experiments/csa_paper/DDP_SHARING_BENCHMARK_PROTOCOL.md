# Two-GPU XE throughput: ordinary vs within-forward feature sharing

User approved shortmeasurement of twoGPUordinary vsfeature-sharing after full-cachebenchmark. Nofulltraining/qualityclaim. FullROI134+bank130 only; no checkpoint/old320cache/API. Reuse validatedtemporarymmapstaging once. Base `model.py` unchanged; wrapper only changes how many image rows enter `encode`, then index_select maps Vo/Vr/valid back tocaptionrows, preservingautograd. No detach or cache of learnedfeatures acrossupdates/ranks. Frozenrawcache remains sharedread-onlyCPUfiles; eachGPU has its ownmodel/banks.

## Matched modes

- DDP FP32, NCCL,twoT4-compatibleworkers, microbatch16captions/rank,accumulation1, globaleffective32CAPTIONS (not32images×5refs), same Adam/clipping. DisableTF32/AMP. GPUmemory/communication must be measured, not inferred2×speedup.
- Groupall566747TRAINcaptionevents by their113287images, seed42shuffleimagegroups, preserveallreferences exactlyonce. Samefirst1280eventindices for bothmodes,10warmup+30timedupdates. Globalbatch32 is splitcontiguously16/16 tokeeprefsnearby; a boundaryimage may be encodedonceonEACHrank, no crossGPUactivationcommunication. No new samplerqualityclaim; groupedorder differs from oldrandom-caption1GPUbenchmark, so oldspeed is historicalcontext, not cleanisolation ofGPUcount.
- Ordinary: encoder gets16rows/rank (repeatedimagesallowed). Shared: encoder getsuniqueimages/rank anddecoderstillgets16captionrows. Banks/projections/decoderremaintrainableasbefore. Sharedencoderdropoutnoise changescaption-gradientcorrelations; no bitwise/equivalenttrainingclaim withdropouton. This is an execution/sampleramendment, not a newcaptioningmethod.
- Loss onrank r: `2 * local_sum_token_NLL / global_valid_target_tokens` beforebackward, because standardDDP averagesgradients. EOSincluded/PADexcluded; allreducevalidtoken counts, check againstglobalbatch reference. No localmean-of-means. Referencesneverbecomeinferenceprefixes; thisbenchmarkdoesnogeneration/metrics.

## Mandatory user-run preflight

Beforetiming, dropout0models inbothmodes, sameinit/weights, fullbatch32, actualepochtail27(split16/11), andstress32 containingmaximumROI/targetlength. CompareglobalNLL with1GPUstandardexpandedreference: absoluteNLLerror<=5e-5. Compareprojection,noun/verbaggregation,firstdecoderobjectgate andheadbiasgradients beforeclipping: relativeL2<=0.005 AND maxabs<=3e-5. Allparameters must havefinitegradients, inspectedencoder/gateprobes nonzero. ReferenceandDDPshare source/config/inputs, nooptimizersteps. Comparisondropout0testslogic/precision, nottrajectoryequivalence. Failbothranks onguardfailure; neverloosentolerancetosaveclaim. NCCLtimeout120s, supervisedtorchrunmax_restarts0, destroyprocessgroup infinally.

## Timing and artifacts

Resetweights/optimizer/seeds for eachmode, dropout.2, constantLR1e-4 forbenchmarkonly. Periterationbarrierbeforestart, CUDA-synchronizedwall includesdata/H2D/tokenallreduce/forward/backward/gradientallreduce/clip/Adam; slowerrankduration isrecorded. Timing-reduction/logwrite overhead excluded and noted. Report10warmup+30timeddurations, perrankpeakallocated/reserved andencodedimagecounts,globalcaption/tokencounts,global40updateorder, median/p90+train-onlyepochprojection(17711updates); speedratioonlybetweenmatched2GPUmodes. Old.414s1GPU/micro4random-captionbenchmark notmatched. No promiseunder30GPUh or billingrate; quota/sessionconstraints recheckedbeforelongrun.

Parentpreparecap900s, fullnotebookworkerbudget1800s, no checkpoint/fullepoch. Reviewincludesstage/source/preflight/perrank/mode/combinedsummaries, no featurearrays/weights. Userexecutesallmodels/staging; agent onlyNumPy/static/mockchecks.

Primaryreference: [PyTorch2.11 DDP documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html), note on gradientaveraging and equivalentbatchloss; standardDDP, notcustomcommunicationhooks.
