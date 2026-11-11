# Full Karpathy XE10 — restored CSA + prompt + grouped feature sharing

11/11/2026. User explicitly requests full training with the successful benchmark configuration. Operational interpretation: fresh10epochXE on113287KarpathyTRAIN, all566747captionevents/epoch; not continuation from smoke, no SCST/TEST/reranking. Account with lastreported30h recommended; ownlast6h insufficient for estimated full10. Do not claim current quota or guaranteed runtime. Parent11.5hSTOPcap, prepare90min; checkpointed partial run can be explicitly resumed with SAME source/data/horizon. User runs Kaggle, no author local models.

## Fixed configuration

Reuse exact restored CSA model/SharedCSAPatchCaptioner, fixedCLIPpatchcache/bank130/prompt175.2completeOSA+1ORA, separate prompt memories,3layer parallel adaptive decoder,dim512/heads8/FFN2048/dropout.2. Accelerate1.15.0 pinnedwheel/no-deps isolated target, nativeNCCL underneath, FP32/noAMP/no compile/accum1/global32captionevents. No new model parameters or sharing implementation. Lastbenchmark1.8046x compares within grouped sampler, not Accelerate vsnativeDDP and not end-to-end/fulldata speed.

Image groups permuted with default_rng(42+epoch), all captionevents retained including variable perimage counts and final27events split16/11. Within each rank read/encode only uniqueimages; expand both memories tocaptionorder inside the model with gradient sum. Encoderdropout correlations and groupedsampling differ from historicalrandom baseline; report complete-system adaptation, not pure isolatedalignment effect. Bank/textvectors frozen, alltrainableCSA/adapter/decoder weights freshseed42. No smokeweights.

Training recipe follows matched10XE patch experiments: Adam(.9,.99),eps1e-8,wd0,clip1; global-token XE (DDPfactor2/globaltokens), warmup first17711updates8e-6→1e-4, then9epochcosine to1e-5.177110totalupdates/5667470captionevents. Not originalCSA15XE+30RL/3epochwarmup; inputCLIPpatch/text and UPTprompt extension remain disclosed substitutions. Checkpoint identity includes model method, source, data seals, sampler,precision,horizon/software.

## Data and runtime

Four existing Datasets: visual-cache,mscoco-2014,bank130,prompt175. Reuse fullmetadata/physical-orderlocalFP16patchstage (~33.17GiB) and fullsemanticstage.40GiB/tmp +8GiBWorking required; no CLIP/UPT inference/newDataset/APISecret. Stage rawTRAIN+VALID only, noTESTfeature reads. RawstorageFP16 converted toFP32 fortraining. Do not retain huge localstage in reviewZIP. Everyresume restages/checks exactpayloads; no assuming/tmp survivesSaveVersion.

Full-data preflight under Accelerate: numerical reference uses rank-shaped ordinarymicrobatch gradients, exactglobaltokens, scale-awareV3diagnostics; sharedactualencoderrows;64TRAINprobeupdates discarded. Actualfull-loader speed andpeakmem recorded. SyntheticTRAIN beam5 timing suppressesEOS on a separate copy to force20decode steps; restorecopy/discard and never treat synthetictexts ascaptions. Check unmodifiedrealbeam paths separately. Forcedlengthestimate avoids underestimatingVALID from earlyEOSsmoke. Mainworker recreatesfreshseed42 and verifiesinitialdigest vs preflight.

Allow progress toward requested10epochs within cap: do NOT require whole10fit before doing anytraining. Beforestarting require measuredcurrentepoch+fullVALID+checkpoint reserve fit; save checkpoint and stop if budgetinsufficient. Periodic guards estimate currentepoch remainingwork only; overall10epochprojection remains informational. No automaticcapextension/restart/horizon/LRchange. Latest/best atomiccheckpoints every500updates andphase/epoch transitions retainAdam/RNG/cursor/exposures/history. Parentcleanup reserve and resumablevalidate phase required.

## Validation / outputs

After EVERY epoch: VALID5000beam5/max20/rawcumulativelogprob/rank0, teacherforcedNLL/CIDEr, meanlength andshortcaptioncount; savecaps/perimageCIDEr/workerstats. Inspectepoch1qualitativeprogress in logs; fixedtraininghorizon10, no adaptivehyperparamchange. Best=maxVALIDCIDEr, earlier ties. Finalbest/end savedcaption scoring: BLEU1/4,ROUGE-L,CHAIRs/i,Recall; METEOR/SPICE/TEST later, not claimed measured here. No referencecaption selection/re-ranking.

Preservecheckpoint.bin ANDbest.bin/receipts, optimizer/RNG, completedvalidationfiles. Incompletecaprun is not full10qualityresult and must not be ranked ascomplete. Newoutput`csa_structure_shared_full_xe10_v1`, notebook`CLIP_PATCH_CSA_SHARED_FULL_XE10_ACCELERATE_2GPU.ipynb`. AuthoronlyNumPy/mock/AST checks; actualfullIO/Accelerate/GPU/scorers user-run. Pushcode/results and log oldSheet, no unsolicitedWord edits.


## Amendment 11/11: V1 packaging failure / fresh V2

V1 review SHA efe2bb45f2997bf1dc4868b1766c64df2a5bd3a88eeba0a5e74972ac65d5ef42: full raw stage completed (118287 TRAIN+VALID images, 2566.4s), semantic stage completed, then model-module import failed because memory_fusion.py was absent from the packaged mapping. No model construction, optimizer updates, checkpoint, or quality result. CUDA2/NCCL were available; this is an author packaging error, not a GPU/data error.

User ran Save Version; /tmp staging is not saved in Output. V2 restages with the same four inputs/architecture/recipe/budget. Add missing module, static reachable local-import closure check in generator, and Kaggle import-only class precheck BEFORE expensive preparation. Preserve V1 notebook. New notebook CLIP_PATCH_CSA_SHARED_FULL_XE10_ACCELERATE_V2_2GPU.ipynb/output csa_structure_shared_full_xe10_v2. No local neural imports/models/Kaggle execution authorized.

## Amendment 11/11: no feature sharing

User requests ordinary per-caption encoding after V2 tail27 failure. Keep image-grouped batches, FP32, Accelerate, data and schedule. Compare ordinary rank-shaped reference with ordinary DDP on both batch sizes, with existing limits. New method identity and output. Prior ordinary benchmark implies approximately 12.7h for 10 epochs of training alone, so use checkpoint continuation within the existing session cap.

## Amendment 11/11: targeted feature-sharing probe

User approved checking feature sharing before committing to a full run. Replay epoch 1 tail27, with the same frozen inputs, rank split, seed and FP32 execution. Read full metadata to recover the sampler and scan prompt links for the selected images. Read only their H5 feature rows. Compare local and DDP gradients and log the decoder neuron 933 ReLU sign changes. Retain the previous gradient criterion and leave the full training notebook independent.
