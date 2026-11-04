# CSA15 → paired full-TRAIN SCST, exactly one epoch

04/11/2026. User approves ONE full-TRAIN epoch first, not five. Latest available GPU quota19h/24h across the two accounts. Each notebook usesT4x2; user runs it. ExistingCSA15 epoch147 checkpoint is the common starting point, freshAdam, not pilot500/2500 weights/optimizer. No newDataset for pilotoutputs required.

## Fixed training and population

- Both arms: CE+.05SCST(CIDEr-D+wRobj),w0/.5. Same model/CSA/CLIPROI/textbanks/tokenizer/labels/reward as2500; Adam5e-6,betas(.9,.99),FP32/dropoutOFF/clip1. TRAIN-only object targets now cover ALL113287TRAIN/restval images. RewardIDF remains first5references of ALLTRAIN. No VALID/TEST labels in reward.
- One deterministic permutation of allTRAIN IDs (hash42order then RNG42). Exactly113287image events,14161updates at nominalglobal8. Lastbatch7images split4/3 across ranks; sentinelpadding is NEVER treated as an example, reward or loss token. No repeated image to fillbatch. CE scales by globally summed validtokens; SCST image-mean scales localmean by world×localimages/globalimages. Same schedule botharms.
- Fresh initialization of optimizer on exactCSA15 weightSHAe02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25. All captionerparameters trainable; featurecache/bankbuffers frozen. No newmodule/lambda, no localmodel execution, no automatic further epoch.

## Evaluation, selection and preservation locked before output

- Evaluate ORIGINAL beforeupdate0, MID after7081updates/56648images, END after14161updates/113287images, each on fullVALID5000 beam5/max20 plusNLL. Original is freshly decoded/scored under this same workflow, not assumed from historical metrics. No TEST, reranking or arbitrary extra snapshot selection. Save MID weights and latest weights, optimizer/RNG/cursor/evaluation receipts for interruption resume.
- CPU score all three with same PTB/CIDEr/B4/ROUGE/CHAIR/recall/length. Eligible iff CHAIRs≤freshORIGINAL AND CHAIRi≤freshORIGINAL AND recall≥freshORIGINAL−.005absolute. Select maximumCIDEr among ORIGINAL/MID/END; exactties earlier. Original fallback is allowed and explicitly reported, not rebranded as a trainedarm. Selection uses reusedVALID, so no independent generalization or isolated object-term claim if either arm falls back.
- Preserve latestcheckpoint, MIDcheckpoint, selectedbest binary/receipt and locked_selection.json with IDs/source/data/metric hashes. Archive smalltextreview; large binaries/stage not embedded. Final result records allpoints plus selection; no scoring subset markedCOMPLETE.
- Same-arm/same-source/same-full1 resume only. Preserve/check validation payload hashes referenced by checkpoint; never regenerate ORIGINAL from a trainedcheckpoint. Completed evaluation phases retained; failed validation resumes at the same modelcursor. Noautomaticrestart.

## Runtime and checks

Parent4h STOPcap remains, not ETA/billing. Atomiccheckpoint every500updates plusmilestones/cap;25step measuredprojection includes pendingVALIDmargin. Ifinsufficient cap, preserve partialrun for explicitresume; do not silently drop images/evaluation or raisehorizon. Latest19h/24h report is not reinterpreted as measured pernotebookruntime.

Inputs unchanged: ROI134Dataset, CSA15v147Dataset, dataset_coco.json, originaldetectorJSON. No newCLIP/model extraction/APIsecret. User-run tiny policy/gradient/DDP checks plus new unequal4/3 DDPguard before expensive staging/training. Agent runs only NumPy/static/mock/bundlechecks. Important verification: fullIDcoverage/lastbatchmask/14161cursor/7081milestone/resumephase metadata/selectionfallback and ties. Checkpoint/source/data identities strict. No guarantee that a full epoch beats QFormer; compare botharms jointly onCIDEr/CHAIR/recall.
