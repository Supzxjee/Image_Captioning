# CSA15 → paired SCST2500, fresh restart from XE15

04/11/2026. User approved this follow-up after the500update pilot. User explicitly declines a new Dataset for pilot outputs: preserve their SaveVersions CE159/OBJECT53 only. **Both new arms start again from the existing CSA epoch15 Dataset**, SHA e02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25, with fresh Adam; NOT pilot checkpoint/optimizer continuation and NOT random initialization. No download/API access to159/53 required. Original500endpoint reports/notebooks remain unchanged and pinned.

## Fixed changes versus the approved500 pilot

- Same two arms, T4x2 each, same CSA model/banks/frozenROI, global8, Adam5e-6/betas(.9,.99), FP32/dropoutOFF/clip1, CE+.05SCST(CIDEr-D+wRobj), objectweight0/.5, same legal sampling actions/max20 and reward parser/annotations. No architecture/lambda/metric change.
- Total2500updates perarm=20000TRAIN image events. Same hash-selected10kTRAIN pool and deterministic seed42 first permutation, then seed43 second permutation. Each10000image pass completes before a new pass; exactly2exposures/image. First500updates/4000IDs and CEevents reproduce original schedule. Second visit rotates CEreference byone from its first selection. Same schedule across both new arms. This jointly changes training horizon and exposure coverage relative to old500: do not attribute all change to time alone.
- Fresh runs leave RESUME_ROOT empty. Optional resume is only for an interrupted SAME2500study/arm/source identity, never for159/53. Checkpoint every100updates and synchronized cap.25step measured projection retains partialweights if the remaining worker cap is insufficient. Parent4h unchanged from measured500 pilot/budget previously reported20h/23h; no new unbounded30RLepoch run. Parent cap is not ETA or GPU billing guarantee.
- Single fixed endpoint2500 then fullVALID5000 beam5/max20, CIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/NLL/length. No intermediate checkpoint selection, noTEST, no reranking. Keep prior500metrics as history; fresh2500does not append to those results.
- Success assessment is joint: CIDEr versus originalCSA15, CHAIRs/i and recall versus original and matched SCST+CE. No promise of improvement or newly invented acceptance threshold. Inspect paired uncertainty and caption counts; flag quality/recall tradeoffs rather than choosing byCIDEr alone. Both arms need complete endpoint/results for comparison.

## Inputs and verification

Existing4inputs unchanged: csa-full-roi-karpathy-v134; csa-full-xe15-v147; mscoco-2014/dataset_coco.json; objectdetectionecache/objectdetectioncache.json. No newDataset/model extraction, API token or GPU execution by agent. Same fullTRAIN5reference rewardIDF, filter TRAINbefore object labels; noVALID/TESTlabels in reward. Cache hashes verified by user-run preparation; source pinning and local NumPy/AST/mock checks do not prove runtimeCUDA/model parity.

Author checks: scheduleprefix/reversedIDdeterminism/full10kcoverage/exact2exposures/reference rotation/heldoutrejection and malformedhorizon; isolated notebook bundles, source hashes and all code-cellAST. Existing tiny policy/EOS/gradient/DDP tests are unchanged and run on Kaggle before training. No local Torch/autograd/model/scorer or Kaggle submission.
