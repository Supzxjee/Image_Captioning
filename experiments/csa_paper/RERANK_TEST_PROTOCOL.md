# Frozen CSA161 plus fixed CLIP-object reranking on Karpathy TEST5000

05/11/2026. User approves the next matched TEST after reviewing VALID R247. Fixed policies: decoder rank0 and .3minmax(raw decoder logprob)+.4minmax(CLIPScore)+.3normalized object consistency; conf>=.5/penalty1/earlier ties. No coefficient/checkpoint/length/policy search or TEST-based fallback. Report both even if the selector fails. Candidate pool163 and scoring164 are VALID and MUST NOT be treated as TEST. Original paper/old project TEST has prior exposure; do not call this a never-seen independent test.

## Inputs locked before decoding

- Captioner exact `ducanh2403/2-vi-cai-kia-qua-nhieu/161` best.bin SHA58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b; CSA15 + ONE fullTRAIN SCST+CE epoch,14161updates/113287events. Not object-reward arm, epoch10 or a newly trained model. Banks embedded. Original model.py SHA139edbb1, per-image beam5/max20/raw cumulative score/eval/FP32/TF32off.
- TEST ROI cache exact own notebook `ducanh2403/2-vi-cai-kia-qua-nhieu/143`: test_regions.npz7fb385e1, test_metadata.json4271b46b, vocabulary.json924be210. Full expected byte hashes/profile/order stored in RERANK_TEST_INPUTS.json, derived from the previously audited143 receipt. Stage only verified bytes, no rawfeature regeneration/latest fallback. TRAIN/VALID0 in cache; full5000COCO IDs/reference provenance must match fixed datasetJSON's TEST split.
- Same rawCOCO JPEG Dataset and detectorJSON as164: datasetJSON SHA2fd99922, detector498825da; map allTEST image names/IDs, hash images, record accepted detector confidences; missing/ambiguous/corrupt rows fail. No caption references, annotation labels or CIDEr enter beam generation or selection. References are isolated for post-selection metrics only.
- Same frozen CLIP57c216476eefef5ab752ec549e440a49ae4ae5f3, official JPEG processor and CLIPScore=2.5max(cosine,0); model/processor byte hashes from164 locked in inputs. No BLIP model/policy in this run. Formula inherited, not new alignment. Detector absences remain uncertain; CHAIR is object-category metric, not full semantic truth.

## User-run stages

One T4x2 notebook on the user's own account. Attach raw mscoco-2014(images+JSON) and objectdetectionecache, enable Secret(codex default). Kaggle2.2.5/cli.main + mock-check exact161/143 routes; download only onebest.bin and threeTESTcache files. Optional mounted roots may replace transport only if exact hashes match. No new Dataset/banks/ROI134/visualH5/BLIP required.

CPU metadata checks/tests then full input validation and written test_lock.json before any model execution. Decode all5000TEST images once with frozenCSA161, save all25000beams/IDs/rawscores with complete2500worker coverage and measured cap. References are not fed into decoding. Prefetch CLIP once using isolated temporary HF dependencies as164, validate exact assets, tokenize all25000texts without truncation. Score five candidates perimage and apply locked selector, same2500images/worker. Retain allscores and chosenranks; no automatic retry/training. Do NOT load or choose using VALIDoracle ranks/CIDEr or TESTreference scores.

## Evaluation and preservation

OfficialPTB/CIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/length for both full5000policies, identical references/IDF corpus. Save per-imageCIDEr/CHAIR counts and repair/newflag pairs for later conditional image bootstrap. No artificial equality gate to VALID scores, and no NLL change from reranking(same frozen captioner); no extra teacher-forced target pass. No oracle output or tuned policy in primaryTESTtable. Existing NoCSA10 vsCSA10 TEST is separate matched10XE study; do not attribute differences versusplain10 solely to CSA when this model also has15XE+SCST/rerank.

Last user quota16h before164(~16min scoringworkers, billing not inferred). Use2h whole-notebook STOPcap, decode and scoring projections,20min metric reserve; cap is not ETA/billing. Partialbeams/scores/logs preserved and markedINCOMPLETE, never a subset ranked as fullTEST. Keep wholeoutput including exact161weights/cachefiles/locks/scores/receipts; send smallreviewZIP excluding weights/NPZ/JPEG binaries. Author executes only metadata/NumPy/static/mock tests, not Torch/models/scorers/Kaggle. User-run compatibility/performance/gain pending.

## Prepared handoff

Source0f511bc; `CSA_CE161_FIXED_RERANK_TEST5000_2GPU.ipynb`16.5KB,17SHA-pinned files.6safe testsPASS repeated from the isolated exactGit bundle: checkpoint/cache exact bytes; featureloaderhash/no target arrays; fixedselector/reference independence/ties/nonfinite errors; full5000beam coverage/receipt/token/padding; lock rejects TEST-time coefficient changes; protocol metadata/sourceAST/no training or reference access in inference functions. Three empty-output codecells parse; both download regexes/own161+143route guards/supportedCLI/isolatedHFtarget/prepare→prefetch→decode→textcheck→CLIP→metrics order checked. Actual API/model/CUDA/scorer execution NOT performed by author. Keep both policies in result even if fixedselector disappoints; no conditional policy replacement. Output `/kaggle/working/csa_ce161_fixed_rerank_test5000`, send corresponding `_review.zip`.
