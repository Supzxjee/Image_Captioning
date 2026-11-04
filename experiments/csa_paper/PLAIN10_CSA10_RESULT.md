# NoCSA10 training + CSA10 KarpathyTEST — COMPLETE, 04/11/2026

Two authorized independent user-run notebooks completed. Source `63e11dbe9a17ef035ad72d7d404757ec03432711`,33 identical source pins. [Audited result](PLAIN10_CSA10_RESULT.json), [locked protocol](PLAIN_VS_CSA10_PROTOCOL.md). No baselineTEST result yet: do not compare baselineVALID to CSA TEST as a same-cohort effect.

## Actual CSA epoch10 TEST5000

| Metric | Value |
|---|---:|
| CIDEr | **1.019207** (101.9207 on ×100 scale) |
| BLEU4 | .333378 |
| ROUGE-L | .534649 |
| CHAIRs | 6.18% |
| CHAIRi | 4.6118% |
| Object recall | 41.4327% |
| Hallucinated captions | 309/5000 |
| False object mentions | 341/7394 generated mentions |
| Correct class/image coverage | 6524/15746 |
| Teacher-forced TEST NLL | 2.452753 |
| Mean words / at most2words | 8.1216 /388captions |

Exact original139 latest checkpoint272f843a…/177110updates, frozenepoch10/beam5/max20; notbest9 or15. FullTESTmetadata has5000unique IDs, disjoint from archivedVALID5000, source_split=test. The original dataset split/detector SHA was verified by user-run preparation. SameCLIPprofile/vocabulary/roi_pool; no newvisualmodel. Cache preparation1384.842s≈23.08min; inference/NLL workers260.809/268.621s. These exclude setup/scoring and are not quota-billing estimates. No gradients/optimizer or TESTbasedselection.

TEST cache payloadSHA: `test_regions.npz` **7fb385e1a03c5241c4d65cc240d0434e041284a9a829198c5167f0725be57b1a**; metadata4271b46b…; original vocabulary924be210…. Preserve whole `/kaggle/working/csa_epoch10_karpathy_test` folder including NPZ, metadata, receipt andvocabulary. ReviewZIP excludes actualNPZ bydesign; array bytes not locally reopened. User reports exact source `ducanh2403/2-vi-cai-kia-qua-nhieu/143`; must verify payloadSHA when transporting that version, never substitute latest.

## NoCSA training finished, TEST still pending

Fresh10epochs/177110updates/5667470TRAINcaption events, no TEST consumed.39,010,576trainableparameters; sameROI134/vocab/shareddecoder/head/initialcommonweights/Adam/LR/global32/FP32/seed/epochs as documented. Plain3visualTransformer layers/shareddecoder memory; no semanticbankforward. Actual-input plain smoke/commoninitialization/finitegradients and sixDDPguards passed. Parametercount and activeencoderdepth/operations differ fromCSA; equalepochs is not parameter/computematching.

| Same VALID5000, exact epoch10 | NoCSA | CSA |
|---|---:|---:|
| CIDEr | .560024 | .997179 |
| Observed TRAIN NLL | 2.502177 | 2.417765 |
| VALID NLL | 2.768793 | 2.435452 |
| Mean words | 5.3292 | 8.0850 |

This is a useful same-epoch development comparison, not finalTEST evidence. NoCSA improving late (epoch9 .437647→10 .560024) and shorter descriptions mean convergence cannot be assumed. Do not change the locked10epoch endpoint after seeing TEST or claim a newalignmentmethod from this inheritedCSA comparison. One seed, different parametercapacity; large validationgain alone does not isolate semanticalignment from all architectural differences.

Plain best/latest both **a75d7bad5a0d8dc976546b70145a29d76748142ad82190f4f04180418987135f**, validatedepoch10; nextcursor11 is not an extraepoch. Worker15883.059s≈4.412h includingvalidation/checkpointoverhead, not billing. Exact source confirmed `ducanh3/notebook7bc83c4c76/45`. Preserve `/kaggle/working/plain_full_xe_10` checkpoint+receipts/provenance.

## Required next step

Save exact baseline45 and TESTcache143 as reusable inputs, verifying expectedbinary/NPZ/metadata SHA. Then generate baselineepoch10 TEST captions using the **same** cache, vocabulary, beam5/max20 and scorers. Only after this can a final NoCSA10 vsCSA10 TEST table report CIDEr andhallucination changes. No retraining or extraepochs is required for that inference. Dataset transport/baselineTEST notebook not prepared/executed in this review. No newmethodbranch/RL launched.

Audit98storedmetadata/text/count checks and33Gitblobs passed: all10VALIDpredictioncohorts, originalshareddata/software identities, plaininitialization/gradients/DDP receipts,TEST5000coverage/vocabulary/metadataSHA/profile, token-caption mapping, CHAIR counts/recall, teacherNLLtoken counts/weighted mean and exact checkpoint locks. No localTorch/model/officialscorer or rawfeature/checkpointbinary reads. Primarymetric values are user-run officialscorer outputs, not an independent localrescore.

ZIP SHA256: plain `d8952be8a1963f2bf0350784c92ae39da6959cdc11fb905fb1734dbe27cf7634`; CSA TEST `956010e67d0587c083ce053db38cd5d9087b2d4ebe16cf90241e5cb0c91a0f6a`.
