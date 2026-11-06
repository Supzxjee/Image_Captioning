# Frozen prompt TEST5000 result

06/11/2026. User confirms CONTROL output184, PROMPT output61. Frozen checkpoints remain CONTROL180/PROMPT60. All six prelocked conditions complete on same KarpathyTEST5000. No training/TEST-based policy tuning.

| Condition | CIDEr x100 | BLEU4 x100 | ROUGE-L x100 | CHAIRs % | CHAIRi % | Recall % | Flagged captions |
|---|---:|---:|---:|---:|---:|---:|---:|
| CONTROL/normal | 105.5606 | 34.3829 | 54.0408 | 5.6200 | 4.1556 | 40.8294 | 281 |
| PROMPT/normal | 107.3603 | 34.6452 | 54.4814 | 3.7600 | 2.6765 | 42.4108 | 188 |
| CONTROL/fixed_rerank | 114.3671 | 36.1105 | 56.3504 | 4.5400 | 3.0988 | 44.0556 | 227 |
| PROMPT/fixed_rerank | 115.3522 | 36.1731 | 56.6351 | 3.2000 | 2.1181 | 45.1797 | 160 |
| PROMPT/zero | 89.3710 | 29.4661 | 50.1775 | 5.1400 | 4.5809 | 32.8591 | 257 |
| PROMPT/shuffle | 89.6182 | 29.9002 | 50.5937 | 10.7800 | 8.8281 | 34.2373 | 539 |

Main point differences: raw PROMPT-CONTROL CIDEr+1.7997points, CHAIRs5.62->3.76%(281->188;93fewer), recall40.8294->42.4108%. Fixed rerank: CIDEr114.3671->115.3522(+.9851points), CHAIRs4.54->3.20%(227->160;67fewer/-29.52%), recall44.0556->45.1797%.

Same PROMPT60 weights: normal CIDEr107.3603/CHAIRs3.76% vs zero89.3710/5.14% vs shuffle89.6182/10.78%. Normal-zero CIDEr+17.9894points; normal-shuffle+17.7421points. Foreign text raises hallucinated captions188->539. This supports useful image-specific text conditioning at inference. Geometry/ROI/masks/weights remain fixed; donors follow prelocked global hash42 role-specific foreign-image slots, not within-batch roll.

Do not equate sensitivity with a fully matched training contribution: zero/shuffle can be out-of-distribution, adapter capacity and initialC10/P9 training histories remain confounded. One seed and prior project TEST exposure are disclosed. CHAIR measures objects, not relation truth. These are inherited CSA alignment/prompt components, no new-method/SOTA claim.

Historical best CSA15+SCST+CE+rerank TEST117.0633/CHAIRs4.40%(220flags) still has higher CIDEr than newPROMPT115.3522, while newPROMPT has lower CHAIRs3.20%(160flags). Different initial training/history, so not an isolated prompt-gain subtraction.

Preserve CONTROL184 andPROMPT61 entireoutputs, beams, CLIP scores and logs; weights remain180/60. Recommend freezing this candidate and writing the experiment/results section before further changes. No automatic new training, donor/selector tuning, or TEST-based fallback.

Audit/conditional paired bootstrap details are in PROMPT_TEST_RESULT.json and audit_prompt_test_result.py. No author Torch/model/CLIP/official scorer execution. Reference/caption identity, source and stored-score/count replay only. No teacher-forced NLL was measured in these inference notebooks.

Audit complete:139checks PASS, both5000-image runs/6conditions, source40pins/arm, exact endpoint/cache/reference/donor locks, all10000 fixed choices replayed. Conditional2000paired-bootstrap95%CIs (unadjusted): raw PROMPT-CONTROL CIDEr[+0.5089,+3.1007],CHAIRs[-2.4605,-1.28]pp; rerankCIDEr[-0.3863,+2.4276],CHAIRs[-1.92,-0.7795]pp. Thus raw CIDEr difference is directional under this conditional check, rerank CIDEr remains uncertain; neither is training-seed uncertainty. Normal-zeroCIDErCI[16.3480,19.5535];normal-shuffle[16.2102,19.2020]. Donor slots 16821, unchanged donor strings 2105; different source image does not guarantee a different string.
