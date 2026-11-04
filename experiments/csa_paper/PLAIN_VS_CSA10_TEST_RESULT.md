# Exact10XE NoCSA versus CSA — KarpathyTEST5000 COMPLETE

Reviewed04/11/2026; [numeric audit](PLAIN_VS_CSA10_TEST_RESULT.json). Baseline selfTEST ZIP SHA256 `ebca3334a96547e9526eafcd6b15bdd895c8aa2e064f3d141c42acdfb13702ce`; source46files at runtime recorded in `plain_self_test_review_text/source_receipt.json`. Baseline checkpoint from `ducanh3/notebook7bc83c4c76/45` SHAa75d7bad…, CSA epoch10 from139 SHA272f843a…. Both exact10epochs/177110updates, samefixedTEST5000/vocab/ROI/beam5max20; no checkpoint selection or training on TEST.

## Final same-TEST comparison

| Metric | NoCSA10 | CSA10 | CSA minus NoCSA |
|---|---:|---:|---:|
| CIDEr ↑ | .569220 | **1.019207** | +.449987 |
| BLEU4 ↑ | .150839 | **.333378** | +.182539 |
| ROUGE-L ↑ | .403299 | **.534649** | +.131350 |
| CHAIRs ↓ | **3.92%** | 6.18% | +2.26percentage points |
| CHAIRi ↓ | **4.4285%** | 4.6118% | +.1833percentage points |
| Object recall ↑ | 24.0633% | **41.4327%** | +17.3695percentage points |
| Teacher-forced TEST NLL ↓ | 2.784031 | **2.452753** | −.331277 |
| Hallucinated captions ↓ | **196** | 309 | +113 |
| False object mentions ↓ | **229** | 341 | +112 |
| Total generated object mentions | 5171 | 7394 | +2223 |
| Recovered class/image pairs ↑ | 3789 | **6524** | +2735 |
| Ground-truth class/image denominator | 15746 | 15746 | 0 |
| Mean words | 5.1618 | 8.1216 | +2.9598 |
| Captions≤2words | 1853 (37.06%) | 388 (7.76%) | −1465 |

CIDEr on publication×100scale is56.9220 versus101.9207. Keep normalization explicit; do not compare raw.569 to paper130.2. These are user-run officialscorer outputs; savedCSA CIDEr/BLEU/ROUGE replayed exactly under baseline'smetricstack before COMPLETE.

## What the comparison supports

Within this locked10epochbudget/single seed, replacing plainvisualencoder with the inheritedCSAencoder substantially improves reference-caption scores and objectcoverage. The noCSA model produces far more veryshort descriptions, so lower hallucinated-caption count alone is not evidence of better overall descriptions. CSAcovers2735more correctclass/imagepairs, but alsohas112more falsementions. Onimage flags, CSA repairs91 baselinehallucinatedcaptions and introduces204elsewhere, net+113. Therefore **CSA has not reduced objecthallucination here**. CHAIRi is also slightlyhigher, notimproved after normalizing bymentions. No statisticalequivalence/significance ormulti-seedgeneralization is inferred from pointvalues.

The result is not a newmethodcontribution: CSA is inherited. Equaldata/commoninitialweights/decoder/schedule/budget were checked, but plainhas39,010,576params and CSAencoderhasdifferentcapacity andadditionaloperations. This is an operationalencoder comparison, not isolationofsemanticpriors fromcapacity. Plainwasstillimproving late at10epochs; convergenceequivalence notestablished. Endpoint10 waslockedbeforeTEST, no post-hoc checkpointsubstitution. Futureextended-training experiments must be separatelypredeclared; do notchange the currenttable afterseeingTEST.

## Input equivalence and execution

Baseline notebookregeneratedTESTlocally withsamepreparecode/rawCLIP/detector/JPEGheaders, noAPItoCSA. NewTESTNPZ SHA **7fb385e1a03c5241c4d65cc240d0434e041284a9a829198c5167f0725be57b1a**, metadata4271b46b… andvocab924be210… match originalCSA143 **byte-for-byte** inuser-runchecks; onlyelapsedprepseconds differ. Full5000IDs/order/TESTsplitprofile consistent. Pooling1243.643s≈20.73min; inference/NLLworkers244.076/245.018s exclude setup/scoring andarenotquota-billing estimates. No train/foreigncacheAPI/optimizer used.

Audit69storedmetadata/text/countchecks+46Gitblob hashes PASS: fixedcheckpoint/epoch/batch/beam locks, immutableTESTreceipts, metadata/vocabbyteSHAs,5000predictioncoverage/token-captionmapping, CHAIRcounts/recall, workerpartition andNLLdenominator, samecohortmetricreplay andpairedhallucinationcounts. No agentTorch/model/officialscorer oractualNPZ/checkpointbinary read. NPZ/checkpointSHAverification was performedby theuser-run code; reviewexcludes thosebinaries. SourceSaveVersion ofthisselfTEST notprovided; originalbaselineweights45andCSA TEST143sourcesalreadyknown. Preservewholeoutputandoriginal45checkpoint.

## Next research decision

The two-armTESTbaseline is nowcomplete. Stoprequesting anotherbaselineTESTorblindextraXE. Nextbrainstorm an explicitimprovement toCSA'salignment that aims toreduceCHAIR whilepreservingCIDEr andrecall; propose matchedcontrols beforetraining. Do notrebrandSCSTor inheritedCSA asnewcontribution. No additionalGPUtraining/module/RL started inthisreview.
