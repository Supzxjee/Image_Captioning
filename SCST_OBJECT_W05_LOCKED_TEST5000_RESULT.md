# Locked test5000: CIDEr gain reproduced; fixed hallucination reduction small

User-run locked evaluation complete; local artifact audit PASS. Same original QFormer and selected full-train object0.5 checkpoint, same beam5 and fixed reranking, no further training or parameter/checkpoint selection on these test results. Test split has prior project exposure; do not call it a never-seen independent test. One training seed.

## Primary result: fixed reranking on same5000 test images

| Metric | Original | Full-train object0.5 | Difference |
|---|---:|---:|---:|
| BLEU1 | .772729 | .775407 | +.002678 |
| BLEU2 | .615227 | .619664 | +.004438 |
| BLEU3 | .478446 | .482998 | +.004552 |
| BLEU4 | .369356 | .373879 | +.004523 |
| METEOR | .290791 | .295139 | +.004349 |
| ROUGE-L | .577798 | .583417 | +.005620 |
| CIDEr | 1.221815 | 1.247342 | +.025526 |
| CHAIRs | 3.48% | 3.44% | −.04 percentage points |
| CHAIRi | 2.3712% | 2.2879% | −.0833 percentage points |
| Object recall | 45.3131% | 45.8402% | +.5271 percentage points |

CIDEr increases about2.09% relative. Hallucinated captions174→172 andhallucinatedmentions180→175; recognizedmentions7591→7649. Correctunique supportedobjects7135→7218. Thus CHAIRI improves with both a small error-count reduction and increased denominator; do not describe the ratio change as a large reduction of hallucinatedobjects.

Paired fixed-caption outcomes:60 hallucinated captions corrected and58 introduced, net2 fewer.3312/5000captions change. This is a captionquality gain with roughly preserved fixed-policy hallucination rate, not robust evidence of substantial hallucination control.

## Secondary result: raw rank0

| Metric | Original | Object0.5 |
|---|---:|---:|
| BLEU4 | .370666 | .377579 |
| METEOR | .285039 | .289350 |
| ROUGE-L | .573118 | .577551 |
| CIDEr | 1.188245 | 1.212024 |
| CHAIRs | 4.32% | 3.98% |
| CHAIRi | 3.0299% | 2.7706% |
| Recall | 43.7508% | 44.4430% |

Raw hallucinatedcaptions216→199,mentions222→205,correctuniqueobjects6889→6998. Paired71fixes/54introductions, net17fewer.2832captionschange. Keep raw-vs-raw andfixed-vs-fixed comparisons; rawnewCIDEr should not be compared against a different baseline selection to claim the primary gain.

## Paired CIDEr uncertainty

5000paired imagebootstrap resamples withtestIDF fixed:

| Policy | CIDEr difference | Conditional95% interval |
|---|---:|---|
| Raw | +.023779 | [+.011139, +.036453] |
| Fixed (primary) | +.025526 | [+.012446, +.038868] |

Both intervals exclude0 for these checkpoints andimages. They do not include training-seed uncertainty, historical test exposure or adaptive experiment selection. No significance/noninferiority claim is made for the2-caption fixedCHAIR difference. The held validation-remainder4000 result was also positive (+.030365fixedCIDEr), but report the two datasets separately and do not average different-IDFCIDEr values.

## Verification

Audit `audit_scst_locked_test.py`: archiveCRC; pinned project21846bc; exactbaseline/selected/prompt/dataset/detection SHAidentities;selectedmetadata;lockedbeam/rerank andprimaryfixed;5000uniqueimage/cocoIDs with5refs;testsplitworkerconfig andbeamjournals;candidatecaptionmembership;officialPTB/CIDEr/BLEU/ROUGE recomputed4policies;CHAIRcaption/IDalignment andaggregatecounts;paireddeltasbootstrap. METEOR wascross-checked,notrecomputed. Weights/fullsource/detections absent: actualtensorforward,fixedrerankformula andCHAIRtokenization notindependentlyreplayed.

Source `qformer_scst_object_w05_test5000_locked_review.zip`,8625055bytes,SHA256 `fdc407b5e4855657c527d6640a73498b4a2e40cf125b183b0f1eabd74150521f`.

Preserved checkpoint:

```text
/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth
```

ExpectedlegacySHA256 `5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75`, confirmed in the user's evaluation protocol. OriginalZIPcheckpointSHA `6225193761e2519994e9922376db9a83b7b0e05b0cbe2cef251cdf166ce7804c`; conversion previouslyverified109tensors+metadataexact.

## Decision and claim boundary

Freeze this checkpoint andfixedpolicy as the improved experimental reference system. Do not alter the rewardcoefficient/reranking/checkpoint based on test. Preserveoriginal for ablations. A furtherseed could assessstability; it would be a new predeclaredexperiment, not anotherround oftesttuning.

This supports the claim that the fulltraining recipe improvescaptionmetrics over the originalbaseline in this run. It does not isolate the objectreward's contribution: the newlytrainedSCST+CE control was not evaluated here because thevalidationselection previously choseoriginalfallback. A directcontrol ablation remainsneeded for attribution. SCST/objectclassreward alone do not establish anewrelationprompt/query/alignment contribution or correctness ofinstance relations. No newtraining/inference/notebook waslaunched duringthisaudit.
