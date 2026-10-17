# Full-train object reward0.5: baseline improved on validation remainder4000

User-run full-train comparison complete. Local artifact audit PASS; the predeclared locked-selection fixed-policy gate PASS. This is a positive development-validation result from one seed, not test evaluation, multi-seed confirmation or proof of scientific novelty.

## Main result: same4000 images after checkpoint selection

| Policy | BLEU4 | METEOR | ROUGE-L | CIDEr | CHAIRs | CHAIRi | Object recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original raw | .371621 | .282769 | .571750 | 1.175837 | 4.45% | 3.1131% | 42.6375% |
| Object0.5 raw | .374134 | .284945 | .573309 | 1.195705 | 3.95% | 2.7435% | 43.2965% |
| Original fixed | .371555 | .289799 | .577056 | 1.208609 | 3.50% | 2.3522% | 44.4811% |
| Object0.5 fixed | .377003 | .293647 | .581614 | 1.238974 | 2.95% | 1.9705% | 44.7949% |

Fixed CIDEr improves+.030365 (about2.51% relative). Hallucinated captions decrease140→118 (22fewer,15.7% relative); hallucinated mentions142→119. Total recognized mentions6037→6039, so lowerCHAIRi reflects fewer hallucinated mentions rather than denominator growth. Correct unique supported objects increase5670→5710. Both quality and hallucination improve at the measured operating point.

Raw also improves:CIDEr+.019867; hallucinated captions178→158 andmentions180→160;correct unique objects5435→5519.

## Paired evidence

5000paired bootstrap samples of images, held evaluation-IDF fixed:

| Object0.5 minus original | CIDEr difference | Conditional95% interval | Hallucinations fixed / introduced |
|---|---:|---|---:|
| Raw | +.019867 | [+.006028, +.034102] | 66 / 46 |
| Fixed | +.030365 | [+.015836, +.045255] | 55 / 33 |

Both intervals exclude0 for this checkpoint/image pool. They do not cover training-seed variability, historical validation reuse or adaptive experiment selection. Do not state a universally significant model improvement from this alone.2651/4000fixed captions change;2340raw change.

## Which checkpoint was selected?

Both arms trained14161updates on113287train/restval images with mid snapshot7081 and final snapshot. Input checkpoint/prompt/dataset/detection identities match the original baseline; both start from the same model identity and report the same strictTeslaT4 runtime. Objective coefficients, reward formula, histories and finite updates rechecked.

Checkpoint selection was computed on tune1000 only using the declared fixedCIDEr rule, withCHAIRs/i<=original andrecall>=original−.01, baselinefallback andearlierties:

| Candidate | Tune1000 fixedCIDEr | CHAIRs | Outcome |
|---|---:|---:|---|
| Original | 1.275819 | 3.8% | Available fallback |
| SCST+CE mid | 1.274322 | 3.3% | Eligible, below originalCIDEr |
| SCST+CE final | 1.269167 | 4.1% | Ineligible |
| Object0.5 mid | 1.275019 | 3.2% | Eligible, below originalCIDEr |
| Object0.5 final | 1.286670 | 3.8% | Selected |

The selected **SCST+CE policy falls back to original**. Its4000-image predictions are byte-equivalent at the parsedcaption level to original. Thus the remainderresult shows the object model beats original and the declared baseline-fallback control policy; it does **not** show superiority over an independently retained newly trained SCST+CE checkpoint on these4000images. Do not present the duplicate control row as separate corroborating evidence for the object reward's isolated contribution.

`locked_selection.json` hashes, rule outcome and evaluator-config checkpoint paths agree. The pinned runner writes this artifact before remainder evaluation; physical execution chronology is not independently established from review artifacts alone.4000images are disjoint from this run's1000selectionimages, but may have been inspected historically in validation5000. Tune and remainder metrics use separateCIDErIDF; do not average them to produce a5000-imageCIDEr.

## Preserve this checkpoint now

```text
/kaggle/working/qformer_scst_ce_object_w05_fulltrain/scst_ce_object/checkpoint.pth
```

SHA256 `6225193761e2519994e9922376db9a83b7b0e05b0cbe2cef251cdf166ce7804c`.

This is the final object-reward checkpoint, not mid or the originalfallback. Keep it in a separate stable Kaggle output/dataset store before later notebook versions replace the current output. Also preserve `locked_selection.json`, both arms' mid/final checkpoints and the reviewZIP. No storage notebook was submitted by the assistant.

## Audit scope

`audit_scst_full_train.py` verified ZIPCRC, protocol/input identities, unique/disjoint113287/1000/4000imageIDs, target-row provenance identities andmetadata, annotation-cacheSHA, targetSHA, both14161-updatehistories, CE/SCST/objectreward arithmetic, strict runtimes/initialidentities, all16policycaptionmetrics(CIDEr/BLEU/ROUGE viaofficialPTB), beamjournals/candidatecaptionmembership, CHAIRcaption/IDalignment andaggregatecounts, checkpointselection andselectedcheckpointSHA/evaluatorconfigconsistency. METEOR was cross-checked only, not recomputed. Sourceannotations/detections andtrainedweights are absent; originaldataset splitprovenance, fixedrerankingformula, actualparameterupdates andCHAIRtokenization cannot be independentlyreplayed.

Training summaries report6679.58s control and6940.51s treatment, excludingnotebooksetup andevaluation. FulltrainrewardIDF differs from20kpilot; improvement cannot be assigned solely to data volume versus moreupdates/changedIDF.

Source `qformer_scst_ce_object_w05_fulltrain_review.zip`,52232257bytes; SHA256 `c3fd4904f8a597b4ed6a50a1e124d4d5b03e27dd89c1ca17c78463f3a059ea5b`.

## Next decision

Freeze the finalobjectcheckpoint, coefficients andfixedreranking. First preserve thecheckpoint; then confirm on an additionaltrainingseed or a separately locked evaluation protocol before broader claims. No test results are reported or used for selection here. A direct objectrewardablation would need the newly trainedSCST+CE checkpoint evaluated under a predeclared rule, explicitly secondary after the originalfallbackselection; do not silently change the currentprotocol to claim that ablation succeeded.

This establishes a useful stronger system component. SCST andobjectreward alone do not establish a new semantic-query/relation-alignment contribution; that contribution still requires its own method and ablations on top of the frozen improved system.
