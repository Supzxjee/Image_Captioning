# CSA161 beam candidate diagnostic on VALID5000

Reviewed05/11/2026. User run COMPLETE. [Locked protocol](CANDIDATE_DIAGNOSTIC_PROTOCOL.md), [numeric audit](CANDIDATE_DIAGNOSTIC_RESULT.json), [audit script](audit_candidate_result.py). Source43f48da, exact SCST+CE full1 END161 checkpoint58ce5dc0. ZIP SHA6a224d60284a9fc6dd9fd20fbf914139eef0b3e94c882d9918106b9458cba19f, CRC checked once and text cached locally. User confirms diagnostic SaveVersion163: `ducanh2403/2-vi-cai-kia-qua-nhieu/163`, output `/kaggle/working/csa_ce161_candidates_valid5000`. This is the candidate cache version, distinct from weight source161. Preserve full163output; no second caption decode needed for reranking.

**There is substantial CIDEr selection headroom among the five existing beams. No deployable reranker has been evaluated.** Rank0 reproduces CIDEr106.31; the reference-dependent best-of-five oracle reaches141.43. The35.12point difference is a diagnostic ceiling for this particular pool/metric, not an achieved improvement, inference policy, prediction of attainable gain or a fair comparison with another report's actual121.95. References are used to choose oracle captions; they cannot be used by a deployed selector.

## Same checkpoint, candidates and full VALID cohort

| Policy | CIDEr ×100 | BLEU4 | ROUGE-L | CHAIRs | CHAIRi | Recall | Hallucinated captions | Captions<=2words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current rank0 |106.3103|.348294|.544929|5.20%|3.7369%|40.5387%|260|407|
| Always rank1 |106.5345|.344043|.550217|5.68%|3.8602%|41.9701%|284|141|
| Always rank2 |105.5396|.333408|.550746|6.72%|4.3861%|43.2007%|336|31|
| Always rank3 |103.2182|.318413|.547224|6.62%|4.2492%|44.0105%|331|4|
| Always rank4 |98.5638|.300119|.541050|7.10%|4.3091%|45.0213%|355|0|
| Reference CIDEr oracle — diagnostic ONLY |141.4344|.467008|.621428|4.84%|3.2636%|44.6949%|242|61|

All5000images have five distinct caption strings (not proof of five semantically diverse/accurate alternatives). Oracle chooses ranks0/1/2/3/4 on1165/1133/1038/927/737images, respectively.3835images(76.7%) have a strictly higher reference-CIDEr candidate than rank0. This supports testing candidate selection, not the claim that these alternatives are visually correct.

No image has all five captions<=2words. Oracle changes347short rank0 captions to longer ones and one longer caption to short:407→61. Mean words8.4164→9.3776. Later beams grow steadily longer(8.989/9.713/10.534/11.640words), but fixed rank1 improves CIDEr by only.2243points while increasing hallucinated captions260→284. Merely taking a lower rank or choosing longer text is not supported as a joint-quality solution.

Oracle's hallucination change is modest:79rank0flags repaired and61new introduced, net18fewer(260→242). False mentions267→259, recalled class/image pairs6457→7119, ground-truth denominator15928 unchanged. Maximizing reference CIDEr is not a hallucination oracle and does not validate relation/attribute correctness. No claim of a reliable hallucination benefit or human-confirmed quality.

## Verification and practical limits

73storedmetadata/text/count checksPASS, including11Git source hashes, exact5000reference lists/vocabulary, disjoint2500worker IDs, candidate-fileSHA, all25000token/text/rawscore checks, all5000top1matches with scoreerror0, historical per-image rank0CIDEr equality, five-rank meanCIDEr consistency, oracle argmax/ties/selectedcaption/mean replay, six CHAIR caption alignments/count aggregations and commonGT sets, diversity/length/choicehistograms. User log:7safe tests, Kaggle2.2.5 exact161routing mock, full5000historicalmetric preflight PASS before decoding. The prior CLI failure was repaired; this ZIP is the successful fresh source43f48da run.

Both user workers complete in221.4/219.8seconds; these figures exclude setup/API/staging/CPUscoring and are NOT billed GPU hours. No training, parameter update, TEST or actual ITM/CLIP/object reranking. No author Torch/model/metric scorer execution. CIDEr values checked against saved per-image scores, not independently recomputed with official scorers; BLEU/ROUGE oracle values are stored user-run outputs. CHAIR counts recomputed from stored word lists, not image/annotation re-parsing. Weights/features absent ZIP; reported binary hashes remain user-run evidence.

## Recommendation, not a new run authorization

Prioritize a real inference-only reranker over immediately retraining the captioner, because this pool has measured CIDEr headroom. Retain all25k candidates/scores, per-imageCIDEr and references from this output; no second caption decode is necessary. Compare unchanged decoder-rank0 with the old fixed CLIP+object selector and an image-text matching alternative such as BLIP ITM, using identical candidates and reporting CIDEr/CHAIR/recall/length jointly. Additional image input/model inference and selector coefficients need a separate agreed implementation/protocol; none is silently chosen from oracle labels. Use VALID for any selector decisions, freeze before TEST. If gain fails, distinguish ranking failure from candidate limitations; the oracle alone is not a guarantee of120CIDEr.
