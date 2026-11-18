# TEST5000 of additional XE seeds

User confirmed seed43 source ducanh3/notebook7bc83c4c76/179 and seed44 source ducanh2403/2-vi-cai-kia-qua-nhieu/236. Download only best.bin through the pinned Kaggle CLI and verify the SHA256 from the completed review receipts.

Use the same Karpathy TEST5000 IDs, FP32 per-caption architecture, prompt181, patch profile, beam5/max20/no length normalization and metrics code as historical seed42 best9. Seed43 selects best8 and seed44 best7 by VALID CIDEr. Check their seed, epoch cursor and exposure before model initialization. No training or reranking is performed. Each notebook has a two-hour stop cap and preserves decoded captions before CPU scoring.

Report BLEU1/BLEU4/METEOR/ROUGE-L/CIDEr/SPICE/CHAIRs/CHAIRi and NLL. TEST references serve scoring only. Partial scorers do not trigger automatic re-decoding. GPU inference, source API access and external scorer installation are user-run verification.

## Routing correction, 18 November

The first seed43 TEST attempt stopped in an offline API mock because an inherited ROUTING_PREFLIGHT assignment overwrote the correct source-aware assignment. The inherited string expected kltnnnnnff/v9 and passed an obsolete descriptive source name. The API checkpoint download and model inference had not started. Remove the inherited assignment in both seed notebooks.

The regression check scans all cells, requires exactly one effective routing definition, executes the effective assignment and mocked preflight for both confirmed sources, and rejects a deliberately added second definition. This check passes locally without credentials, network or model execution. Actual API permissions remain user-run verification.

## Completed TEST results, 18 November

User confirmed TEST output sources: seed43 ducanh3/notebook7bc83c4c76/181 and seed44 ducanh2403/2-vi-cai-kia-qua-nhieu/238. These are separate from checkpoint sources179/236. Both complete with 5,000 unique caption IDs. Checkpoint SHA agrees with the full-training best receipt, inference locks, both workers and scoring lock. Every declared scoring payload hash and worker caption hash was checked. Aggregated worker NLL and CHAIR count ratios reproduce reported values. Caption cohorts, reference bytes, beam5/max20/no-length-normalization policy and evaluator file hashes match seed42.

Seed43 best8: CIDEr121.9843, CHAIRs4.06%, CHAIRi2.8035%, NLL2.178126. Seed44 best7: CIDEr121.6711, CHAIRs4.48%, CHAIRi3.1394%, NLL2.188389. Across seeds42/43/44: CIDEr121.7643±0.1913, CHAIRs4.1667±0.2759%, CHAIRi2.9149±0.1945%, NLL2.182884±0.005172. SD uses n−1 across training seeds, not a confidence interval. Full eight-metric table and provenance: CSA_PER_CAPTION_XE10_THREE_SEEDS_TEST_RESULT.json.

The measured stability applies to the XE captioner with beam top1. Additional seeds have no SCST/rerank result yet. Seed42 used the historical I/O path, while seeds43/44 used direct H5; no bitwise equivalence claim. The baseline differs in decoder architecture and currently has one seed. These measurements do not isolate the contribution of alignment or prompt, and do not establish paired multi-seed statistical significance against the baseline. Prior project TEST exposure remains recorded.

Results written and read back in existing Google Sheet, Kết quả mới!A198:P212. Exact numeric values and formatting metadata checked; native visual fit not inspected. No model, notebook or training was run locally during review. No formal report was edited.

## Approved fixed rerank of seeds43/44, 18 November

User approved applying the historical fixed XE reranker to the two additional seeds. Question: does the same decoder/CLIP/object selector improve each seed, and how variable is the XE-plus-rerank system across three seeds? Keep decoder/CLIP/object weights0.3/0.4/0.3, CLIP revision, detector file, confidence threshold and tie-breaking unchanged. No TEST tuning or new training.

The original TEST outputs181/238 contain only top1 captions, not five-beam payloads. Therefore regenerate five beams from the exact best checkpoints179/236 through the already verified API path. Require each rank's top1 caption file SHA to match the reviewed TEST output before CLIP scoring; any replay mismatch stops without changing tolerances. Preserve the regenerated beams and CLIP scores in output for CPU recovery or reuse. These notebooks require the existing visual-cache, MS COCO images/metadata, prompt-test-cache-v181 and objectdetectionecache; checkpoint access uses the matching owner's Kaggle Secret.

Reuse the successful seed42 rerank worker/data/selector and the seed-specific checkpoint contracts/API routing. Score both original beam top1 and the frozen rerank output on all eight metrics. This measures a fixed postprocessing effect within each trained seed; it does not isolate prompt or alignment training. Prepare notebooks locally and run static/mock checks only. The user runs T4x2 with a three-hour stop cap, not a runtime estimate.


### Rerank notebooks ready; offline verification

CSA_PER_CAPTION_SEED43_RERANK_TEST5000_API_2GPU.ipynb and CSA_PER_CAPTION_SEED44_RERANK_TEST5000_API_2GPU.ipynb are generated by make_seed_rerank_notebooks.py and copied byte-identically to Downloads. Both inherit the verified seed-specific checkpoint contracts and the historical seed42 beam worker, CLIP worker and selector. Output paths are seed-specific. After scoring, all original top1 metrics must replay within1e-7; both policies receive the same newly measured frozen-captioner NLL and explicit source provenance.

Ran test_seed_rerank_notebooks.py: four tests PASS. Checks compile every cell and embedded source, execute the effective mocked API routing179/236, verify source/hash pins and replay receipts, accept a synthetic complete5000-image beam cohort, reject wrong reference/checkpoint/top1/beam hash and metric inputs, and confirm the CLIP rerank source and beam worker match the historical successful notebook exactly. No Torch imports, model, network or credentials used by these tests. git diff --check passed. Actual Kaggle API access, CUDA replay and external scorers remain user-run verification. No result row was added to the Sheet for prepared-only notebooks.


## Fixed rerank completed, 18 November

User-confirmed new output sources: seed43 ducanh3/notebook7bc83c4c76/183 and seed44 ducanh2403/2-vi-cai-kia-qua-nhieu/240. Checkpoint sources remain179/236; original top1 TEST sources remain181/238. Both new run_status files are complete. Stored audit caches: CSA_PER_CAPTION_SEED43_RERANK_TEST5000_RESULT.json and CSA_PER_CAPTION_SEED44_RERANK_TEST5000_RESULT.json; combined summary: CSA_PER_CAPTION_XE10_THREE_SEEDS_RERANK_TEST_RESULT.json.

Verified root caption SHA against the prior TEST workers, beam payload SHA, all25000 candidate text/ID pairs per seed, CLIP score payload SHA and image identities, fixed selection replay on every5000 image, exact original top1 metrics, both policies' full CHAIR counts and transitions, per-image CIDEr means, reference/inference seals, checkpoint identities and NLL sums. The reviewer executed only standard-library data checks and the pure Python fixed scoring rule; no Torch/model/scorer or new inference. CLIP asset declarations match the frozen lock; model asset bytes are absent from the review ZIP and were not independently rehashed locally. Actual caption changes equal nonzero selected ranks:2903 for43 and2840 for44.

Seed43 CIDEr121.9843→124.3639, CHAIRs4.06→3.66%, CHAIRi2.8035→2.4509%;69 CHAIR repairs and49 new flags. Seed44 CIDEr121.6711→125.0599, CHAIRs4.48→3.90%, CHAIRi3.1394→2.6420%;70 repairs and41 new flags. Across42/43/44, rerank CIDEr124.8848±0.4591, CHAIRs3.6267±0.2914%, CHAIRi2.4528±0.1882%. Mean within-seed deltas: CIDEr+3.1205, CHAIRs−0.5400 percentage points, CHAIRi−0.4621 points. BLEU4 falls in43/44; mean37.8007→37.6460. NLL unchanged because rerank does not change model parameters. All SD values use n−1 across training seeds, not confidence intervals.

The fixed reranker improves CIDEr and both CHAIR rates in all three runs. This supports the postprocessing effect on this trained system, not an isolated causal prompt/alignment-training effect. No SCST for43/44. The single historical baseline differs in decoder and is not reranked; its comparison describes full systems and cannot assign the gain to one module. Prior project TEST exposure remains recorded. Existing Sheet Kết quả mới!A214:P229 written and read back exactly; formatting metadata checked, native visual fit not inspected. Formal reports untouched. No additional training or experiment was started.
