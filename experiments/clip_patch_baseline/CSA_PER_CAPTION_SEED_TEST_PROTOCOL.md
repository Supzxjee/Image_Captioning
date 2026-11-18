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
