# TEST5000 of additional XE seeds

User confirmed seed43 source ducanh3/notebook7bc83c4c76/179 and seed44 source ducanh2403/2-vi-cai-kia-qua-nhieu/236. Download only best.bin through the pinned Kaggle CLI and verify the SHA256 from the completed review receipts.

Use the same Karpathy TEST5000 IDs, FP32 per-caption architecture, prompt181, patch profile, beam5/max20/no length normalization and metrics code as historical seed42 best9. Seed43 selects best8 and seed44 best7 by VALID CIDEr. Check their seed, epoch cursor and exposure before model initialization. No training or reranking is performed. Each notebook has a two-hour stop cap and preserves decoded captions before CPU scoring.

Report BLEU1/BLEU4/METEOR/ROUGE-L/CIDEr/SPICE/CHAIRs/CHAIRi and NLL. TEST references serve scoring only. Partial scorers do not trigger automatic re-decoding. GPU inference, source API access and external scorer installation are user-run verification.

## Routing correction, 18 November

The first seed43 TEST attempt stopped in an offline API mock because an inherited ROUTING_PREFLIGHT assignment overwrote the correct source-aware assignment. The inherited string expected kltnnnnnff/v9 and passed an obsolete descriptive source name. The API checkpoint download and model inference had not started. Remove the inherited assignment in both seed notebooks.

The regression check scans all cells, requires exactly one effective routing definition, executes the effective assignment and mocked preflight for both confirmed sources, and rejects a deliberately added second definition. This check passes locally without credentials, network or model execution. Actual API permissions remain user-run verification.
