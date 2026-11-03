# CSA XE15 continuation — COMPLETE, 04/11/2026

User-run fullTRAIN extension10→15 and subsequent CPU quality scoring both complete. ZIP SHA256 `f62e93b028fb1a03dbc4716996e9215b37adad99ce74516d58e97334df08a2ad`. See [audited numbers](XE15_RESULT.json), [protocol](XE15_CONTINUATION_PROTOCOL.md), small receipts in `xe15_review_text/`.

Exact latest139 checkpoint/Adam/both-rank RNG resumed via explicit identity migration. Added88,555updates; absolute265,665updates and8,501,205caption events across15epochs. Old1–10 history matches archived run exactly. Model/source/loss/FP32/global32/beam5max20 unchanged; no RL or TEST. Worker session10,875.001s≈3.021h, excluding setup/CPUquality and not a billablequota estimate.

## Full5,000 validation, same references and evaluator

| Epoch | CIDEr ↑ | Train NLL ↓ | Valid NLL ↓ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Hallucinated captions |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9 (old best) | 1.002933 | 2.441058 | 2.446409 | 5.66% | 4.095% | 40.149% | 283 |
| 11 | .988057 | 2.395469 | 2.426740 | 6.20% | 4.645% | 38.957% | 310 |
| 12 | .998213 | 2.375002 | 2.420166 | 6.90% | 5.086% | 39.597% | 345 |
| 13 | .997090 | 2.356313 | 2.413761 | 5.92% | 4.219% | 40.545% | 296 |
| 14 | .997158 | 2.340132 | 2.405592 | 6.34% | 4.856% | 38.963% | 317 |
| 15 (selected by CIDEr) | **1.009453** | **2.325270** | **2.405520** | 5.86% | 4.441% | 40.118% | 293 |

Epoch9→15 CIDEr+0.006520 (equivalent+.652 on paper ×100 scale), hallucinatedcaptions+10, falsementions303→315, correctclass/image coverage6395→6390. Captions average8.263→8.221words; ≤2word409→425; empty0→0. BLEU4 .335528→.332274 and ROUGE-L .535968→.535130. Paired caption flags:146hallucinations repaired but156introduced. These are point differences on reused validation, not significance/multi-seed/test claims.

The15epoch XE baseline is now trained. Loss improves, but only a small CIDEr gain and no hallucination reduction: the joint quality objective has not been achieved by training longer. This does not by itself establish overfitting or identify feature/model/decoding root causes. Not an exact full published CSA reproduction: cachedCLIPROI/frozenbackbone differs and the paper's30epochRL stage remains undone. Do not compare validation1.00945 directly to published test1.302/1.452 as same-setting scores.

## Checkpoint preservation and recommendation

Selected best is epoch15 by the predeclared CIDEr rule, not by CHAIR. `best.bin` and latest `checkpoint.bin` both report SHA256 **e02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25**. Receipts say cursor epoch16 because that is the next epoch, not a completed16th epoch. One binary copy suffices if both receipts/provenance are retained and required filename aliases are restored before a future resumable run. Preserve original139 best9 as the old hallucination comparison.

Recommendation: stop extending XE without a new hypothesis; preserve15endpoint. If continuing baseline reproduction, next discuss a bounded SCST pilot before committing to30RL epochs, with a matching continuation control and CHAIR/recall monitoring. SCST is an inherited optimization phase, not a novel alignment contribution. No SCST/newmodule/GPU budget was automatically started by this review. Asked user for exact Kaggle SaveVersion; provenance pending that answer.

Audit:79 storedmetadata/text/count checks and31source Git hashes passed; exact originalhistory, references/vocabulary/best9caption bytes, six5000prediction cohorts, token-caption consistency, CHAIR counts/recall/length and best/latest receipts verified. No local Torch/model/official scorers or feature/binary checkpoint loads. Original checkpoint bytes are not in review; their SHA checks are user-run runtime evidence, not independent local byte verification. Core trainer `chair_computed=false` describes training; separate `quality_result.json` and runstatus correctly report CPUscoring complete.
