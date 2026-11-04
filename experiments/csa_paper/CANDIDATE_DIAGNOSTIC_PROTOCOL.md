# CSA caption candidate diagnostic before reranking

05/11/2026. User approves checking five candidates before introducing a reranker; reports16 GPU hours remaining and checkpoint not saved as Dataset. Exact source `ducanh2403/2-vi-cai-kia-qua-nhieu/161`, not latest. No training, TEST, new model or automatic subsequent run.

## Fixed comparison

Does CSA15 + SCST+CE already generate better captions than selected rank0? Preserve model, per-image beam5, cumulative raw log probability, max20 actions, FP32, TF32 off, eval. Full Karpathy VALID5000. Checkpoint SHA58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b; full1 end14161updates/113287events, SCST_CE. Banks are embedded. Stage only fullROI134 VALID, no TRAIN stage/rawimages/detector/CLIP extraction. Two user-run T4 workers retain original order/batches8/rank interleave. Every top1 caption/token content must match saved endpoint; score error above2e-5 stops. Model.py SHA139edbb1 unchanged.

## Measurements and interpretation

Reproduce rank0 CIDEr1.0631029131024006, BLEU4.34829378731737376, ROUGE-L.5449286537695498, CHAIR counts260/267/7145/6457/15928. Score each rank separately with official PTB/CIDEr and exactly the SAME5000 reference keys/lists: never change IDF by expanding to25000 candidate IDs. For each image choose highest CIDEr candidate, earlier rank on ties, then rescore that assembled set including CHAIR/recall. Label **reference-dependent oracle, diagnostic only**, not deployable reranking or achieved method score; maximal CIDEr need not minimize hallucination.

Report candidate diversity, all-five-short images, rank0 short<=2words, oracle chosen ranks/changedcaptions. Preserve all5 texts/tokenIDs/rawscores/per-imageCIDEr for future real reranking without re-decoding. No ITM/CLIP/object selector, coefficients, learned ranker or retrieval/facts module in this first diagnostic. User separately decides actual selector after seeing headroom; inference must not use references/GT. No promised120CIDEr or arbitrary success threshold.

## Runtime and preservation

User-run T4x2/Internet, fullROI134 Dataset, enabled Kaggle API Secret for exact161; no new Dataset needed.2h whole-notebook STOPcap with measured projection after32images/rank plus20min metrics reserve; not ETA/billing. Preserve partialJSONL/logs/reviewZIP on failure, no auto retry. Local checks metadata/NumPy/mock only; actual Torch/model/scorers user-run. Publish pinned notebook after safe checks. Results and reranker decision pending user output.
