# Inference-only reranking of the fixed CSA161 VALID candidates

05/11/2026. User approves the proposed three-way comparison after R246. Reuse exact candidate output `ducanh2403/2-vi-cai-kia-qua-nhieu/163` from weights161, not latest. No CSA loading, caption regeneration, training, weight changes, oracle-based selection or TEST. Last reported user GPU budget16h;2h whole-notebook STOPcap with measured projection and20min metric reserve, not ETA/billing.

## Locked selectors and inputs

Exactly the same5000VALID images and25000captions, original beam5/max20/logprobs. Three policies, earlier beam on ties:

1. Decoder rank0, historical CIDEr1.0631029131024006/CHAIRs.052/CHAIRi.03736878936319104/recall.4053867403314917/260flags. Must replay exactly before model inference.
2. Existing `captioning.multiscore_reranking.rerank_multiscore`: .3minmax(raw decoder logprob)+.4minmax(CLIPScore)+.3normalized object consistency. CLIPScore=2.5max(cosine,0); original parser/index_detections/conf>=.5/hallucination_penalty1. Formula/weights fixed from old QFormer selector, not tuned on this output. Detector absences may be misses/topK truncation; unsupported labels are not ground truth. Evaluate recall/length alongside CHAIR.
3. BLIP ITM alone: argmax positive-class logit minus negative-class logit, equivalent to argmax binary positive probability; earlier beam on ties. No fusion coefficient, fallback or tuning added after seeing metrics. Save both logits/margin/probability for audit.

Raw images from attached `mscoco-2014` (default `/kaggle/input/datasets/vuthetam/mscoco-2014`), dataset_coco.json SHA2fd99922, detectorJSON SHA498825da. Check full5000filename/VALID/COCO-ID/reference identities and image headers/SHA before inference; missing/ambiguous/corrupt input fails, no web-image substitution/subset score. Candidate file SHAs9010f882/7b350462 fixed from163, historical references/vocabulary/top1token/text/logprob identical to R246. Inference payload contains candidates/logprobs/detector output/raw image only; no reference captions, COCO annotations or per-imageCIDEr/oracle ranks. References and CHAIR cache are used only in the post-selection scorer.

## Frozen scoring models

- CLIP `openai/clip-vit-base-patch16`, revision57c216476eefef5ab752ec549e440a49ae4ae5f3, official processor. Fresh image forward, unlike historical cached-CLS scoring; reuse formula/weights, not a claim of exact historical CLIP-score replay. Processor/preprocessing identities recorded.
- BLIP `Salesforce/blip-itm-base-coco`, revisionbed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c, `BlipForImageTextRetrieval(use_itm_head=True)`, positive class1. Official processor. Model card identifies COCO fine-tuning; external checkpoint split provenance not independently audited, no claim of unseen/pretraining-free evaluation.
- Both eval/frozen/FP32/TF32off, two independent T4 workers,2500images each; no DDP collectives or optimizer. Five image-text pairs per image, no candidate truncation (fail if token limits exceeded). Fixed image/pair order; save raw image SHA, score vectors and chosen beam ranks. Models/processors prefetched once before workers, exact downloaded file hashes recorded.
- Isolated temporary Python dependency target: transformers4.57.1/tokenizers0.22.1/huggingface-hub0.36.2/safetensors0.6.2, installed with --no-deps; do not replace global Torch/Gradio dependencies. Kaggle2.2.5/cli.main and exact163 request routing mock for transport.

## Evaluation and decision

Official PTB/CIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall for all three full5000sets, identical reference lists and IDF corpus. Report changed captions, repaired/new CHAIR flags, selected ranks, mean length/short captions and per-imageCIDEr. No NLL comparison (same frozen captioner). Baseline replay before scoring models. Conditional paired bootstrap if useful can be done from stored metric/count outputs, without rerunning models. No guarantee of oracle141.43/120CIDEr, no arbitrary pass threshold. Inspect jointCIDEr/CHAIR/recall; freeze any later chosen policy before a separately approved TEST run. Preserve complete score cache and small reviewZIP, no automatic retry or further run. Author only metadata/NumPy/static/mock checks; actual downloads/models/scorers user-run.

Primary sources: [BLIP model card](https://huggingface.co/Salesforce/blip-itm-base-coco), [Transformers4.57.1 BLIP implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/blip/modeling_blip.py), [Salesforce ITM example](https://github.com/salesforce/BLIP/blob/main/demo.ipynb). Revisions/package metadata checked05/11; no author model execution.

## Prepared handoff and verification

Sourcef72d54d, notebook `CSA_CE161_RERANK_VALID5000_2GPU.ipynb`14.7KB,13SHA-pinned source/assets including3existing captioning modules.6safe testsPASS repeated from an isolated exactGit bundle with cleanPYTHONPATH: fixed policies/reference independence; margin/ties/probability saturation; invalidscores/order; image path/missing/ambiguous/escape; exact full5000candidate file/token/coverage/tamper; AST/inference no training/references. Three empty-output notebook codecells parse; supportedCLI+exact163regex, isolatedHFtarget and metricpreflight→prefetch→worker ordering checked. No author neural/scorer/API/notebook execution. Runtime model downloads/compatibility/throughput/quality remain user-run checks, not certified by static tests. Last reported budget16h, no new train quota assumed. Output `/kaggle/working/csa_ce161_rerank_valid5000`, send corresponding `_review.zip`, retain scores/modelsreceipt/image identities for later reuse.
