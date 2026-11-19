# Seed43 VALID audit and SCST+KL500 pilot

Approved19 November2026 after the three-seed SCST tradeoff review. User runs Kaggle T4x2; no local model/notebook/Torch execution. Keep architecture, cached inputs, TRAIN order, rollout/reward and reranker unchanged.

## Hypothesis and sources

Pure CIDEr-D SCST raises post-rerank CIDEr but worsens both CHAIR rates in all three runs. Test whether a frozen copy of the same seed's XE policy limits distribution drift and retains lower-hallucination behavior. Drift is a hypothesis, not an established error cause. KL is not a new visual alignment module or grounding guarantee.

Screen seed43 only, without representative-seed claims. XE: ducanh3/notebook7bc83c4c76/179, SHA a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5. Pure-SCST500 control: same notebook/version185, update0500/pilot_checkpoint.bin, receipt SHA8f1a1fc181b335543d8be117da1315260cdc01b4d4fd4b464d8367a82003d500. New KL arm starts fresh from XE, never from control/SCST1500. Same Adam5e-6/betas(.9,.99)/eps1e-8, batch8, FP32, grad clip1, first4000 TRAIN cache events, RNG4342+rank, dropout disabled.

Lock beta=1.0 before the pilot; no sweep. Loss=original sequence-sum SCST+beta*KL. KL is D_KL(current || frozen XE) on current sampled TRAIN prefixes, summed over valid actions (including EOS), averaged over images. Exclude PAD/BOS logits before log-softmax to avoid subtracting negative infinities. Teacher frozen/detached/outside optimizer; creation must not advance sampling RNG. Kaggle neural preflight checks initial equality, masking, teacher gradients and beta0 gradient parity. Local static/math checks do not establish CUDA correctness. Measure extra teacher forward/memory cost. Dynamic sampled-prefix teacher outputs are not a reusable fixed cache.

## Audit and comparison

First compare saved XE best8 and SCST1500 top1 on identical VALID5000: archive SHA, caption/CHAIR identity and same ground-truth object sets. Report new flags, repairs, categories and reproducibly sampled cases. CHAIR flags are not independent visual confirmation. Do not use TEST cases to tune this loss; prior TEST exposure remains disclosed.

Reuse sealed SCST Dataset Part0, readonly patch links and original hashes. Evaluate XE, saved SCST500 and KL500 on identical cached VALID5000, same beam code/batch shape and frozen0.3 decoder/0.4 CLIP/0.3 object rerank. Report top1/rerank, all8metrics and NLL. Current-evaluator XE is the matched reference; compare saved original XE captions separately for replay diagnostics. No TEST or automatic extension.

Point screening at fixed500: versus XE+rerank require CIDEr+0.2 points or more and no increase in either CHAIR. Versus SCST500+rerank require CIDEr within0.2 points, CHAIRs reduction>=0.2 percentage points and CHAIRi not worse. Report paired image-bootstrap intervals; a point gate on one seed is not statistical confirmation. Failure does not disprove every KL coefficient. Expansion to1500/three seeds requires a new user decision.

Preserve checkpoint/optimizer, download receipts, logs, VALID beams and CLIP scores. Three-hour parent STOP cap includes API/cache checks and scorers, not an ETA. No automatic retry or new quota assumption. Reports unchanged; only completed audit numbers go to Sheets, not hypothetical pilot metrics.

Basis: https://arxiv.org/html/2610.14976v1#S3.SS2 uses reference-policy KL with object consistency under PPO/InstructBLIP. This is a limited adaptation to SCST, not reproduction of its results.

## Preparation and audit completed19 November

[Saved audit](CSA_SEED43_SCST_VALID_REGRESSIONS_RESULT.json): same VALID5000 reference object sets, XE209 flagged captions versus SCST273; 103 new flags,39 repairs,170 shared,4688 clean in both. Fixed30-image sample inspected with actual JPEG hashes. [Assisted notes](CSA_SEED43_SCST_VALID_ASSISTED_REVIEW.json): among20 new flags,7 visually supported errors,3 category/annotation ambiguities,8 uncertain,2 parser mistakes (orange as color; stuffed bear). These are nonblind assistant judgments requiring user verification, not an independently measured hallucination rate. No manual labels used for training. Self-contained review page in Downloads/seed43_scst_valid_audit.html. Sheet Kết quả mới!A247:F254 written/read back; wrap/font metadata checked, native visual fit not inspected.

[Pilot notebook](CSA_SEED43_SCST_KL500_VALID5000_2GPU.ipynb) ready, copied to Downloads. Exact179/185 API only, three input Datasets: csa-scst-full-karpathy-cache-v1, mscoco-2014 including JPEGs, objectdetectionecache. User runs ducanh3/T4x2 with secret codex. Eight offline checks passed in test_scst_kl500.py: all embedded sources parse, exact API routing mock, frozen-teacher guards, checkpoint receipt/binary metadata, VALID-only cached data and three arms, point gate, independent scalar KL derivative, audit transitions/reference mismatch. No Torch/model/autograd/scorer run locally. Actual CUDA tensor preflight, captioner guards, external downloads and runtime/quality remain untested until Kaggle execution. Completion is not a recommendation to expand training yet.

## User-run outcome19 November — pilot complete, locked screen failed

User-confirmed source ducanh3/notebook7bc83c4c76/186. [Audited result](CSA_SEED43_SCST_KL500_VALID5000_RESULT.json), archive SHA92729062eb866f96ec66678e98fff700a9494707473db658f520c8645222b638. KL checkpoint26c0c388…; frozen-reference, RNG, on-policy replay and CUDA tensor guards passed on both ranks. All builds2.11.0+cu128. Train500 log252.97seconds, maximum rank peak allocated1.48GiB; not total notebook time. Original XE captions replay0/5000 changes. Verified all three5000-image cohorts/reference hashes,75000 beams,15000 saved fixed selections, worker/score hashes, CIDEr means, CHAIR counts, NLL and gate arithmetic using cached text. No local model/caption metric scorer. Binary weights/raw arrays/JPEGs absent from review; receipt hashes are evidence of the user run, not independent binary validation.

After fixed rerank, XE / SCST500 / KL500: CIDEr124.3882 /123.6563 /125.0583; CHAIRs3.48 /3.76 /3.66%; CHAIRi2.44489 /2.52278 /2.46832%; NLL2.155262 /2.179242 /2.157232. KL−XE +0.6701 CIDEr but +0.18pp CHAIRs/+0.02343pp CHAIRi. KL−SCST +1.4020 CIDEr, −0.10pp CHAIRs/−0.05446pp CHAIRi. Saved paired-bootstrap95% CIDEr intervals: versusXE[−0.1170,+1.4627], versusSCST[+0.3369,+2.4414]. All CHAIR difference intervals include0. These are conditional on one seed/reused VALID and were not recomputed locally.

KL improves CIDEr versus the matched500-update SCST control and keeps NLL close to XE, consistent with anchoring. It does not establish hallucination reduction relative to XE:174→183 flagged rerank captions (23new/14repairs). Before rerank, KL CHAIRs4.58% versusXE4.18%/SCST4.52%; benefit cannot be claimed for raw decoder hallucination. Three of six locked gates fail (both CHAIR versusXE, ≥0.20pp CHAIRs reduction versusSCST). Do not automatically expand to1500/three seeds or TEST; keep as an exploratory quality-oriented variant, not the chosen joint-improvement model. This does not rule out other KL settings. User decision needed for any new experimental direction. Sheet Kết quả mới!A255:F271 updated/read back; no formal report edited.
