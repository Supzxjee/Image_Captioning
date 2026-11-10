# CSA-structure patch + prompt smoke — reviewed complete

10/11/2026. User confirms Save Version15; notebook `kltnnnnnff` inherited from the active experiment context. Source0323747,22pinned assets. 52cached-text/source/log/count/arithmetic checks passed. ZIP read once into ../.review_cache/csa_structure_smoke_v1/archive_text.json; no local Torch/model/scorer/binary load or Kaggle API.

| Item | Observed |
|---|---:|
| TRAIN images / source captions |256 /1280|
| Main updates / global caption events |200 /6400|
| Fixed TRAIN probe NLL before → after |9.239455 →4.358260|
| Last25 online train-loss mean |4.702218|
| Trainable parameters |53,183,736|
| Steady global update median /p90 |0.244246 /0.249521s|
| Global caption events/s |131.015|
| Max preflight allocated memory |2.643GiB/GPU|
| Max train/final-probe allocated memory |1.991GiB/GPU|
| Raw selected patch staging |78.91s|
| Worker invocation incl. guards/training/sample beam |60.57s|
| Staged prompt slots |779object /109relation|

## What passed

Both TeslaT4 ranks on Torch2.11.0+cu130 completed. Two complete object fusion layers, one relation layer and a3layer parallel decoder were present. Object1 selected semantics were carried into object2. Packaged dual memory was split before separate decoder cross-attentions. Perturbing either stream separately changed initialized logits (max0.13135object/0.12782relation). Causal future invariance, token-normalized DDP gradient parity, masked prompt locality, invalid-slot invariance and beam5 saved-path score replay passed.

Nonzero real-probe gradients reached object1/2/relation, both prompt adapters and both bank aggregators. Mask/text sensitivity guard uses a controlled nonzero-output copy, so it is not evidence that the trained system already benefits from text. TRAIN probe NLL decreased; reported final visual perturbation response was0.19222mean logit difference. These checks establish operational wiring and a learning signal, not accurate grounding.

The raw selected-row fingerprints, cache profile and vocabulary match smoke192. Prompt175 and bank130 provenance seals match fixed source packages. Review ZIP contains receipts rather than raw NPY/H5 payloads, so those binary hashes were verified by user-run preparation and were not rehashed locally.

## Critical warning: generated samples

All8first-rank TRAIN examples decode only **"a"**; the other rank repeats the same probe images. Do not present this as successful caption quality or convergence. A model can reduce teacher-forced XE while its autoregressive beam output remains too short. The run is only200updates, so this does not by itself establish a bug or irreversible collapse. No additional training or quality evaluation is claimed.

Beam8 timing (~0.32–0.34s) is dominated by very early termination and is unsuitable for predicting full VALID decoding once captions become longer. Do not change EOS/beam/length rules merely to conceal these outputs.

## Scheduling implication and recommendation

Multiplying measured tiny-pool median by17,711updates gives **~1.20h of training per full epoch**, or~12.02h for10epochs. These are arithmetic projections, not ETAs: they exclude full staging, validation, metric scoring, checkpointing, changing sequence lengths and large-corpus IO. Do not promise a10epoch job fits one session. Last reported own6h/Ngân30h is before/around this run, not a fresh quota measurement.

Recommend starting fresh full-TRAIN XE with a checkpoint and VALID quality inspection after the first full epoch, using the30h account for subsequent longer work. Check whether captions become meaningful and VALID CIDEr improves before committing to a long uninterrupted run; preserve optimizer/RNG for explicit continuation. Exact training horizon/LR/phase caps still need scheduling before implementation; do not silently turn a proposed1epoch checkpoint into a changed cosine schedule. Do not infer improved CIDEr from this smoke or run TEST. No new run was launched or approved by this upload.

This remains CSA structural inheritance with CLIP-patch/text substitutions and additional UPT per-image prompt adapters, not exact detector/RegionCLIP replication. Historical patch and ROI experiments remain unchanged.
