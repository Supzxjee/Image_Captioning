# CSA paper baseline — implementation contract

01/11/2026. User approved implementation after X024/X025. GPU budget reported now: 20 hours on the account to run the notebook. This is not authorization for an unmeasured 15-XE + 30-RL run.

Authority: `../csa_inheritance_audit/CSA_PAPER_IMPLEMENTATION_SPEC.md` and `CSA_PAPER_IMPLEMENTATION_REVIEW.md`. Self-written implementation from the declared equations/conventions; no copied author classes, original-source execution, C0, Q-Former, or per-image legacy prompt. Visual/text encoders are substitutions, not exact benchmark reproduction or novelty.

## Ordered deliverables

1. Inventory actual available assets, with unknowns explicit.
2. Core encoder, dual-stream decoder, beam search and XE/RL loss contracts; local NumPy/static checks only.
3. User-run synthetic tensor smoke on Kaggle: tiny fixtures first, production dimensions next, two XE updates and one beam-mean RL gradient check. Synthetic rewards check gradients only, not CIDEr or model quality. Hard wall cap 30 minutes; one GPU sufficient for this milestone, second GPU not silently required. No external dataset or model download for this core smoke.
4. Build actual TRAIN-only POS/CLIP banks, vocabulary and region assets; real-data smoke, then budgeted pilot. These are subsequent gates; a synthetic smoke does not open training or certify cache coverage.

## Locked core conventions

D512, 8 heads, FFN2048, dropout .2, 150 semantic prototypes, 3 decoder layers. No region positions/geometry/class embeddings. Separate noun/verb aggregation: LWa → two-layer K→K→K MLP; softmax over words. Cosine alignment, region temperature .1, prototype softmax temperature 1. Two object blocks with selected semantics carried to the second; one relation block. Fusion scalar gate weights self stream as G. Decoder uses independent object/relation gates conditioned on raw preceding-word embedding, sums streams without sqrt(2), and sinusoidal text positions. Banks frozen, all caption modules initialized anew.

PAD0/BOS1/EOS2/UNK3. Training maximum sequence 54 including BOS/EOS. Beam5, max20 new tokens including EOS, cumulative unnormalized log probability, completed beams retain their score. RL beam-mean surrogate uses mean over images/beams and divides sequence log probability by fixed20; no greedy SCST/unbiased-MC claim.

## Asset evidence and limits

`LOCAL_ASSET_INVENTORY.json` records dataset JSON hash, split counts and actual shard sizes. Previous user-supplied cache metadata reports full-image224 bilinear preprocessing, raw CLIP last_hidden_state, fp16 storage, unknown model revision. h5py is unavailable in this local runtime: shard internals/profile/ID joins have NOT been independently reopened now, full shard hashes NOT checked. No feature tensors read. Detector file not located locally; producer threshold/top-K and coordinate provenance remain unknown. Real-data preparation must fail on missing/malformed records, and independently verify HDF5 IDs/profile and dimensions before ROI pooling. Never silently equate missing records with empty detections.

## Validation boundary

02/11 asset-inventory stage: CPU-only user notebook reads actual HDF5 metadata/ID/filename tables (not feature arrays), validates full Karpathy and detector coverage, samples RGB headers for dimensions, and builds full TRAIN vocabulary. Known dataset/detector SHA pinned. No GPU, POS/text model, feature extraction or training. Sample geometry is not proof of producer coordinate provenance or all-image dimension validity. Output remains inventory, not training-ready cache. Model revision null is retained as an unknown, not replaced with the text encoder revision. No large HDF5 payload hash implied by checking metadata/ID tables.

Local tests: hand-computed fractional ROI weights, stable sorting, malformed/empty detection distinction, TRAIN-only vocabulary and EOS shift, AST compilation and notebook pin integrity. Runtime script: masks, aggregation normalization, invariance to region padding/permutation, causality, dual-gate gradients, teacher-forcing/step parity, EOS beam handling, beam1/greedy, two XE updates and one RL backward. Runtime script is prepared, never imported or executed by the agent locally. GPU failures must preserve an INCOMPLETE review ZIP.
