# CSA core smoke — user-run COMPLETE

Reviewed 02/11/2026. ZIP SHA256 `d93d94677a171b7d0911dae74a6ba144c60b0be7ef07dfc08f29a68e75cdb5f8`. Source commit `6c365d3f9f57da3f69cd43244e7b5fd2d51af128`. Four small text members cached in `review_text/`; no neural execution or scorer rerun by agent.

## Verified evidence

Source receipt matches the packaged commit and all five SHA256 values; independently compared against Git blobs. Summary and run status agree: `CORE_SMOKE_COMPLETE`, synthetic only, no quality metrics/real-data verification/checkpoint/DDP.

- Tesla T4, Torch2.11.0+cu128; worker8.3336s, peak allocated1023860736bytes (976.43MiB). This is worker time/memory, not total notebook wall time or billed quota, nor a full-data training estimate.
- All six NumPy/data/static checks PASS in notebook log.
- Tiny tensor suite completed: aggregation normalization, masks/padding, region permutation, causal prefix, batch/single and teacher-forcing/prefix equivalence, beam1/greedy, gradients including both decoder gates, frozen banks, EOS/XE and beam-mean RL contracts. No per-test numeric error maxima saved; do not claim exact arithmetic equality or independent rerun.
- Production shape D512,8heads,FFN2048,K150,3decoderlayers,vocab10000,raw768; **49239288 trainable parameters** with this vocabulary and configuration. Bank sizes in smoke only17nouns/11verbs and small region count; memory is not a worst-case full-training bound.
- Two synthetic XE updates: loss9.441246→9.154867. One synthetic beam-mean RL update: loss0.0006899834. No convergence/quality inference from these values; dropout and two updates on one fake input do not constitute a learning experiment.
- Five beams sorted by cumulative log score, each21IDs including BOS, no EOS, all hit20-token cap. Mostly repeated1032. Expected to be uninformative with random banks/features and almost untrained model; do not present as captions or proof of semantic correctness.

## Decision

**GO for real-data asset validation/preparation**, not full training. Core is no longer blocked on the synthetic tensor test. Next: independently inspect actual HDF5 profile/ID mapping, predicted-box records and dimensions; construct TRAIN-only POS lists/vocab and frozen CLIP text banks; pool real regions; run a small real-data smoke. Keep unknown detector producer thresholds/top-K/coordinate provenance and legacy CLIP revision explicit. Missing record is not an empty detection.

Not established: useful region/semantic alignment, improved CIDEr or hallucination, exact CSA reproduction, novelty, full-bank/150-region memory or throughput, two-GPU parity, training schedule feasibility. Do not rerun this synthetic smoke solely to obtain better-looking generated tokens. Current next step remains the approved CSA baseline, not another novelty brainstorm.
