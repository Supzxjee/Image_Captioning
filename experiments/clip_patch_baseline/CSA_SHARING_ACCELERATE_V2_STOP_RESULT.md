# Accelerate sharing V2 — stopped before timing

10/11/2026. Pinned source7c70747/29assets and archivef062df32 checked; no local model or scorer. Save Version awaiting user. Accelerate1.15.0 initialized on2T4/Torch2.11cu130/FP32/accum1, correctwheel andmanualsharding.

First case grouped32, **ordinary shared=false**,32encoderrows/32captions,379targettokens. Full-global NLL9.16591740 vsDDP9.16591836 (delta9.59e-7). Cumulative maxrelativegradienterror0.09196%; maxabs7.6472e-5 vsfixed5e-5 cap. Rejected tensor core.decoder.2.ffn.net.0.weight. Raw gradient scale and coordinates were not saved. Finite-precision/batch-shape differences plausible, not established. No optimizerupdates, speedratio, peak-memory or quality result; do not call sharing or Accelerate slower/faster or faulty on this basis.

V3 corrects the reference to ordinary16/16(or16/11) microbatch-summed gradients with globaltoken denominator, retaining full-global scalarNLL. This reduces kernel-shape sensitivity without changing the mathematical objective. Saves per-parameter norm/scale/maxerror/coordinates/values. RelativeL2cap.005 is stricter than previous.01; maxerror cap3e-7+.005*referencepeak (nearzero retains3e-7). These are disclosed engineering tolerances, not proof; rejects missing/localized or factor2 errors in numericalfixtures. User-run6GPUcases still required before timing.

No model/precision/loss/scaler/optimizer/sampler/benchmarkstep/cap change. No local model/Accelerate/scorer/Kaggle runs. NewV3notebook/output and freshsmall staging; previous/tmp availability unknown. This review does not automatically relabel or rerun V2.
