# Accelerate + feature sharing V3 — benchmark complete

11/11/2026. Sourcefa9abcc/31pins; archivea1eaa846.103cached-source/log/count/arithmetic checks and1080saved per-parameter diagnostics replayed. No local neural/scorer/Kaggle execution. Save Version awaiting user confirmation (kltnnnnnff per session).

## Matched results

All configurations use Accelerate1.15.0, FP32, twoT4, global32captions, same restored CSA/prompt parameters and TRAIN256cache. Each sampler runs ordinary/shared/shared/ordinary; eachpass40updates, first10timingwarmup excluded. Medians pool60timed samples per mode per sampler. All1280captionevents consumed every pass;320optimizer updates total across8fresh-model passes, not one trained checkpoint.

| Sampler | Encoder mode | Median s/update | p90 s/update | Mean encoded images /32captions | Peak allocated GiB/GPU |
|---|---|---:|---:|---:|---:|
| Random caption | Ordinary |0.246867|0.250930|32.0|2.1597|
| Random caption | Shared |0.245030|0.248574|31.1|2.1125|
| Image grouped | Ordinary |0.259026|0.262220|32.0|1.9928|
| Image grouped | Shared |0.143539|0.145456|8.0|1.5483|

With matched GROUPED order, sharing gives **1.8046×throughput**, **44.58%lower update time**, **22.31%lower peak allocated memory**, and75%fewer encoder image rows. Decoder still handles all32captions. Shared grouped passes have medians0.14337/0.14415s, ordinary0.25181/0.26101s; substantial effect despite pass-to-pass timing drift. With RANDOM order, only32→31.1encoderrows, speed1.0075× (~0.74%less time); this is too small/short a result to claim a robust practical speedup.

Groupedshared vsrandomordinary is1.7199×, but also changes sampler. **These runs do not isolate Accelerate versus native DDP**: all four configurations use Accelerate. Do not attribute1.80×to switching library alone.

## Numerical checks

All6cases pass: grouped32/tail27/random32 ×ordinary/shared. Unequal tail preserves16/11rank captions and314global target tokens. Ordinary/grouped firstcase32captions and379tokens encodes32rows; shared encodes8. Tail encodes27→7rows. Per-step caption-image plans reconstructed with NumPy, and their caption_image.npySHA exactly matches the runtime stage. Target-token counts agree between ordinary/shared/repeatedpasses for each sampler.

Maxrelative gradient error across tensors/cases4.6375e-5 (=0.00464%); maxabsolute3.7495e-6, below even V2's old5e-5 absolute threshold. Ordinary/grouped now maxabs2.2352e-8 vs V2's7.6472e-5 after switching to rank-shaped reference. This supports batch-shape/reference arithmetic as the V2 issue under the revised computation; it is not a claim of bitwise equivalence with arbitrary fullbatch kernels. Fullglobal scalarNLL and microbatch-summed scalarNLL remain consistent. Saved180parameter diagnostics percase allreplay under the lockedV3criteria.

Runtime logs confirm Accel1.15.0/Torch2.11cu130/NCCL/2GPU/noAMP/accum1/manualcaptionranksharding andno preparedDataLoader. All31source pins match Git; rawpatch/profile/tokenizedcaption/semanicpayload seals match originalsmoke15. Rawbinary gradients/features absent from review; replay uses saved scalars and runtime provenance, no local binary validation.

## Decision and limits

Feature sharing is operationally ready to include in a **future image-grouped full-XE run**. It is not useful enough with this random-caption sampler to justify a speed claim. Keep normal decoder work and sum gradients across captions, with no detach. No full-training run, newmodel, newprecision, TEST or checkpoint creation was launched by this review.

Sharing changes encoder-dropout correlations, and grouping changes SGD batch composition. Eval/dropoutOFF numerical agreement does not prove identical training trajectories, preserved CIDEr/CHAIR, or no generalization loss. Check VALID quality after the first full TRAIN epoch before committing to a long run; fresh initialization, horizon/LR and session budget still need specification.

Multiplying0.143539s by17711updates gives~42.37min training-only per epoch and~7.06h for10. This is a small-pool arithmetic projection, NOT an end-to-end ETA, guaranteed session fit or billedGPU estimate. Fullstaging/IO/VALID/metrics/checkpoints/variablecaptionlengths excluded; full-data throughput must be measured. There is no CIDEr, CHAIR or convergence result in this benchmark.
