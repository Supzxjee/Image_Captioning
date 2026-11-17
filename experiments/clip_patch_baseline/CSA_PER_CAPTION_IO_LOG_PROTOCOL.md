# Per-caption FP32 I/O and logging probe

Compare the old local FP16 patch memmap and per-step loss reduction with mounted H5 plus buffered logging. Keep the old per-caption FP32 model, Adam, LR, global token normalization, clipping, seed and caption order. Use a compact local memmap containing the exact benchmark rows; log its setup separately and do not extrapolate the copy to a full cache. Use paired alternating runs in one session.

Before timing, compare patch/caption events and loss/gradient/parameter digests on eight shared steps followed by the exact epoch tail. During timing, use 256 full batches per run; keep the same batch inputs and check final weights and epoch-loss accounting. The optimized path buffers detached FP64 losses and reduces the chunk at logging time, retaining gradient DDP and token-count synchronization. This is a bounded equivalence check, not a full-epoch statistical result.

## Diagnostic amendment, 18 November

The first GPU run failed exact gradient and parameter hashes from update 1 despite identical input hashes and identical first-step local losses. Final summed NLL differed by 0.0015869 on 29456.22. Hashes cannot quantify tensor error, and an old-versus-old repeat was missing. The cause is unresolved.

V2 diagnoses four short sequences of eight full batches and tail27. Run the old path twice, direct H5 with ordinary logging, then direct H5 with buffered logging. Compare initial weights, CUDA RNG before each forward, inputs and final gradient/weight error magnitudes against the first sequence. Retain exact comparisons and report failures as diagnostic outcomes. Copy only diagnostic patch rows, omit the timing experiment and preserve all diagnostics in the review ZIP. No optimized recipe adoption follows from a small loss difference.
