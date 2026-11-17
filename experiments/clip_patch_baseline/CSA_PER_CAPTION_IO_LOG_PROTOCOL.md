# Per-caption FP32 I/O and logging probe

Compare the old local FP16 patch memmap and per-step loss reduction with mounted H5 plus buffered logging. Keep the old per-caption FP32 model, Adam, LR, global token normalization, clipping, seed and caption order. Use a compact local memmap containing the exact benchmark rows; log its setup separately and do not extrapolate the copy to a full cache. Use paired alternating runs in one session.

Before timing, compare patch/caption events and loss/gradient/parameter digests on eight shared steps followed by the exact epoch tail. During timing, use 256 full batches per run; keep the same batch inputs and check final weights and epoch-loss accounting. The optimized path buffers detached FP64 losses and reduces the chunk at logging time, retaining gradient DDP and token-count synchronization. This is a bounded equivalence check, not a full-epoch statistical result.
