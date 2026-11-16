# Mounted H5 training throughput

The one-epoch pilot stopped at update 0 because its preflight p90 projection exceeded the cap. The approved follow-up measures updates 1–2000 from a fresh initialization using the same mounted H5 inputs, image order, full-epoch warmup, two T4s, FP32 and per-GPU encoder sharing. The short-run length does not change the learning-rate schedule.

Keep the real GPU preflight and the hard one-hour notebook cap. Do not apply the full-epoch projection to the bounded run. Record training elapsed time every 100 updates, including routine checkpoint overhead in the intervals after writes. Save checkpoints every 500 updates and verify the final update and caption counts. This run measures throughput on the first 64,000 caption events. It produces no validation scores or estimate of variation across epochs or seeds.
