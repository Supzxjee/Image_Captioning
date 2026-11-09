# Full10 v1: stopped before training — 09/11/2026

The user-supplied ZIP records complete TRAIN+VALID metadata and both-rank causal/memory/DDP-gradient/beam-path guards PASS. The 64 benchmark updates were discarded; main training updates=0, no VALID score or selected best exists. A printed best.bin path is not evidence that the file exists. Save Version not supplied.

Measured direct-H5 full-caption-shuffle median1.40264s/update, P902.20960s; smoke median~.0564s on256images is not representative. Median extrapolation alone is69.01h for177110updates, excluding validation/setup/scoring. The implemented P90 projection plus35%margin/reserve is148.38h vs10.11h available, arithmetically replayed. These are estimates, not observed full training times. Decode64TRAIN images/rank took6.80/7.12s; GPUpeak.790GiB. No OOM/CUDA failure was observed.

Raw files total34.77GiB, uncompressed float16, one197×768image per chunk. Full sample contains repeated random mounted-H5 access across the corpus, unlike smoke's small reusable set. This supports an IO bottleneck hypothesis, but step timing combines loader, H2D, compute and DDP; it does not isolate storage/CPU/GPU time. Low peak memory alone does not prove low GPU utilization.

The direct-streaming choice did not meet the practical budget. Do not merely increase the cap, discard data/captions or group same-image captions to hide the bottleneck. Recommended next experiment: short CPU-only comparison of identical source rows from mounted H5 and sequentially staged local cache, reporting read/stage time, disk/RAM capacity and exact feature replay. Then use measured capacity/speed to select a bounded local IO path, retaining the architecture, all data and random-caption schedule. No new GPU/model run or changed sampling approved here.

Audit uses cached text once extracted from CRC-checked ZIP; projection equation, coverage, zero-update state, source chunks and both-rank initial digest/guards checked. No local Torch/model/official scorer or checkpoint loading.
