# CPU IO probe result — 09/11/2026

User ZIP complete, source a3a927b; Save Version pending. No models or TEST feature reads. Four identical event streams (2048TRAINcaptionevents/2029uniqueimages) have matching outputSHA across mounted1, mounted2, local1, local2. Selected localFP16 rows individually replayed by runtime; author verifies stored reports/hashes/metadata/arithmetic, not the actual feature payloads. See IO_RESULT.json for check count.

| CPU delivery per global32events | seconds/batch |
|---|---:|
| MountedH5 pass1 |0.936379|
| MountedH5 pass2, same sample |0.046021|
| Localmemmap pass1 |0.039298|
| Localmemmap pass2 |0.039004|

First mounted to repeated mounted changes20.35×; warm mounted vs warm local is1.18×. First mounted vs local is24.01× but **not a demonstrated full training speedup or isolated H5-vs-NPY format effect**. OS cache state is uncontrolled; identical mounted sample was reread, local sample was freshly written. Timers exclude process startup but include stream hashing in task-wall; read+stack timers are also in worker reports. Four-reader timing has no CUDA/H2D/DDP.

Sample582.54MiB staged in54.07s; total probe143.60s. Local /tmp free966.78GiB, RAMtotal31.35GiB, available29.61GiB at measurement. Exact fullTRAIN+VALID patch payload33.17GiB exceeds that RAM but fits measured local disk; GPU runtime capacity must be rechecked, and cgroup limits may differ from /proc/meminfo. Do not preload all features in RAM or multiply small-sample cached throughput into a full10epoch ETA.

Evidence strongly supports mounted first-access/data availability as a contributor to previous full-step latency, without proving all slowdown is IO. Recommended loader repair: sequentially stage full TRAIN+VALID196patchfloat16 into /tmp, preserving exact feature bytes and allcaption shuffle; then actual user-run full-data preflight (loader+H2D+compute/DDP+beam) again. Keep original model/seed10epochrecipe/cap, no TESTfeature reads. Stage capacity and cost must be measured; full local corpus exceeds RAM, so the small sample cannot settle its eventual random-read rate. No new fullGPU notebook or run is claimed here.
