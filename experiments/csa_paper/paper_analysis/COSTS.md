# Existing output cost summary

| Arm | Trainable caption params | XE10 worker hours | SCST full1 worker hours | Beam5 max-rank s | Rerank max-rank s |
|---|---:|---:|---:|---:|---:|
| CONTROL | 49,239,288 | 5.914 | 1.615 | 216.305 | 98.952 |
| PROMPT | 53,708,024 | 5.261 | 1.799 | 205.656 | 88.502 |

- Frozen CLIP/YOLO/UPT and rerank encoder parameters excluded from caption trainable count.
- Two T4 workers: max rank seconds is a phase proxy, not measured end-to-end latency.
- Preprocessing, prompt/feature extraction, model loading, barriers and CPU scoring excluded from inference timing.
- Worker times from separate sessions; not a controlled speed benchmark or GPU quota consumption.
- XE workers ran 10 epochs; selected SCST starts C10/P9.

Sources: pinned TEST184/61 text caches; XE178/58 runtime; SCST180/60 audited metadata. No model or official scorer executed.
