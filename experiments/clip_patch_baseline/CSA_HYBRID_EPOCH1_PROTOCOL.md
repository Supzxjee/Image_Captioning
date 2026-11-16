# Full Karpathy XE throughput pilot

Decision approved after the full-run throughput probe. Run one fresh XE epoch with Accelerate, two T4s, FP32 and the existing seed, warmup, image-grouped caption order and global batch of 32. Share encoder features locally per GPU in the 17,710 complete batches. Both GPUs use ordinary per-caption encoding on the final 27-caption batch. The exact partial batch failed the earlier shared-gradient guard.

Read from the mounted H5 visual cache and stage the metadata and semantic inputs. This deliberately measures whether skipping the full local patch copy pays off across a real epoch. Earlier timing used a small staged subset, so this run has a three-hour stop cap and records whether actual throughput fits. Log preparation, preflight, train, VALID decode and VALID NLL durations. Keep the periodic checkpoint and produce a review ZIP.

Run ordinary-path gradient preflight on first32 and tail27, then time 64 shared training batches on a discarded model before fresh initialization. Evaluate beam top-1 CIDEr and XE NLL on all 5,000 validation images. There is no TEST, SCST, CHAIR or reranking in this pilot. Sharing changes train-time dropout correlations and the wall-clock comparison includes a storage change, so this run cannot isolate sharing speed or establish matched quality across seeds.
