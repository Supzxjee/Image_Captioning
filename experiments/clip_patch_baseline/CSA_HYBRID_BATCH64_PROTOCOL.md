# Matched-exposure batch throughput comparison

Measure global batch 32 and 64 on the first 64,000 Karpathy TRAIN caption events, in fresh runs on two T4s with FP32, mounted H5 and per-GPU feature sharing. This changes optimizer update frequency. Index the existing XE warmup by caption exposure, log every 3,200 captions and checkpoint every 16,000 captions in both arms.

Use two runs of the same notebook with identical inputs. The existing gradient and beam preflight remains a common 32-caption diagnostic. The timed worker checks rank-local batch size and completion counts, measures actual throughput and peak memory, and stops after 2,000 or 1,000 updates. Compare caption events per second and GPU memory. No validation, checkpoint selection or long-run batch-64 adoption is part of this timing test.
