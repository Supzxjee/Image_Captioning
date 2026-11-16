# Full XE epoch timing

Measure one fresh Karpathy XE epoch including VALID5000, using global batch 32, mounted H5, two T4s and FP32. Keep two workers per training rank, grouped sampling, feature sharing on full batches and ordinary encoding on tail27. Keep the three-hour hard cap and checkpoints. Retain the preflight projection as a diagnostic and allow the worker to start despite that conservative estimate.

Log host batch wait for every step and CUDA-event timing on every twentieth step. The transfer metric covers patch and token tensors. The training interval includes semantic preparation and transfer, forward, backward, distributed synchronization and the optimizer. These intervals are diagnostic timings, not GPU kernel utilization. Log VALID beam decode, teacher-forced NLL and CIDEr scoring separately. Use the measured breakdown to decide whether to test loader workers, precision or validation cadence.
