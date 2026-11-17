# Matched FP32 and FP16 throughput

Compare fresh 2,000-attempt XE runs at global batch 32 on the same first 64,000 TRAIN caption events. Keep two T4s, mounted H5, grouped sampling, feature sharing, seed and warmup. Use the common FP32 architecture/gradient preflight. In the FP16 worker, Accelerate enables autocast and gradient scaling while keeping model weights and the output used for cross-entropy in FP32.

Report wall-clock caption throughput, peak allocated VRAM, token-weighted train NLL, nonfinite gradient norms, skipped optimizer steps and completed optimizer steps separately. Use fresh Kaggle runs with identical settings. Interpret loss and gradient as stability diagnostics. A faster stable run is a candidate for full-epoch validation, not a quality result.

## Same-session follow-up

After the two separate Kaggle sessions gave inconsistent FP32 timing relative to the historical pilot, run FP32 → FP16 → FP16 → FP32 within one T4 ×2 session and one 60-minute cap. Prepare and run the FP32 preflight once. Reinitialize each timed run in a separate subprocess, using the same mounted cache and 64,000 caption events. Compare both adjacent pairs and pooled throughput by precision. The reversed order helps detect order effects, while shared host page-cache warmth and time-varying load may still affect the measurement.
