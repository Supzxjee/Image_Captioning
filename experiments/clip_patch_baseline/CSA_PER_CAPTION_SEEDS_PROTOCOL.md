# Additional seeds for historical per-caption XE10

User approved seeds 43 and 44 after the I/O diagnostic. Use the original 5eb0122 per-caption FP32 runner and model, global batch32, grouped sampling, Adam and ten-epoch warmup/cosine schedule. Change initialization seed, grouped epoch permutation seed and rank-local dropout RNG consistently. Seed42 replay must keep the original 42+epoch and 4242+rank formulas.

Use mounted H5 for patch reads, retaining original per-step loss reduction, clipping, gradient synchronization, per-500-update checkpointing and VALID5000 beam/NLL at every epoch. Select maximum VALID CIDEr with earlier ties as in the historical result. Keep best and last weights. The approved change saves full patch-copy setup; short diagnostics establish matching input bytes but not exact long-run numerical replay.

Cap each invocation at 11.5 hours including setup and scoring. Log throughput projections as estimates. The hard deadline saves model, optimizer, per-rank Python/NumPy/Torch/CUDA RNG and the exact epoch cursor. Resume only from this seed's full output folder, with the identical pinned source, data and software. Full10 may need more than one Kaggle version. Pause at the hard deadline and resume explicitly.

Run each notebook from fresh initialization first. Preserve the entire output folder when interrupted, attach it as an input and set RESUME_ROOT to its folder for the next invocation. Use a new OUTPUT folder per invocation. A completed result requires 177,110 updates, 5,667,470 caption events and ten VALID history entries. TEST decoding and the final mean/standard-deviation table follow after both seeds complete.

## Author verification

The generator embeds the four amended original files and downloads the remaining source at its recorded revision and SHA256. All notebook and embedded Python compiles. NumPy checks replay seed42 exactly at epochs1/10, confirm distinct full exposure for seeds43/44 and reconstruct resumed sampler suffixes. Cursor/caption accounting covers mid-epoch, pre-validation and completed states. Static inspection confirms optimizer/RNG restoration, strict resume identity, original per-step loss reduction and byte-identical forward/backward/update block. Model execution, CUDA preflight on the new seeds, and binary checkpoint loading remain user-run verification.
