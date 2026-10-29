# Shared pair readout I/J: matched 500-update pilot

Both user-run v2 jobs completed. Source, initialization, frozen state, input identities and fixed A captions match. Exported code matches the pinned Git blobs. No local model execution.

| Model | CIDEr | CHAIRs | Hallucinated captions / 1000 | Object recall |
|---|---:|---:|---:|---:|
| A | 1.205513 | 5.2% | 52 | 45.80% |
| I | 1.035871 | 8.3% | 83 | 46.02% |
| J | 1.134207 | 6.1% | 61 | 45.43% |

## Paired CIDEr differences

| Contrast | Mean | Bootstrap 95% interval |
|---|---:|---|
| J_minus_I | +0.098336 | [+0.066233, +0.130566] |
| J_minus_A | -0.071306 | [-0.108043, -0.036360] |
| I_minus_A | -0.169642 | [-0.211611, -0.130025] |
| J_on_minus_off | -0.034014 | [-0.054391, -0.014389] |
| I_on_minus_off | -0.086126 | [-0.109959, -0.063040] |

Bootstrap: 10,000 paired image resamples, seed 20261028. Intervals are conditional on this run and development set.

J ranks the positive image above its negative on 46/64 whole-tail pairs versus I 44/64; predicate-only ranking is 43/64 for both. These are likelihood tests, not evidence that generated captions express correct relations. The locked semantic panel is not scored yet.

J improves on the matched independent reader, but both degrade the stronger A baseline. Turning pair scores and pair values off also improves CIDEr within each trained model; entity contexts remain active. This does not establish the cause: off changes pair scores and values together, and training also updates two decoder layers.

Recommendation: preserve A as the best model. Do not treat this as a successful caption improvement or automatically extend training. Send the evidence to Astra for the next decision.

## Limits

- One seed; reused development and adaptation-heldout data; bootstrap does not measure seed uncertainty.
- No checkpoint or complete HDF5 payload in review ZIP: full tensor/checkpoint hashes remain runtime reports.
- Off simultaneously zeros pair scores and pair values; not an isolated intervention on interaction appearance.
- Metrics were aggregated from exported outputs, not rerun with neural models or external metric engines.
