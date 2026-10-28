# Packaging checks — 28/10/2026

No local Torch/model/autograd/inference/training or Kaggle job has been run. CPU metadata/NumPy and mocked notebook checks only. The source includes real user-run model preflight; its outcome is pending, not inferred from static tests.

Local evidence:

- All320frozen assertion targets map to the original A vocabulary; predicate and full predicate+object-tail masks normalize correctly.
- Fixed1000devimage dimensions extracted from officialCOCOval2014member with SHAe1b4d1bc…; packaged JSON has canonicalLF byteSHA534c0501… verified in tests. This is image metadata, not GT-region substitution.
- All1000devpredicted-box banks from the original SHA498825da… detector cache pass normalization/geometry checks; seven have zero predictions and are retained with null-only readout.
- TRAIN484images, held128images, dev1000images have no adaptation-image intersection. The32pair semantic panel is selected16/stratum using fixed hashes before any new captions.
- Five CPU tests pass: independent probability oracle/null/padding; interaction-to-marginal effect; sorted/clipped boxes and intervening union; frozen targets/cohort/schedules; static arm-switch and sampling-policy contract.
- Five notebook packaging tests pass: both pinned notebooks, empty outputs/static compilation, the single arm-configuration difference and mocked bootstrap/success/worker/evaluator failure paths. No real subprocess is launched by the mocks; HDF5/checkpoints are excluded from reviewZIPs. Together with the five data tests, ten local checks pass.

User-run checks still required: actual input mounts/fullSHA and source/HDF5 schema, Java/PTB/METEOR/CHAIR dependencies, A tensor identity, zero residual logit/greedy parity, real-reader outputs versus FP64NumPy oracle, null/self/padding behavior, dropout-free policy sampling/rescore parity, first zero-projection and subsequent branch gradients, frozen scope, serial/global-versus-two-rank gradients, synchronized update, exact restoration, actual throughput/budget. No cap is a runtime promise.

The user reports25remaining GPU hours; no assumption of50hours or exact per-account split. Each GPU phase has a one-hour two-GPU cap, with CPU setup/staging/evaluation adding notebook-accounting time. Scope is an inherited readout improvement; no novelty, caption gain or journal-acceptance result is established by this package.
