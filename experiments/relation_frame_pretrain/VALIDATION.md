# Packaging validation — 2026-10-28

- Four metadata/NumPy tests PASS: multiword lexical spans/OOV rejection, fractional tiny-ROI normalization, signed ordering/fixed256pair schedules, paired bootstrap gate/identity rejection.
- Three static/mock notebook tests PASS: pinned cells compile with no outputs and below1MB; worker success/failure archives are explicit and no checkpoint is advertised on mocked failure. Mocks launch no subprocess or model.
- Actual frozen V2 manifest: all320pairs/612images/1280predicted referent bindings parsed and normalized; all lexical spans fit the pinned10000word vocabulary. No source/label/cohort change.
- Two notebooks ~8KB each; exact confirmed v105 mount prefilled. Source assets are committed blob SHA-pinned in NOTEBOOK_PACKAGE.json, including the original vocabulary metadata.
- No local Torch import, forward, autograd, neural test, training or Kaggle job executed. Full HDF5/checkpoint SHA and actual model/autograd/distributed behavior remain user-run checks before training. No result/novelty/caption improvement claim.

Research implementation choices accepted in `ASTRA_FRAME_IMPLEMENTATION_DECISIONS_2026-10-28.md`; that review is design acceptance, not a code/runtime validation.
