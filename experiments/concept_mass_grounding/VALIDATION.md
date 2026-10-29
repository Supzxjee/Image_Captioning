# Validation scope

Root has not run any model, local Torch forward/backward, or Kaggle job. Tests below cover data/event semantics and generated control flow, not actual CUDA correctness.

- `test_concept_data.py`:12 metadata/NumPy cases for prefix-free event mass, finite-difference UL sign, vocabulary/special/punctuation rules, compound matching, abstention, detector veto without positive creation, missing-cache abstention, EOS/PAD position mapping, exclusion of surfaces outside the event vocabulary, fixed disjoint cohorts/schedule and unequal-event reduction algebra.
- `test_notebooks.py`:6 checks cover small unexecuted notebooks, identical arm code, pinned Git asset hashes, dev-only label access, and mock success/failure archive flow. Notebook and Python cells compile statically. C/G share source, inputs, policy and code except arm choice. Mock failure/success checks verify errors cannot be archived as completed runs.
- User-run actual-model preflight checks A identity/logit+greedy parity, policy-rescore parity, lexical probability oracle, finite gradients, global two-rank reduction with unequal local event counts, frozen state and no preflight state change. This remains pending until the user runs it.

Source metadata package is generated from the already-verified official COCO TRAIN10k extraction SHAa28e0a35138f14cb4af449a8f53619efca5e41cc23f509d6455e63abd00c43aa, with source member/provenance recorded inside. The development diagnostic's separate package records the pinned official val2014 annotation member. No new images/models were downloaded.

Detector completeness was not verified; the supervision protocol was explicitly revised before a run. Every available valid hint is a veto only; negative evidence comes from annotation/reference absence and remains weak. Unknown provenance is exported. Exact runtime stage coverage, resource use, meaningful negative-event exposure and metrics are not yet known.

Preparation v2 adds6 mocked HDF5 recovery/NumPy tests for source-checked legacy rows, trusted journal reuse, corruption rejection, policy mismatch, output preservation and no neural import. Runtime HDF5 I/O/recovery was not tested locally (h5py is unavailable in this root runtime); no claim of actual user-stage recovery is made.
