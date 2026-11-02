# Local validation — 01/11/2026

- `test_data.py`: **6 tests PASS**, using stdlib/NumPy only. Hand-derived fractional box average and subpatch box; empty vs corrupt records; stable tie ordering; exclusion of val/test from vocabulary; BOS/EOS truncation; POS-list rules; AST parse of all Python modules.
- Notebook generator run against core Git commit `6c365d3`: 5883-byte notebook, code AST parsed, execution counts null/outputs empty. Five source hashes compared with **Git blobs**, not Windows working-copy newline bytes. Notebook SHA verified against `SMOKE_PACKAGE.json`.
- Dataset JSON SHA and split counts computed from actual local file; three local H5 existence/sizes checked. Source metadata report copied with its SHA in `LOCAL_ASSET_INVENTORY.json`. No actual feature arrays read. Attempt to open H5 metadata stopped at missing h5py; no installation or tensor execution followed. Previous metadata is not described as freshly verified.
- Sheet `Experiment Registry`, R223 at row250: PREPARED_USER_RUN_PENDING, with synthetic-only and no quality claim. Readback of cell values/format and following metrics heading; preserved surrounding rows, wrap/font structure. No screenshot/render verification.

**Not run:** Torch import/model/forward/autograd, CUDA, DDP, real text-bank encoding, detector coverage/geometry join, new H5 cache, training/scorer, Kaggle submission or notebook execution. Runtime checks are prepared for user execution, not evidence of runtime success. No claim of novelty, exact CSA reproduction, or improved caption quality.

## 02/11 — full10 XE notebook

- Core `224f270`; `test_full_xe`, `test_full_cache_data`, `test_sharing_data`: **13 tests PASS**, stdlib/NumPy/static only. Actual full566747-event synthetic mapping covers each reference once, changes shuffle per epoch, tail27 splits16/11; ten-epoch counters, three-full-epoch warmup, resume cursor/history/exposure guards, validation-ID coverage and earlier-score ties checked.
- Generated notebook9550bytes, all code cells AST-valid/outputs empty. **17 pinned Git-blob hashes** match; base `model.py` unchanged SHA139edbb1…. **5 mocked control flows PASS**: success and prepare/metric/DDP/train-worker failures. Failure cases archive incomplete status; preflight failures do not launch training. No actual pip, network, staging, Torch or model execution in those mocks.
- Initial mock attempt hit Windows sandbox temp-directory permissions; rerun used a verified workspace child and passed. This was local test infrastructure, not a Kaggle/model failure.
- User-run guards still required: six existing DDP loss/gradient cases, new batched-vs-per-image beam comparison, Java/PTB/CIDEr, full validation/decode and resumable actual CUDA/Adam/RNG path. Local tests do **not** establish their runtime success. Notebook not submitted/run by agent; no full-training quality/timing result.

## 02/11 — full10 XE V2 after beam preflight failure

- User log establishes old attempt stopped at exact beam-ID parity before TRAIN; DDP benchmark completed. No differing IDs/scores available: root cause not established and no equivalence/tolerance claim.
- Removed batched beam path; full validation now calls existing `model.beam_search` directly. Trainable model source unchanged, beam5/max20/raw-score policy unchanged. Structural real-input beam guard replaces the need to validate an unused optimization. Original failed gate is not reported as PASS.
- **14 NumPy/static tests PASS** (`test_full_xe`, `test_full_cache_data`, `test_sharing_data`). Added data-only beam cases for differing EOS stop lengths, unfinished max-length sequences, PAD/BOS content, invalid continuation after EOS, NaN and unsorted scores. No Torch/model execution locally. Actual V2 GPU/validation runtime remains user-run pending.
