# Local validation — 01/11/2026

- `test_data.py`: **6 tests PASS**, using stdlib/NumPy only. Hand-derived fractional box average and subpatch box; empty vs corrupt records; stable tie ordering; exclusion of val/test from vocabulary; BOS/EOS truncation; POS-list rules; AST parse of all Python modules.
- Notebook generator run against core Git commit `6c365d3`: 5883-byte notebook, code AST parsed, execution counts null/outputs empty. Five source hashes compared with **Git blobs**, not Windows working-copy newline bytes. Notebook SHA verified against `SMOKE_PACKAGE.json`.
- Dataset JSON SHA and split counts computed from actual local file; three local H5 existence/sizes checked. Source metadata report copied with its SHA in `LOCAL_ASSET_INVENTORY.json`. No actual feature arrays read. Attempt to open H5 metadata stopped at missing h5py; no installation or tensor execution followed. Previous metadata is not described as freshly verified.
- Sheet `Experiment Registry`, R223 at row250: PREPARED_USER_RUN_PENDING, with synthetic-only and no quality claim. Readback of cell values/format and following metrics heading; preserved surrounding rows, wrap/font structure. No screenshot/render verification.

**Not run:** Torch import/model/forward/autograd, CUDA, DDP, real text-bank encoding, detector coverage/geometry join, new H5 cache, training/scorer, Kaggle submission or notebook execution. Runtime checks are prepared for user execution, not evidence of runtime success. No claim of novelty, exact CSA reproduction, or improved caption quality.
