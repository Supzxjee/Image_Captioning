# Grounded gain preparation — review passed, full payload check pending

Reviewed2026-10-27. User-run source commit28d24f4, protocolv0.3. No local neural execution and no caption training in this artifact.

- Receipt status `PREPARATION_COMPLETE`; prepare runtime1225.54seconds (20minutes26seconds), below30minute cap. This is preparation time, not total Kaggle session/startup time.
- Log confirms3,000/3,000copied inputs:2,000TRAIN+1,000DEV; no new feature extraction.
- Source-hash manifest exactly matches the delivered notebooks, as do policy, A checkpoint, original prompt, dataset and detector identities.
- Object-target JSON payload hash verified. Exactly2,000unique target rows match the frozen training IDs/filenames; no dev reward rows. `aliases_v2`/`unique_count_v2`; salient labels are supported and detector-neutral labels are disjoint from support.
- Small review ZIP contains only receipt, object targets and log. The HDF5 and initial-head files are NOT independently verified here. Receipt declares their hashes; the complete saved preparation dataset will be verified by the user-run GPU preflight.
- Pip warned about gradio/diffusers dependencies. Neither is used by this preparation; this completed stage does not validate the remaining Torch/GPU checks.

Next: preserve the ENTIRE `/kaggle/working/grounded_gain_preparation` folder as one private dataset; share with Ngân and attach to both notebooks. Do NOT use the small ZIP as training input. Keep `inputs.h5`, `object_targets.json`, `gain_metadata.json.gz`, `initial_head.bin`, `receipt.json`. No need to repeat preparation.

Then C/H uses `MODE='train'` and T already uses train; both use T4x2 and the same preparation dataset. Each arm starts from the common initial head/A. No caption-quality or alignment gain is claimed yet.

Receipt SHA256: `d30c2ad5d4632f9e189c3607701defae62c48b21140f18d9c331febe09b1106e`.
Archive SHA256: `3dedcbfd5ae930f56788ea40e521446dcbca2c7d4fd7476029e7fa461bf29e91`.
Machine-readable results and declared payload hashes: `PREPARATION_RESULT.json`.
