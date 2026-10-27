# Save preparation Output102 as a private dataset

Use `SAVE_GROUNDED_GAIN_V102_DATASET.ipynb`, CPU, Internet ON, Save & Run All. No input datasets, original images or model execution are required.

Source locked to `ducanh2403/2-vi-cai-kia-qua-nhieu/102`, as confirmed by the user. The notebook verifies the installed CLI actually sends `version_label=v102` in a mocked request before downloading. No latest-version fallback.

Enable the secret named by `SECRET_NAME` in Add-ons > Secrets. Default is `KAGGLE_API_TOKEN`; change to `codex` only if that is the actual secret name. A token's descriptive label in Settings is not automatically the notebook secret name. Keep the token out of code/chat/output. OWNER defaults to ducanh2403 and must correspond to the authenticated account with source-output access.

Downloads and hash-verifies exactly five files: original receipt, full `inputs.h5`, `initial_head.bin`, `gain_metadata.json.gz`, and `object_targets.json`. The exact receipt and all four payload hashes are pinned from the audited preparation. Stages a flat verified copy and transport receipt, then creates a NEW PRIVATE dataset. No overwrite/update/delete of an existing dataset. Needs about5GiB free for download and copy.

Default new dataset identifier: `ducanh2403/grounded-gain-prep-v102`, title `Grounded gain v102 feature dataset` (distinct from the notebook title). Existing slug conflicts stop rather than creating a version or falsely reporting success. Creation acceptance is not processing completion: inspect status/file listing and wait before attaching. Confirm all five original names at the root; user-run training preflight checks their bytes again.

After processing: share the private dataset with Ngân, attach it to C/H and T. Expected input root: `/kaggle/input/datasets/ducanh2403/grounded-gain-prep-v102`. No extra `grounded_gain_preparation` suffix when files are at the root.

Implementation reuses the established uploader, with the [official Kaggle CLI source pinned at6894864](https://github.com/Kaggle/kaggle-cli/blob/6894864ff4095cf98e9e36d3a694ff5fc9ba3818/src/kaggle/api/kaggle_api_extended.py). Its version routing, create response handling, license metadata and create-only behavior were inspected; asynchronous server-side processing is not validated locally.

Validation: five synthetic transport/notebook/secret-wrapper tests pass; all notebook code cells parse and are unexecuted; CPU metadata set. No real Kaggle API call, credential access, download/upload, model or GPU run by Codex.
