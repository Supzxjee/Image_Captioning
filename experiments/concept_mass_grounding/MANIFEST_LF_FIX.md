# Manifest LF packaging fix — 30/10/2026

User original eval core13daa4f failed the first metadata suite before prepare or inference. Expected31559d... wasSHA of WindowsCRLF working-copy bytes; downloadedGit/LinuxLF blob is8f68ab.... Root reproduced both hashes, parsed JSON equality and unchanged4000IDs/order. This is an assistant packaging bug, not evidence against the method. Original local checks missed Git newline conversion.

Permanent fix: lock `.gitattributes` eol=lf for this manifest only; pin exactGitLF SHA8f68ab9c45ab9c2d11618e837b8d945f20c0c3148c3b1705ff352cdf244a6fab in expanded_data and document byte-only amendment. No ID/model/scientific-policy change. New notebook fetches core6645673.

16localtestsPASS(8metadata/statistics/transport,6notebookstatic/mock including Git-blob-vs-constant regression,2known-failure/repair mocks). Also exported36pinned assets byte-for-byte from Git and ran the8metadata checks there:PASS. This was on Windows with exactGit/LF bytes, not a Linux OS or CUDA test. No neural/HDF5/API/Kaggle execution locally.

If original failed Jupyter kernel is still live: paste FIX_EXPANDED_MANIFEST_LF_CELL.py into a new cell, run it once, then rerun ONLY the final evaluation cell. It verifies exactknownoriginalsource and both equivalent serializations, preserves config backup/failedZIP, changes only the one constant plus declared source hash, and adds explicit technical_amendment. Fresh child processes load the repaired module; no parent import cache reset needed. Refuses recovery if preparation already made OUTPUT, or any source/manifest differs from known failure. Originalsourcecommit13daa4f is retained with exact byte-level overlay receipt for audit. Newly produced reviewZIP ends `_manifest_lf_fix_review.zip`.

If kernel/variables are lost: use updated evaluation notebook in a fresh session. CPU save-weights notebook is unchanged; do not recreate dataset or retrain. Actual cache staging/CUDAparity/newmetrics remain pending user-run. Do not skip assertions or substitute cohorts.
