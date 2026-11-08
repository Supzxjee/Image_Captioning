# VALID rerank diagnostic — 08/11/2026

User approved diagnosis of five saved candidates and D/C/O scores on VALID, not changing weights, retraining, or editing any user report. Use end14161 outputs of CONTROL180/PROMPT60 only; exact5000VALID refs, no TEST records. Previous report edits were unwanted: present findings in chat and keep research artifacts separate.

Questions: is a better caption already among five candidates, or does candidate generation fail? Replay fixed0.3decoder+0.4CLIP+0.3object selector, inspect component spreads/contributions, duplicate/repetition/length metadata, and existing per-image CIDEr/CHAIR changes rank0→selected. These are diagnostics, not a new selection policy or visual correctness labels.

All-five candidate CIDEr is not in existing outputs. Prepare CPU-only user notebook on a gzip bundle of already saved VALID text/scores/refs. Official pycocoevalcap1.2/PTB scores on each rank use same5000reference corpus; replay existing rank0 and selected per-image scores before oracle analysis. Reference-based oracle is diagnostic upper bound only, never a deployable selector. No local model/official scorer or Kaggle execution. No automatic new weights, labels, TEST evaluation, or report edit. Manually inspect a stratified small VALID subset with original JPEG SHA matching stored inputs; ambiguous images stay uncertain.
