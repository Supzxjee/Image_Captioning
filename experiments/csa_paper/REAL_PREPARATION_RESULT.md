# Actual words + ROI preparation — CPU COMPLETE, lexical concern open

02/11/2026. User confirmed source notebook version **126** of `ducanh2403/2-vi-cai-kia-qua-nhieu`. Review ZIP SHA `9bd5a8b1e643186fa001a20d2782705898227fac54590b2ca468b8adf53a18b8`; source `b8d186d34f553b02f00b093f8ac1433ea7b2956f`. Fixed pipless bootstrap executed successfully. User-run CPU worker791.19s (~13.19min), excludes installation/full notebook wall time. No GPU/caption model/optimizer or CLIP text encoding.

## Verified against review and original source JSON

- Five source hashes match Git/package; three present payload hashes match receipt (vocabulary, bank words, pilot metadata). `regions.npz` is intentionally not in the review; SHA `054f79bcf13de02255893d5741e695ebfb453f66bc5e488234fba65702904b6c` is reported, **not independently verified from binary yet**. Dataset transport will verify it before upload, GPU reader must validate contents before use.
- POS processed all566747TRAIN caption documents. Independently recomputed the exact length-prefixed original TRAIN text digest `1f265866576cc478a5be562b01d7cae9313a58e44b4511360ba4da5eab1287c8`; matches. No local POS/model rerun. Counts and POS linguistic correctness are not independently reproduced.
- **5269noun +1594verb lemmas**, each unique ASCII alphabetic, count>=5, sorted(-count,word), structurally PASS. spaCy3.8.11/model3.8.0/Thinc8.3.11; official wheel SHA and26installed-model file hashes recorded in source/summary. No forced paper bank size.
- Exact deterministic256TRAIN+64val IDs/order/source splits/captions match original dataset; 0test/320unique. All saved54token sequences independently reproduced with frozen vocabulary. JPEG/raw-feature-row hashes have valid format; their source bytes not independently available locally.
- **2180regions**, max35/image,5full-frame fallbacks,0boundary clips. Sums/kinds/source counts match metadata and summary. All320 image dimensions positive. No independent NPZ numerical check or raw feature repooling performed by agent. Legacy visual revision/detector provenance remain unknown.

## Important lexical screen — not semantically cleared

Top verb lemmas include `rid`18575 versus `ride`881. Original TRAIN raw surface counts: `riding`17287, `rides`1872, `ride`1202, `ridding`88, literal `rid`1 (regex screen, not POS tags). See `POS_RID_LEXICAL_SCREEN.json`. This discrepancy is a **suspected lemmatization problem**, not a confirmed token-level `riding→rid` trace: exact token/POS/tag/lemma outputs were not exported. Do not silently merge/rename the word or claim all counts verified. No local spaCy execution was used to guess the cause.

**Preserve version126 unchanged**. Before CLIP text encoding, inspect a small real CPU POS trace on TRAIN examples with the exact model/settings, then apply any justified deterministic lexical correction under an explicit amendment and new bank checksum. Do not repeat raw feature extraction or delete the completed ROI cache. Do not force this semantic concern into a model-quality claim; no captioner has trained yet.

## Next

Save the exact eight preparation/provenance payloads from Output126 as a Private Dataset after all file SHA checks. No latest fallback or overwritten existing Dataset. Source word lists are archived as produced, including the flagged item; preservation is not semantic approval. Then resolve lexical screen and create actual frozen CLIP text embeddings + real-data model smoke. Not full training or a novelty result.
