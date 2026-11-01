# Actual semantic word lists + bounded ROI preparation

02/11/2026, following approved X024 and COMPLETE inventory. This is CPU preparation, not a new architecture or a training experiment. User runs notebook; no POS/neural/model execution locally by agent.

## Frozen choices before code

- Exact original dataset/detector SHA from `ASSET_INVENTORY_RESULT`. Vocabulary must reproduce SHA `924be210aa0daa8b0968a6d69ad2c66b3de603f49c35f4aa67247c75d6173fc7`, 10000 tokens. TRAIN=train+restval,113287 images/566747 caption events. No val/test text enters POS counts or bank words.
- POS library spaCy3.8.11, `en_core_web_sm`3.8.0; exact official model wheel SHA `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`. spaCy has a CPython3.13 Linux x86_64 wheel on PyPI. Notebook installs in a temporary venv with system site packages and pins existing NumPy version; no global transformers/huggingface changes. Installed component versions and model-file hashes recorded. Transitive dependency lock is recorded at runtime, not represented as fully pre-pinned.
- Process original TRAIN sentence `raw` text, in source order, with parser/NER excluded; keep tok2vec/tagger/attribute_ruler/lemmatizer. Single process/batch256, no GPU. NOUN/PROPN → noun; VERB → verb, excluding AUX. NFKC/lowercase ASCII alphabetic lemma, minimum5 occurrences, sort(-count,lemma). Bank sizes are measured outputs, never forced to paper numbers. CPU POS is a pretrained neural tagging model, not rule-only parsing; it is not the caption model.
- Pilot split256TRAIN/64val. Within each group sort by SHA256 of `csa-paper-pilot-v1/42/` + filename and take the first N. TRAIN and val image-disjoint. No test images staged or evaluated. Reused development val is not an independent test.
- Actual region inputs: independently read selected raw CLIP rows, verify shape/finite and H5 complete/profile/COCO-ID/filename mapping, remove CLS→14×14patches, read source RGB header/dimensions and hash JPEG bytes. Pixel-xyxy **interpretation remains an operational assumption** because detector producer provenance is not recovered; record all clipped boxes. A numeric audit does not prove this interpretation. Fractional-area pooling/raw768/fp32 accumulation; store fp16 ROI values, no learned projection cached. Stable confidence sort/cap150. Empty stored list → exactly one full-frame mean token. Missing/malformed/outside-zero-area record fails rather than silently dropping an image.
- Cache only320 images at this stage. Save concatenated features/boxes/scores, offsets and IDs in NPZ; metadata contains all selected captions and their encoded54-token sequences. TRAIN captions are training targets; val captions remain evaluation metadata, not bank input. No class-label embeddings/GT boxes/relations.
- Output status CPU_PREPARATION_COMPLETE is **not** semantic-bank embedding completion or training readiness. Noun/verb lists still require pinned frozen CLIP text encoding/L2 normalization512D on the subsequent GPU notebook; actual caption forward/gradient and scorer remain unrun.
- CPU worker wall limit2hours, installation15minutes separately; limits are stopping caps, not runtime predictions. No training, optimizer or caption checkpoint. Preserve entire output folder as Dataset; only small JSON/log/wordlists in review ZIP, NPZ stays in output with SHA receipt. No Kaggle API token or automated submission.

## Sources verified for dependency choices

- [spaCy3.8.11 PyPI metadata and wheels](https://pypi.org/project/spacy/3.8.11/) (Python metadata>=3.9,<3.15; CPython3.13 Linux wheel exists; old description paragraph has stale Python ceiling).
- [Official en_core_web_sm3.8.0 release](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0) (wheel checksum above, compatible spaCy>=3.8,<3.9, MIT).
- [spaCy Language.pipe](https://spacy.io/api/language#pipe) for streaming batches/components. Existing CSA spec supplies POS/bank templates and CLIP revision; no source code or binary banks from CSA authors reused.

## Checks

Local metadata/NumPy tests: split and deterministic selection, TRAIN-only fake POS stream with held-only sentinel, filtering/count/sorting, encoded EOS, variable-region packing/fallback/box sorting, malformed data failures, source/AST/notebook pinning. No spaCy tagging, HDF5 feature reads, Torch/CLIP/caption model locally. User-run preparation does real POS, headers and ROI pooling; its review ZIP/receipt must be checked before claiming assets built. Full H5 byte hashes are not implied by selected-row verification.
