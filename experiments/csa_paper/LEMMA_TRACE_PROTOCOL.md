# Exact POS/lemma spot check, no bank mutation

02/11/2026, user approved token→POS→lemma check. Dataset creation has not been confirmed; Output126 is the saved source. Diagnostic does not require a Dataset, API token, images, raw visual cache or ROI re-extraction.

Use exactly spaCy3.8.11, Thinc8.3.11, NumPy2.1.3, en_core_web_sm3.8.0 with the original wheel SHA and installed model-data hashes recorded by Output126. Same excluded parser/NER, active tok2vec/tagger/attribute_ruler/lemmatizer, pipe batch256/process1. User-run CPU only; agent does not load POS/neural models locally. Reuse the proven pipless-venv bootstrap, install in temporary venv; no global package changes.

Packet: at most six original TRAIN captions per surface form riding/rides/ride/rid/ridding/ridden/rode, first encountered in original source order, deduplicated by image/sentence index; include the three visually discussed TRAIN captions if not selected. Add five clearly marked synthetic language controls for contrast, never added to TRAIN/bank counts. Packet source JSON SHA pinned. This is a purposive linguistic spot check, not a dataset error-rate estimate.

Export full token text/index/POS/tag/morph/lemma and a focus table. For VERB-tagged surface riding/rides/ride/ridden/rode, screen expected lemma ride; for rid/ridding, expected rid. Non-VERB focuses get NOT_VERB_REVIEW. Expectations are a morphological screen, not annotation of whether the image supports the sentence. Keep original outputs unchanged. Legitimate rid is never automatically relabeled. No silent correction, bank regeneration, text encoding, caption training or Dataset upload.

Output JSON/HTML/log ZIP, COMPLETE means trace was captured, not that words passed or were fixed. If mismatches appear, inspect their actual contexts and make a separate evidence-based preprocessing amendment. Preserve original version126 and completed ROI tensors. Correct earlier visual-review overreach: the motorcycle image may allow riding for the central person; the concern was the entire group/back-of-bike assertion. None of the images establishes a spaCy lemma mapping.
