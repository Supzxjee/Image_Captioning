# Narrow lemma repair — user-run COMPLETE

02/11/2026. Review ZIP SHA `ab9fb2761d6b426f3ae0fe3cb57fd836a0a18c1f9d7f76c2543640278d2701ab`; code/modules match pinned a1ca476 package. Corrected bank SHA `cec76a0229f30361235a4b64e9712ac6bb2e79fc62a9241079a7af3fa95d13ac`.

| Count | Original126 | Corrected |
|---|---:|---:|
| rid | 18575 | 88 |
| ride | 881 | 19368 |
| noun words | 5269 | 5269 |
| verb words | 1594 | 1594 |

Actual redirected tokens: **18487**, comprising16694surface riding and1793surface rides, each POS VERB and original lemma rid. Existing minimum5 rule and ordering retained; omitted mass0. The88remaining rid counts are preserved by this narrow rule, **not certified to all mean “remove”**: original captions also contain misspellings such as ridding skateboards. No broader spelling rewrite or manual cleanup is introduced.

## Evidence independently checked without local POS

- Module hashes matched original Git code. Corrected file SHA matches receipt. Agent rebuilt the complete corrected JSON from original bank + measured18487 delta, exactly matching every exported field.
- All5269noun entries and1592unrelated verb entries unchanged; frequency mass conserved; unique/alphabetic/min5/sorted constraints pass.
- Independently scanned SHA-matched source JSON:566747TRAINcaptionevents and19090captions containing riding/rides. These counts match the user-run report. POS executed only on the19090subset, not allTRAIN.
- Twenty bounded example records match source image/sentence/caption fields and the declared surface/POS/lemma correction. The full18487-token ledger digest is recorded, but no independent POS rerun or full ledger reconstruction was performed locally.
- Same spaCy3.8.11/Thinc8.3.11/NumPy2.1.3/model3.8.0,26model-data hashes verified by the executed guard. Worker34.1517s excludes install/full notebook wall.
- Original bank checksum450fc969… still unchanged. Caption targets, vocabulary and ROI126 were not rewritten; no GPU, CLIP text, caption model, optimizer or Dataset creation in this job.

## Decision

**Important model-effect limit:** both noun and verb word sets are exactly unchanged. rid remains88>=5, ride was already retained; the correction changes frequencies and verb row order only. Current `SemanticAggregation` takes embeddings, not frequency counts: A=softmax(MLP(LWa),word-axis), P=AᵀLWb, which is mathematically invariant to a joint permutation of word rows. Therefore this particular repair supplies no new semantic content to the current bank/module and is not expected by itself to improve CIDEr; no actual CLIP/model parity run has been performed, and floating-point ordering can differ. This is preprocessing/provenance hygiene. Do not add frequency weighting to manufacture an effect or introduce another lexical-audit branch; proceed to the actual model baseline.

**The confirmed riding/rides lemma blocker is resolved.** Use `bank_words_corrected.json` for the next frozen CLIP text encoding. Do not start another whole-bank manual-review gate or rerun ROI. Remaining automated-POS/caption noise and unknown legacy feature provenance are declared adaptation limits, not claims of perfect supervision.

Save the exact original eight Output126 files plus the corrected bank and its correction receipt in a new Private Dataset. Preserve originals and both receipts, explicitly use the corrected file in the future loader. Dataset creation has not yet been confirmed. Once mounted, validate NPZ bytes/contents, encode actual word vectors and run bounded real-data CSA smoke before training/quality evaluation. No CIDEr/CHAIR/novelty or model-improvement result established by this preprocessing fix.
