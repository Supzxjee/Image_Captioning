# Real-data CSA XE pilot — complete, generation still inadequate

Review ZIP SHA256 `fce0e6e1615e3a90b87fe3daddb7a94c454ea7f715bd5cb9fd10c61542a6cfb3`. Exact source `19fed3eea2cd864415bd0503d38bdac4566d1871`. [Audit facts](REAL_XE_PILOT_RESULT.json); original reviewed text in `xe_pilot_review_text/`.

## Actual run

Completed15subsetepochs/600optimizerupdates,19200captionevents/219375validtargettokens.256TRAINimages×5refs=1280events/epoch;64val×5=320events,0test. Fresh CSA; fixedendpoint selection; one TeslaT4 GPU0, FP32/noAMP/noTF32. Same originalROI/vocab and corrected frozenbanks, no recache. Worker331.797seconds (~5.53min), excluding notebook/source setup; peakallocated1013.002MiB, not totalreserved memory. Cached features/tiny subset explain why this is short; this is not a fullCOCO runtime estimate.

| Measure | Before | Epoch15 |
|---|---:|---:|
| Fixed32TRAINimages/all160refs eval-mode tokenNLL |9.224752|3.173585|
|64valimages/all320refs teacher-forced tokenNLL|9.215229|4.556784|
| ValgeneratedEOS |0/64|64/64|
| Single-token repeated output flag |55/64|1/64|

Every recorded valNLL decreases; fixedTRAINprobe also improves. This supports supervised prediction learning. It does not isolate semantic aggregation/alignment contribution or prove image reliance.

## Critical limitation: extremely short output

63/64 finalbeam captions have2words; one has1word. Mean1.984375words vs reference mean10.478125words.29distinctstrings/64images. Most frequent: “a table”7, “a street”6, “a bench”5, “a group”4, “a laptop”4, “a man”4. Single-token repetition flag improvement is confounded by termination/short length, not evidence of highquality.

Examples (reference match is illustrative; no imagejudgement inferred):
- COCO234592 → “a bed”; firstreference describes a canopy bed and estate furniture.
- COCO535808 → “a table”; firstreference describes a bicycle parked before a bookstore.
- COCO200563 → “a snow”; firstreference describes a snowboarder in a snow-covered forest.

NoCIDEr/CHAIR/groundingaccuracy was measured. This model is not yet a useful quality baseline, not a failureproof of CSA, and cannot be compared numerically with A. Raw cumulativebeam score withoutlengthnormalization was the locked convention; earlyEOS/search preference is a plausible explanation to test, not a diagnosed implementationbug. Undertraining/tinydata can also contribute.

## Independent metadata audit

503assertions PASS:7Gitblob hashes; exactpreparedinputs and Output130bankpackage; policySHA; all16epochrecords and40update/epoch schedule;19200events/219375targettokens recomputed from originalTRAINmetadata;60loggedLR/finitegradientnorm snapshots;128initial/finalcaptionrecords aligned with exact64valIDorder, vocabularydecode, EOS/repetition flags and allreference strings; finalcheckpointreceipt agreement. No localTorch/model/scorer/checkpointload or featurebinary reads.

Reportedcheckpoint SHA `99b1f9936b864e549182f2307c369a221e4ca4f17b0f0cad9c60f3fb1cb7017e`; binary absentreview, not independently rehashed. Preserve `/kaggle/working/csa_real_xe_pilot_256/checkpoint.bin`.

`progress.json.status` still saysRUNNING because this version never assigns a finalvalue; this is a reporting omission. Its15completedepochs/600updates match completedresult and endpointreceipt/log. Do not rerun for this cosmeticfield. Generation source feeds only visualfeatures/banks/BOS/predictedprefix, notvalreferenceprefix; refs stored for laterassessment.

## Recommendation, not a new authorized run

Keep endpoint. Before fulltraining or changes to the locked decoder, do a small same-checkpoint greedy-vs-beam check and examine EOSscores after the generatedprefixes. This separates search-length preference from short-output model probabilities without training again. Then choose largerTRAIN ROI preparation/XE or a justifiedimplementationcorrection from evidence. Do not addRL, lengthpenalty or moreepochs silently to make this pilot look good. New run/scope awaits user decision. Pipeline drawing remains at the user-requested planning stage; no figure/deck was authored.
