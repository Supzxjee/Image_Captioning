# CSA word–region audit V1 — incomplete

User ZIP SHA256 `3c1dda40be41c37b4899f6f1b1c057a76f9a1647db46c50e30b2bf16023ee562`.
The 14 fixture tests passed in Kaggle; actual processing reported 5,964/10,000
TRAIN images before `positive_set` raised `ValueError: Corrupt GT`.
No summary, coverage, training, or quality result is available. The fixture's
one-image “COMPLETE” message in the log is not the actual run's status.

11 metadata/source checks passed: 5 Git pins, cohort count/unique IDs/TRAIN splits/
hash order, incomplete status/missing summary and actual log failure. The official
annotation member/archive hashes match previously recorded provenance. Payload
NPZ and filtered GT are not in the review ZIP; no local model/scorer/binary run.

The old guard combined nonfinite GT coordinates and nonpositive GT width/height
into one error. The exact offending annotation is not included in this ZIP, so
we cannot claim its ID or exact values from the log alone. V2 handles finite
degenerate boxes conservatively: skip grounding for the entire same-category
mention, retain its XE target, record invalid annotation IDs/boxes and skip counts.
Nonfinite annotation or corrupt proposal boxes still fail. Cohort/IoU/proposal
identity and payload checks are unchanged. New zero-width, zero-height, negative
width and mixed-valid/invalid fixtures check this branch; 16 tests now pass.

V2 uses a fresh output folder and may reuse the previously downloaded official
archive after its full SHA check. It must rescan ROI boxes: V1 held labels only in
memory and failed before serializing them. No neural extraction/training repeats.
Run the updated notebook's setup, runner and execution cells (or Run All in a new
session); send `csa_grounding_data_audit_10k_v2_review.zip`. Keep old failure output.
