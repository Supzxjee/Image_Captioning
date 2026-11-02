# Decode131 attempt — incomplete at input-root gate

ZIP SHA `737bdf254e38d1d16cc6343c5f19231f6461714d4ddf58458fc2a3113f2833ce`; exactsource165dd753… and10Gitblob hashes independently verified.15metadataassertions PASS. Download receipt confirms exactOutput131 and checkpointSHA99b1f993…; checkpointbinary absentreview, no independent local rehash.

Worker failed in `load_prepared(PREPARED_ROOT)` with `Expected one corrected assets root`, before bankloading/checkpointmodel loading/greedy/beam. No decode result exists. The old message means marker count was not1; it does not distinguish absentmount/missingmarker/multiplemarkers. Root cause within those choices is unconfirmed. No conclusion about EOS, beam or modelquality is possible from this attempt.

Fix: new decode notebook validates BOTH fullSHA-gated cached inputs before APIsetup/download. Sharedvalidators distinguish missingdirectory and marker counts0/multiple, name requiredDatasets. No hashes relaxed, automatic wrongDataset substitution, architecture/training/decoding change. Need both csa-real-v126-lemma-fixed and csa-semantic-banks-v130; correct PREPARED_ROOT/BANK_ROOT if actualmounts differ. Existing ROI full notebook/source remains pinned to oldcommit; not regenerated for this fix.

37metadata/NumPy/static tests PASS after fix, including missingmount/0markers/2markers fixtures. Agent did not loadTorch/weights or runKaggle. User must attach/correctinputs and rerun decode; updated preflight cannot attachDatasets itself. Inputloop failure did not invalidate trainedOutput131checkpoint.
