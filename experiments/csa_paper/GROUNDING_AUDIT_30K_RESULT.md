# Grounding TRAIN30k label preparation — complete

04/11/2026 UTC+7. ZIP SHA256
`f84920a0eb35a2b7aff351b14fe211f8cbc2b727d654ba4a129914832c47c18a`.
Pinned source24ccd81; 245 metadata/source/hash/count checks PASS. No local model,
Torch, feature arrays, official scorer or annotation download was executed.

| Quantity | Result |
|---|---:|
| Karpathy TRAIN/restval images |30,000|
| Reference caption events |150,076|
| Caption events with mapped mention |119,339 (79.52%)|
| Mapped category mentions |174,465|
| Eligible grounding mentions |158,197 (90.68% mapped)|
| Positive proposal coverage conditional on usable GT |158,197/161,557 =97.92%|
| Observed mapped categories |79|
| CPU worker seconds |339.316 (~5m39s), excludes setup|

Skipped grounding mentions:6,855crowd,6,047missingGT,3,360no-positive,5invalidGT,
1unknown vocabulary token; total16,268. XE training still retains their captions.
Counts are mention occurrences across reference captions, not unique objects or
word–box pairs. Each eligible mention can have multiple positive proposals.

## Consistency

The cohort is unique/TRAIN-only/hash-sorted and its first10,000 entries equal the
original10k cohort exactly. Prefix SHA1faedfb6… and full cohort SHA8bb5e06e… agree
with metadata. Six source hashes match Git;228 reported shard hashes match pinned
v134; per-class sums and coverage denominators agree. Parsed lexicon mapping is
identical to10k. Its file SHA differs because key order differs; semantic mapping
equality, not byte equality, was checked. No thresholds/synonyms changed.

The output records the selected round-robin sampler and fixed2,000updates/64k
events. This is still label preparation, not evidence all30k images were trained
or caption quality improved. Full mentions and filtered GT are absent the review
ZIP, so their actual contents and payload bytes are not independently verified
locally. Labels remain automatic and unreviewed by humans.

## Next

Preserve entire `csa_grounding_data_audit_30k_v2` output, including
`mentions.jsonl` and `train_instances.json`. Pin its actual Save Version before
writing Dataset transport; do not guess version or use latest. After transport,
prepare matched XE/GROUND from ORIGINAL CSA10, same round-robin seed42,
global32/2,000updates/64k events and VALID5000. This exposes every30k image twice
and4k images a third time, not all150,076 caption events for a full reference epoch.
Both pool and sampler differ from the historical10k pilot, so cross-study gains
cannot be attributed solely to dataset size. Current GPU budget must be confirmed
before launching; no training/TEST or new checkpoint is produced by this audit.
