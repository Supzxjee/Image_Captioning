# CSA word–region data audit V2 — complete, human review pending

04/11/2026 UTC+7. User ZIP SHA256
`9ba6ee8e1930c5b4bdd39924c86182bdcf2c834ac2448ffe76af9a9a05c6c449`.
Pinned source `691c03d`; actual CPU worker completed in 289.073 s (~4m49s),
excluding notebook setup/install. No caption model, GPU, optimizer or scorer ran.

| Quantity | Result |
|---|---:|
| Karpathy TRAIN/restval images | 10,000 |
| Reference caption events | 50,022 |
| Caption events with a lexicon-mapped mention | 39,843 (79.65%) |
| Mapped category mentions | 58,575 |
| Eligible mention/proposal-set labels | 52,956 (90.41% of mapped mentions) |
| Valid-GT mapped mentions with a positive proposal | 52,956 / 54,088 = 97.91% |
| Observed mapped categories | 78 |
| Eligible person mentions | 17,223 (32.52% of eligible mentions) |

Mentions are repeated word occurrences across references, not unique images,
instances or word–box pairs. A mention can have multiple positive proposals.
97.91% is conditional proposal coverage, not precision, grounding accuracy or
coverage of all nouns. The operational 80% proposal-coverage threshold passes.

| Grounding skip reason | Mention count |
|---|---:|
| Crowd annotation in category | 2,412 |
| No GT for mapped category | 2,069 |
| Valid GT but no proposal IoU >= .5 | 1,132 |
| Unusable finite GT geometry | 5 |
| Unknown vocabulary token | 1 |
| Total | 5,619 |

All 5 unusable-GT mentions are in the person category. Counts refer to mentions;
the review ZIP does not include full labels/GT needed to identify how many unique
invalid annotations/images caused them. V2 now completes beyond V1's failed guard.

## Verification and remaining limits

847 stored-metadata/source/count/sample-geometry assertions pass. Five source
hashes match Git; 228 reported ROI JSON/NPZ hashes match pinned v134 receipts;
TRAIN cohort IDs/splits/order, lexicon/cohort hashes, aggregate class counts and
coverage denominators agree. Given saved GT boxes, positive proposal sets replay
on 196 eligible review occurrences (96 general +100 eligible sample occurrences).
This is not independent inspection of the whole actual ROI payload, region kinds,
annotation contents, source captions or semantic correctness. Those payloads are
not in the ZIP. No model or metric replay was performed locally.

The random general review sample contains 96 eligible /2 crowd /1 missing-GT /
1 no-positive. The separate eligible sample has 100 eligible cases. Neither page
contains human judgments. In general sample #26 the caption “An orange-and-black
cat wearing a tiny black hat” lexically maps “orange” to fruit, but is skipped
because GT is missing. This is a concrete warning about noun/adjective ambiguity,
not proof that retained labels are clean or that every missing-GT mention is an
annotation error. A color mention could also coexist with a real fruit elsewhere.

Next inspect the fixed eligible 100 for (1) intended object meaning of the word,
(2) category-matching GT, and (3) reasonable positive proposals. Do not judge the
whole reference caption as a model output. Review cases with multiple same-class
instances at category level. A missing photo is unreviewable, not confirmed.
Keep the proposed >=95/100 manual check and report uncertain/error cases; do not
lower it based on these counts. General sample is diagnostic, not a second 100
mandatory confirmations or an estimate of eligible-set precision.

Preserve the entire Kaggle output folder `csa_grounding_data_audit_10k_v2`,
especially `mentions.jsonl` and `train10k_instances.json` excluded from the small
review ZIP. No further CPU audit rerun is needed. Label review and current budget
precede matched XE vs XE+.1 grounding pilot preparation. No quality gain, new loss,
exact instance grounding, or automatic training GO is established.
