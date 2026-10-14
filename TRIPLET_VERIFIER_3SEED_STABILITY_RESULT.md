# Triplet verifier: paired three-seed development stability

Candidate-specific predicate/object text retains its development AP and true-relation retention advantage across the three tested seeds. It does **not** uniformly improve precision and does not resolve actor binding. Keep this branch for further research, without promoting a best seed or full prompt cache.

## Fixed setup

Seeds42,7,123; same1,776 training pairs/826images,489dev pairs, model dimensions,30epochs, lr0.001, losses and training-only threshold-fit rule as R69/R70. Each seed compares candidate-specific text with the shared train-only relation bank. No inference threshold is tuned from dev or the100-image audit. Seed42 heads are reused by locked SHA; only four new trainings and transfer evaluations run locally. No Kaggle/GPU extraction is performed.

Protocol: `audit/triplet_verifier_seed_stability_protocol.json`. All three seeds are reported. Original seed42 checkpoints remain fixed; no best-seed selection, epoch selection, ensemble or automatic checkpoint promotion.

The489-pair dev set,364-pair detector-matched transfer cohort and100-image candidate audit are all previously inspected development data. Additional seeds do not create an untouched heldout test or independent annotations.

## Every paired run

TP/FP below refer only to accepted known-labelled claims in the same reused class/IoU-matched364pair/286image transfer pool. Unknown links are excluded from precision and are reported in JSON.

| Seed | Dev AP: relation bank → candidate text | Transfer TP/FP: relation bank → candidate text | Candidate-text full100 claims |
| --- | --- | --- | ---: |
| 42 | 0.679835 → 0.760567 | 28/2 → 35/1 | 11 |
| 7 | 0.642327 → 0.753136 | 22/1 → 34/2 | 10 |
| 123 | 0.671379 → 0.732905 | 30/2 → 36/3 | 14 |

Candidate-specific text raises dev AP in all three matched runs, by0.080731,0.110810 and0.061526. It retains7,12 and6 more known true claims than the matched relation-bank control. It removes one more known false claim at seed42, but admits one more false claim at seeds7 and123. Thus the precision benefit seen at42 is not uniform.

## Mean and variation across seeds

| Measure | Shared relation bank | Candidate-specific text |
| --- | ---: | ---: |
| Dev AP, mean ± sample SD | 0.664514 ±0.019674 | 0.748869 ±0.014316 |
| Known-label transfer precision, mean ± sample SD | 94.25% ±1.24pp | 94.66% ±2.46pp |
| Conditional transfer recall, mean ± sample SD | 10.97% ±1.71pp | 14.40% ±0.41pp |
| Known positives retained, range | 22–30 | 34–36 |
| Known negatives retained, range | 1–2 | 1–3 |

The SD is across only three random initializations. It is **not** an image-level confidence interval, statistical significance test or evidence of broad robustness. These arithmetic means are not pooled/micro precision.

The original R50 proposer retains43known true and6known false claims, with conditional recall17.70%. Candidate-text seeds still lose8,9 and7 true claims relative to that proposer. Absolute supported-relation coverage remains low; better retention versus the ablation does not establish superiority over all baselines or caption quality.

Dev actor wins are271,260,268 out of334 for candidate text and268,269,264 for the relation-bank control. This ordering measure also does not improve uniformly; the text input's AP/retention advantage must not be described as solved actor grounding.

## Crosscheck of previously inspected full100 errors

Reusing R68's method-aware assistant assessments, seed7 selects a previously contradicted background spectator→motorcycle riding claim in image455667. Seeds42 and123 do not select any of the exact contradicted tuples recorded in that review. This is not exhaustive accuracy: unreviewed/promoted tuples and ambiguous claims remain, and no independent human annotation has been added.

Each candidate-text seed still selects the right-officer/right-motorcycle claim that was not in the original R68 selected-union review. It requires separate annotation rather than being automatically marked correct. The crosscheck is saved in `previous_visual_findings_crosscheck.json`; human-review fields are not overwritten.

## Decision

Retain candidate-specific text conditioning as an experimentally supported component under this setup. Keep fixed seed42 as the existing reference and do not pick the seed with highest AP or lowest FP. Full prompt generation and caption training remain premature.

The next substantive experiment should address supported-relation losses and the GT-box→detector-box gap, while preserving confirmed supervision and unknown masking. A proposed detector-box training change would require new frozen training features; this three-seed run does not already test that hypothesis. Independent full-candidate annotations and a newly fixed evaluation are still needed before a publication claim.

## Artifacts and reproduction

Output: `C:/KhoaLuanTotNghiep/Code/output/triplet_verifier_seed_stability_3seed`. It contains all new seed heads, logs, full100 predictions/prompts, frozen transfer summaries, aggregate JSON, per-seed SHA checks and visual crosscheck. Compact review archive excludes checkpoint binaries and does not rerun seed42.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_triplet_seed_stability.py --output 'C:/KhoaLuanTotNghiep/Code/output/triplet_verifier_seed_stability_3seed'
```

All four new training/evaluation pipelines completed with source-hash, fixed-config and image-overlap checks. Seed42 reuse matched both locked head hashes. No weekly report document was edited.
