# Frozen triplet verifier with a learned soft R50 prior — R80–R81

The soft-prior prototype is implemented and trained locally. It restores some correct relations blocked by the hard proposer gate and suppresses four newly inspected errors, but **does not outperform the identically calibrated triplet-only control on matched detector transfer**. Neither new head is promoted.

## Method and control

Freeze the preserved full-GT TripletVerifier and R50 residual scorer. For each of six predicates, combine their continuous evidence:

`fused_logit = nonnegative_weight_triplet * standardized_triplet_logit + nonnegative_weight_R50 * standardized_R50_score + bias`.

There is no R50 threshold veto. Per-predicate mean/population standard deviation uses only known labels on1,141 matched detector training pairs /616images. Standard deviation floor.001. Train100Adam steps,lr.03, with masked BCE plus.2 type-balanced confirmed actor/predicate ranking (margin1logit) and.01 ridge. Weights are constrained nonnegative by softplus. Fixed seed42; initialization is deterministic and no seed/epoch selection occurs.

Both conditions have18 nominal scalar parameters. The control zeros the normalized R50 input, leaving12 effective parameters. It uses identical labels, schedule, normalization policy, losses, initialization and threshold rule, so the comparison isolates the addition of continuous R50 evidence. No object whitelist, manual audit labels, new negative labels, or dev-dependent threshold adjustments are used. Unknown labels remain masked.

Thresholds are fitted on training scores only using the existing >=90% empirical precision / >=5known accepts rule. Sigmoid outputs are not calibrated probabilities. Frozen source heads previously trained on GT crops of these training images: this is **in-sample stacking**, not out-of-fold calibration, and training precision can be optimistic.

Locked protocol: `audit/triplet_soft_prior_protocol.json`. Frozen head/source hashes, model revision, feature order/normalization/finiteness, labels, zero training/evaluation image overlap, and recomputed R50/triplet scores are checked. Legacy R55 transfer archives use their own pair-hash/manifest reader rather than the newer full-candidate archive schema.

## Common detector development validation

Same310pairs /128images;196known positive labels. Eating/talking_on each have only4positive dev examples.

| Method | Macro AP | Correct actor contrasts | Accepted correct /wrong /unknown | Known precision | Conditional recall |
|---|---:|---:|---:|---:|---:|
| Triplet-only calibrated control | 0.771942 | 152 /194 | 80 /25 /16 | 76.19% | 40.82% |
| Triplet + soft R50 prior | 0.742097 | 162 /194 | 73 /15 /15 | 82.95% | 37.24% |

R50 evidence improves actor ordering and reduces known false acceptance on this dev split, but lowers macro AP and recall. A90% training-precision target does not guarantee90% dev precision.

## Frozen R55 matched detector transfer

Same364pairs /286images,243known positives. No threshold refit; known-label precision excludes unknown claims. Counts are before per-image cap, and the pool excludes unmatched detector proposals.

| Method | Correct kept | Wrong kept | Unknown | Known precision | Conditional recall |
|---|---:|---:|---:|---:|---:|
| Preserved hard R50 AND triplet pipeline (context) | 35 | 1 | 8 | 97.22% | 14.40% |
| Original triplet alone, original thresholds (context) | 118 | 12 | 38 | 90.77% | 48.56% |
| Triplet-only calibrated control | 116 | 11 | 36 | 91.34% | 47.74% |
| Triplet + soft R50 prior | 102 | 12 | 27 | 89.47% | 41.98% |

Soft prior loses14known correct claims and admits1additional known wrong claim relative to its control. Transfer macro AP is0.825026 versus control0.827678. Actor contrasts improve54/60 versus51/60. Better actor ordering therefore does not establish better acceptance of whole triplets.

Historical hard-gated and original-triplet results use different calibration rules/domains; they are context, not the controlled fusion ablation. The improved retention relative to the hard gate comes with substantially more known false claims. No caption improvement can be inferred.

## Full100 candidate regression

Same1,321detector pairs /100images,cap3 and detector inventory preserved.

| Method | Above threshold before cap | Selected relations | Images with relations |
|---|---:|---:|---:|
| Triplet-only calibrated control | 154 | 87 | 42 |
| Soft R50 prior | 97 | 69 | 39 |

Both methods retain the previously inspected child-holding-donut and foreground-man-holding-frisbee tuples. The soft prior excludes both background-person-riding-person errors; the control still selects them. Both methods exclude the two inspected standing-men-sitting-on-cake-table tuples. These exact checks reuse the purposive, method-visible six-claim spot-check and are not independent accuracy estimates.

However, the soft prior selects four older assistant-reviewed contradicted tuples:

- `459543`: far-edge person riding another person's motorcycle.
- `418569`: incorrect person holding an airborne frisbee.
- `455667`: background spectator riding police motorcycle.
- `130024`: holding skis at the feet when poles are held.

These reviews are used only for regression reporting, never training. Many of the69selected claims are unreviewed; unknown/new claims are not automatically correct. Full100 remains inspected development data, not an untouched test set.

## Decision

Keep the preserved hard-gated pipeline and both source heads unchanged. The new fusion is a useful counterexample to treating improved actor ranking or success on a few failure cases as overall improvement. It does not meet the current acceptance-quality requirement.

A scalar combination of two scores cannot recover missing evidence about exact instance contact or target identity. The next research question is whether richer instance-specific visual evidence can improve that discrimination; this trial does not establish which architecture would succeed. Do not tune new thresholds or weights to these reviewed examples. No claim of scientific novelty is made for this small score-fusion prototype.

## Reproduce and artifacts

Run `run_triplet_soft_prior_cached.py` with explicit training/transfer/full100 archives, preserved triplet directory, R50 head, and frozen reference/scorer directories. The script runs both protocol conditions and writes separate heads/caches; it never overwrites an existing production prompt cache.

Source triplet head SHA256: `de8124a30555b39eabdf45cdf649d599788174361c94bd8036ebed714f61fab4`.

R50 head SHA256: `86930afcbed9573a4a48a2a0b5a7d94507d52b67aaa63b73e5a5f7717907a438`.

Control fusion head SHA256: `fcc2e01428420ea447aff7b4ce58af9e9209f7dad2cf5dfc7006f61ba2279da7`.

Soft-prior fusion head SHA256: `1e67800d3bbe58b14d763a7ff66772a034279e14c09bd7a27a4ac9d31733e03b`.

Local artifacts: `C:/KhoaLuanTotNghiep/Code/output/triplet_soft_prior_v1`. Compact review ZIP: `C:/KhoaLuanTotNghiep/Code/output/triplet_soft_prior_v1_review.zip`, containing both small fusion heads, histories, summaries, train-only normalization/thresholds, detector/full100 predictions, prompt caches, protocol and this result document. Source image/feature ZIPs and frozen large heads are referenced rather than duplicated. No Kaggle GPU quota was used.
