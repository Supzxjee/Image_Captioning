# Matched FULLFRAME versus PAIRCROP — received results, 21 October 2026

Both arms completed 300 CE-only updates on two T4 GPUs. The archives agree on initial parameters, baseline checkpoint, manifest/schedule, feature-cache and receipt hashes, source hashes, vocabulary, optimizer, loss and runtime. Original parameters remain unchanged, frozen hashes agree, and each rank has 848384 trainable parameters. Within each arm, both ranks have the same final hash. Before-training evidence-off records are identical across arms. All archived source hashes and manifest bytes pass verification. No technical reason to invalidate the comparison was found.

## Teacher-forced ranking, not free-generation semantic accuracy

| Partition | FULLFRAME correct in both directions | PAIRCROP correct in both directions |
|---|---:|---:|
| Fit | 14/14 | 14/14 |
| Previously exposed development | 6/7 | 5/7 |
| New-to-adaptation probe | 1/8 | 3/8 |

On the new probe, directional agreement is 9/16 versus 11/16. Mean positive margin is 0.19224 versus 0.53294. Neither quantity measures whether freely generated phrases describe the selected pair correctly.

## Free-generation observations, pending human review

PAIRCROP changes one skateboard event from `a person carrying a skateboard` to `a person standing on a skateboard`, and another from `a person on a skateboard on a ramp.` to `a person carrying a skateboard`. These changes agree with the previously human-confirmed target descriptions; the person/object selection and any additional facts still require visual review.

For all four surfboard groups, the carrying-target event still produces a phrase about standing on or being on a surfboard. PAIRCROP does not produce a carrying description for any of these four targets. This is evidence of a persistent generation failure in this contrast, not a human semantic accuracy score. Crop helps some skateboard cases but has not established a general solution across both families.

The prespecified primary criterion remains blinded human assessment of all 16 new-probe events, grouped into eight pairs: at least6/8 joint successes, netgain>=3, regressions<=1, with gains in both families, robust to unclear judgments. Do not replace this criterion with ranking, exact string matches, or fit performance. The current evidence does not justify full-scale training, a SCST rescue run, crop-margin/seed/checkpoint searches, or a claim of improved caption CIDEr. Finish the fixed semantic review; if the criterion fails, stop this crop variant.

## Review artifact

`compare_pair_sources.py` validates the archives and builds a blank, model-identity-hidden review at local `review/pair_source_blind/index.html`. Sixteen event cards each show the selected S/O boxes and two independently scored candidate phrases (32 judgments). X/Y order varies deterministically by event. The arm key stays outside the page. No ranking, target description, arm identity or model scores appear in the page; no labels are prefilled. The user may still infer some identities from previously discussed outputs, so this is identity masking, not a claim of a fully independent blinded study.

Statuses distinguish specific correct, generic, wrong relation/additional fact, wrong instance, and unclear. Partial export is allowed and marked incomplete. Return `pair_source_blind_review.json` for aggregation with the fixed key. Human labels remain separate from the archive and from any assistant interpretation.

Validation: archive assertions and isolated DOM tests PASS (32 unique candidate IDs, 16 image pairs, existing local image assets, no arm labels in payload, blank state, name/empty-export guards, partial/complete exports, unresolved and invalid-status handling). Browser visual QA was not completed because the browser tool blocks local file URLs. No local model, training or notebook was executed. Review archives do not contain the cache or adapter weight bytes, so those assets cannot be rehashed locally from these ZIPs.
