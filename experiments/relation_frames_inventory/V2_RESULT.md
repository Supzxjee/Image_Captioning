# Relation-frame data v2 — feasible metadata manifest

28/10/2026. **DATA_V2_PASS_PENDING_AUDIT_FEATURES**. CPU metadata selection only; no neural execution, model scores, feature extraction or GPU use.

## What changed, and why

V1 remains `DATA_COVERAGE_NO_GO`: its independent image split and per-predicate quota left insufficient held-out coverage. V2 was specified before any learning or model scoring. It selects true/false pairs first and then excludes all their images from training. It retains the same16predicate labels, validated VSR truth labels, >=3unanimous validators/no dissent, unique noncrowd GT instances and top12predicted boxes with exact class and IoU>=.5. No split-seed search, label replacement, GT-box substitution or relaxed detector threshold.

V2 pools all3,358previously detector-matched records, rather than using v1's capped512/116pair pools or v1's image partition. Hash one-to-one pairing of the full pool gives1,034candidate edges; no Cartesian expansion. The two strata and64/256sizes were fixed in Astra's v2 decision before solving.

## Feasibility result

| Selected cohort | Pairs | Distinct images | Distinct statements | Projective/containment | Contact/proximity |
|---|---:|---:|---:|---:|---:|
| Relational train | 256 | 484 | 219 | 151 | 105 |
| Adapter-held-out | 64 | 128 | 63 | 32 | 32 |

All128held-out images are distinct and disjoint from both relational train and caption-adapter TRAIN2k. All219records belonging to these held-out images, including statements not selected for evaluation, were removed before constructing train pairs. Re-pairing the remaining3,139records leaves928candidate train pairs. Final train selection preserves one example per available predicate, then fills by fixed hash rank.

Both cohorts include all16predicate labels; this is a coverage fact, **not a generalization claim**. Held-out counts: touching26, on8, beneath5, in5, next-to3, under3, below2, contains2, inside2, on-top-of2; above/far-away-from/far-from/near/outside/within each1. In train, near/within also have only1pair each. Primary probe is the predeclared equal-weight two-stratum ranking accuracy; per-predicate results with very small denominators are descriptive only. Touching still dominates its contact/proximity stratum.

Image grouping, truth labels, entity concepts, class/IoU matches, no duplicate pair records, and held-out predicate support in training all pass independent checks. Train can reuse an image across different statements; held-out cannot. The held-out cohort is unseen to these adaptation phases, **not unseen to frozen A's original pretraining**, and not the official VSR benchmark split.

## How selection was solved

SciPy1.16.3 / NumPy2.2.6 / bundled HiGHS, installed only in a Git-ignored local CPU solver folder. Two MILP stages, both reported optimal:

1. Maximize held-out predicate coverage subject to32/32strata, image degree<=1, caption TRAIN2kexclusion and256simultaneously feasible train-witness edges with matching predicate support. Optimum is16of16labels.
2. Fix that coverage, minimize the sum of fixed hash ranks of held-out edges. Sorted variables/constraints and deterministic solver; no seed trials. Multiple equal-cost optima are possible; the exact manifest is the reproducibility authority.

After selection, train is reconstructed from remaining records using the original one-to-one hash pairing. Train-witness constraints guarantee enough capacity; the actual final train and all disjointness checks are verified again. Solver timeout would be computationally incomplete, not proof of infeasibility. Neither happened. The solve/selection/report phase took about0.91s, excluding package installation, source materialization and synthetic checks; this is not a training runtime estimate.

Synthetic checks cover feasible constrained matching with a shared endpoint, exclusion of caption-training images, an infeasible stratum case, and rejection of repeated held-out images. The earlier full GT/cache checker was rerun only to persist the exact complete matched-record pool; its v1result is unchanged. Initial sandbox imports could not access the installed packages; approved CPU runs used the same commands with package access. No neural library was loaded.

## Frozen artifact and next prerequisite

`V2_MANIFEST.json.gz` SHA256:

`6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7`

It stores exact training/evaluation pairs, truth labels and verified predicted boxes, alongside GT provenance for verification. Training inputs must use the predicted boxes; GT is not an inference input. `select_v2.py` refuses to overwrite the frozen manifest with different bytes. Full source/master/adapter/parent-result identities are recorded in `V2_RESULT.json`; raw matched records remain Git-ignored.

The **metadata selection prerequisite passes**. Still pending: fixed32-record assisted/human label audit, RGB/frozen-feature availability and the exact original-prompt path for decoder probes. No training notebook or run is claimed complete. Novelty, auxiliary truth discrimination, decoder use and free-caption gains remain untested. V2data feasibility does not authorize or predict a positive model result.

Reproduce after `check_box_coverage.py` has materialized the pinned matched pool: `select_v2.py --solver-path <SciPy1.16.3/NumPy2.2.6 package directory>`. Use `--self-check` for the synthetic CPU checks alone. Preserve v1results and the v2decision separately.
