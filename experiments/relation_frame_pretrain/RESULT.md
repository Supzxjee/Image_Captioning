# Relation-frame pretraining result — 28/10/2026

**PRETRAIN_GATE_NO_GO.** Both user-run arms completed200updates and actual two-GPU preflight. Technical completion does not establish semantic or caption improvement. Keep the precommitted decision: do not start caption controls or extend/select epochs to rescue this pilot.

| Fixed held-out metric | RelPos | RelContrast |
|---|---:|---:|
| Correct image for the same assertion,64pairs |40/64 (62.50%)|41/64 (64.06%)|
| Position/containment stratum,32pairs |16/32 (50.00%)|22/32 (68.75%)|
| Contact/proximity stratum,32pairs |24/32 (75.00%)|19/32 (59.38%)|

The primary difference is **+1.5625percentage points**.10,000paired bootstrap draws with32/32fixed-stratum resampling and seed20261028 give95%CI **[-12.50,+15.625]percentage points**. None of the three locked conditions pass: RelContrast≥70%, gain≥10points, lowerCI>0. The counterexample loss changes which samples succeed:13Contrast-only versus12Pos-only,28both correct,11both wrong. No likelihood gaps have absolute value<1e-5; no exact ties.

The subgroup movement is descriptive: six additional successes in position/containment are offset by five fewer in contact/proximity. This reused64pair set is too small to establish a subgroup-specific method claim, and the failed overall gate cannot be replaced after seeing these scores.

## What was verified

Bundled source bytes match pinned Git commit `5a1130cd3913054fcef3121f4ad7bd36be28d33b`; both configs/result protocols and payload identity receipts match the notebook package. All64pair IDs/relations/strata match the frozen V2 manifest. Recomputed accuracy, finite margins, logged objective formulas and endpoint traces agree.

The worker reports actual model preflightPASS for both arms: identical initial digest `9748c05db637fab4c9d7ded25c3f84429d9204e122cc471fc4c1032b124b65b3`, same frozen embedding,4,720,672trainable and5,120,000frozen parameters,2TeslaT4s andTorch2.11.0+cu128. Maximum batch/single error8.345e-7; distributed/global-serial gradient error5.960e-8. Positive and auxiliary gradient paths, unknown-predicate masking, frozen-state and rank synchronization checks passed. The earlier cuDNN mode error was corrected; these are completed reruns.

Checkpoint byte hashes **reported by workers**, files absent from reviewZIPs:

- RelPos: `9650160651c8c9883c1bcf033afa1f27aa9c761cdefb7f75e87edf59ae7dc731`.
- RelContrast: `358b3278859e60ca75a078c1f6b9f6584cca5e28b9dd8bd09612885899af9c20`.

Root did not load checkpoints, rehash the fullHDF5, execute neural code or launch these runs. Preparation/source and resulting metadata are audited; numerical worker evidence is reported, not locally reproduced. Reported measured two-GPU-phase usage is .00626/.00653device-GPU-hours; excludes earlier notebook startup/preparation and is not total Kaggle billing.

## What the losses do and do not show

At logged update200, RelPos positiveCE=2.69871. RelContrast positiveCE=2.75472, observed-predicate trainingBCE=.024395, pairloss=.000663. Low auxiliary training losses show optimization on training samples; they do **not** give held-out truth-head accuracy or establish that the GRU/decoder uses the learned relation signal. Those held-out truth-head scores were not exported in this protocol. A disconnect between auxiliary classification and language conditioning is a possible explanation, not a measured diagnosis.

This frame-only stage has no free-caption generation, CIDEr or CHAIR result. It cannot be compared numerically to baselineA's captionCIDEr or called a caption improvement. The spatial scope, one seed, adaptation-heldout images potentially seen byA, and assisted sample audit remain limitations. Preserve both checkpoints for a separately specified diagnostic if warranted; ask Astra for the next research decision using this compact result, without rereading ZIPs or launching more training automatically.

Machine-readable numbers/provenance: `RESULT.json`. Audit code: `audit_results.py`, consuming local ignored cached review texts, with no neural imports.
