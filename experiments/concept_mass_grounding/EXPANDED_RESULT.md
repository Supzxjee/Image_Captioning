# Expanded A/C0/C evaluation — 30/10/2026

**Valid complete evaluation on the exact locked4000images. C has the highest CIDEr and lowest hallucination point estimates, but the prelocked joint success gate fails.**

## Cohort and validity

Freshfixed core6645673, exactGit/LFmanifest8f68ab9c..., same pinned A/C/C0 endpoints.4000uniqueprimaryimages, no currentdev/TRAIN overlap;16devtechnicalsentinels excluded. BothGPU workers completed2000primaryimages/model, all native wrapperdigests correct and all sentinel mismatches0. No training. Raw source texts agree with pinnedGit assets, metadataSHA matches receipt, all caption/CHAIR/per-image IDs and raw counts aligned.

| Same4000images | CIDEr ↑ | CHAIRs ↓ | Hallucinated mentions / all mentions | Recall ↑ | Words |
|---|---:|---:|---:|---:|---:|
| A | 1.189771 | 239/4000 (5.975%) | 254/6484 | 46.8087% | 10.12875 |
| C0 | 1.204560 | 235/4000 (5.875%) | 248/6502 | 47.1085% | 9.99500 |
| C | 1.207042 | 230/4000 (5.750%) | 243/6477 | 47.0296% | 9.93825 |

## Locked comparisons

- C-C0: CIDEr +0.002481,95%CI [-0.00135716233511175, 0.006262829041919476]; CHAIRs -0.00125,95%CI [-0.0027500000000000024, 0.0002500000000000002]. Repaired7, introduced2, exact McNemar p=0.179687.
- C-A: CIDEr +0.017271,95%CI [0.008818907607254522, 0.025852987175757087]; CHAIRs -0.00225,95%CI [-0.006000000000000005, 0.0015000000000000013]. Repaired35, introduced26, exact McNemar p=0.305677.
- C0-A: CIDEr +0.014789,95%CI [0.006062943648548707, 0.023628838061796158]; CHAIRs -0.00100,95%CI [-0.0049999999999999975, 0.0029999999999999957]. Repaired34, introduced30, exact McNemar p=0.707981.

CIDEr gains of both C and C0 over A are supported by conditional paired image intervals. C−C0 CIDEr remains uncertain. C reduces false captions239→230 and false mentions254→243 versus A as measured point estimates; unlike the pilot denominator-only CHAIRi change, the false-mention numerator is lower here. Neither CHAIRs nor CHAIRi reduction has a95%paired interval excluding zero. This is not proof of no effect, equivalence, or harm.

## Apply the prelocked decision

- CIDEr(C−A) lowerCI>0:PASS.
- CHAIRs(C−A) upperCI<0 plus exact p<.05:FAIL.
- CHAIRs(C−C0) upperCI<0 plus exact p<.05:FAIL.
- CIDEr(C−C0) lowerCI>−.01 and recall lowerCI>−.005 versus C0/A:PASS.

Preservation-relative, dual-goal and joint success flags are all false. Follow protocol: end the current claim/investment round; no tuning, coefficient/step/seed sweep or larger-n extension on this cohort to rescue it. Preserve positive CIDEr evidence and all fixed checkpoints. Do not turn this conditional conclusion into a claim that preservation cannot ever work. No distinct successful alignment contribution has been established. Astra has received verified compact facts for a next decision; a change to article scope or a new method requires discussion with the user.

## Audit and limits

The metadata audit has28165 assertions, including row-wise comparisons; these are not independent experiments. Reproduced saved per-image CIDEr means, all rawCHAIRcounts, aggregate ratios,20kpaired image bootstrap intervals and exact McNemar results. No model/checkpoint/HDF5 loaded or executed locally, and no official caption scorer was rerun locally; inputs/binary hashes are evidenced by user-run checked source/receipts.

Historically reused validation, one training seed and post-pilot selection. Image bootstrap does not remove those limits. No independent test, pooled5000 or extra semanticdiagnostic. CIDEr4000IDF differs from dev1000; do not compare their raw scores as a change in model quality. FullCHAIR80 measures object hallucinations, not all relation/attribute errors.

Stage took1447.79s; workers181.25s/177.57s include load/parity/inference. Those do not establish total job wall time, wholepipeline latency or account billing.

Reproduce from ignored cached `review/expanded_loaded.json` with `audit_expanded_results.py` (metadata/NumPy only). Source ZIPSHA256: `4397bbc5a1f47d6234cab232ebd57e5a098a49a7fe7d22e7921d080c1f91665e`.
