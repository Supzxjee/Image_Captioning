# Grounded gain C/H/T pilot — completed quality No-Go

28/10/2026. All three arms completed500updates on the same2,000TRAINimages, then raw greedy evaluation on the same historically reused development1,000. No local model run or new caption inference was used for this audit.

## Outcome

| Arm | CIDEr (raw) | CIDEr ×100 | CHAIRs | Hallucinated captions | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen A | 1.205513 | 120.55 | 5.20% | 52/1000 | 45.80% |
| C: shared adaptation objective | 1.196397 | 119.64 | 5.20% | 52/1000 | 45.77% |
| H: endpoint-ranking control | 1.208655 | 120.87 | 5.00% | 50/1000 | 45.83% |
| T: proposed directional gain | 1.201013 | 120.10 | 5.20% | 52/1000 | 45.68% |

T does not meet the predeclared continuation gate. It is below A and H in point CIDEr, and the required positive paired confidence intervals are absent. Keep A; do not scale, sweep or add seeds to rescue this recipe. This is a COMPLETED_QUALITY_NO_GO, not an interrupted computational run or evidence that all semantic alignment is impossible.

## Paired image uncertainty

10,000shared image-bootstrap draws, seed42, percentile95%intervals. Numbers below use RAW CIDEr, not the ×100 display.

| Difference | Point | 95% CI |
| --- | ---: | --- |
| C − A | −0.009116 | [−0.021260, +0.002482] |
| H − A | +0.003142 | [−0.010935, +0.017172] |
| T − A | −0.004500 | [−0.020298, +0.010676] |
| T − C | +0.004616 | [−0.006074, +0.015299] |
| T − H | −0.007642 | [−0.017687, +0.001293] |
| H − C | +0.012258 | [+0.003299, +0.022511] |

H improves over C under this one-seed development comparison, but H−A crosses zero and its point gain is below the+.01raw continuation target. Do not call H a robust new best, a confirmed baseline improvement, or an original method; H is the adapted prior-inspired control. T−H crossing zero also does not prove equivalence.

H−A/H−C are secondary exploratory contrasts; these percentile intervals are not adjusted for multiple comparisons. H−C is a useful signal to discuss, not independent confirmatory evidence of a new method.

CHAIRi for T is0.032515 versus A0.032475: the strict no-worse guard fails numerically, but hallucinated mentions are unchanged at53 and the object-mention denominator falls1632→1630. Do not describe this as an increase in the count of hallucinations. H has51mentions versus A53; two fewer hallucinated captions is a small descriptive change, not independent factuality confirmation.

## Validity checks

- Actual archived bytes for all14runtime sources match each arm's declared SHA; same policy/preparation receipt and exact initial head digest.
- Both ranks in each arm report500completeupdates, identical final head digests within arm, and unchanged frozen-A tensor digest. Physical checkpoint files are absent from these small ZIPs, so these tensor identities are worker-reported, not independently recomputed locally.
- All user-run preflight flags pass, including zero-init parity, nonzero sample/rescore parity, numerical gradient fixture and distributed head parity. Runtime manifests are identical across arms/ranks.
- Baseline A captions match the locked baseline on all1,000IDs. Per-image CIDEr averages reproduce summaries. CHAIR counts were independently reaggregated from saved evaluator outputs.
- Every history has exactlysteps1–500. Logged main/auxiliary formulas, global-eligible auxiliary scaling and common averaged gradient norms are consistent. All logged auxiliary completeness residuals satisfy the pinned tolerance.

C/H/T change102/125/168captions relative to A. Thus the new branch affects outputs; output sensitivity alone has not produced the desired quality improvement.

## Mechanism evidence and limits

Across matched training minibatches in updates401–500, each arm has800eligible-group observations. Among positive netgain groups, the median paired-support shareCG/Delta is1.47%forC,1.52%forH,1.03%forT. Only2ofT's458positive-gain observations meet the80%support target. The mean T netgain across these800observations is−0.004398 and meanCG−0.000218.

These logs do not show the intended directional constraint being attained in this pilot. They are observations along each arm's OWN TRAINING trajectory, not a final frozen-head comparison on held-out grounding data. Scalar loss sizes do not establish gradient domination by SCST or identify the cause of failure. No claim that regions were correctly recognized, that attribution is causal, or that CLIP/Q-Former has reached a ceiling follows.

No held-out phrase/box labelbank is packaged. Image bootstrap measures image variability conditional on one seed, not seed variance or reuse/selection bias. No test split was used. Full machine-readable audit, hashes, guards and diagnostics are in `RESULT.json`; code is `audit_gain_results.py`. Raw cached ZIP text remains local/ignored.

The numerical summary and scope were sent to Astra without requiring it to reread the ZIPs. No new notebook, GPU launch or method change was authorized by this result audit.
