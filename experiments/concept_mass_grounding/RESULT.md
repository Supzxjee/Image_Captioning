# C/G concept-mass pilot — 2026-10-29

Both user-run jobs completed 500 updates on matched 4000 unique TRAIN images and fixed dev1000. Runtime preflight passed on both ranks. Metadata-only review passed185 assertions; no neural execution locally.

| Model | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Words |
|---|---:|---:|---:|---:|---:|
| A | 1.205513 | 52/1000 | 3.2475% | 45.8039% | 10.213 |
| C | 1.225678 | 51/1000 | 3.1863% | 45.8654% | 9.987 |
| G | 1.229535 | 52/1000 | 3.2456% | 45.7731% | 9.968 |

C is continued SCST + KL to frozen A + positive concept-mass preservation. G adds weak-negative concept-mass suppression; architecture unchanged.

## Matched comparison

- C-A: CIDEr delta +0.020166, paired image bootstrap95%CI [0.002931394894035268, 0.038609952275659894]; CHAIRs delta -0.0010, CI [-0.009, 0.007].
- G-A: CIDEr delta +0.024023, paired image bootstrap95%CI [0.006684421412635773, 0.04280734745410855]; CHAIRs delta +0.0000, CI [-0.008, 0.008].
- G-C: CIDEr delta +0.003857, paired image bootstrap95%CI [-0.0032356779008074496, 0.011555245718140249]; CHAIRs delta +0.0010, CI [0.0, 0.003].

CIDEr improved versus A in this reused dev sample. The added G objective is not demonstrated to improve over C: its CIDEr interval includes zero, and hallucinated captions are52 versus51. G did not reduce hallucinated captions or mentions versus A. Its tiny CHAIRi change is denominator growth (53/1632 to53/1633), not fewer hallucinations.

## Mechanism evidence

G saw211 weak-negative events; C monitored213. Both had4447 positive anchors. No zero-event or missing-gradient failure was reported. Neutral-prefix70-class macro separation: A .10145701, C .10657654, G .10629720. Macro weak-negative mass rose from .0001295722 to .0001310964(C) / .0001313372(G); G does not show stronger conditioning than matched C on this diagnostic. Separation improvements are driven mainly by higher supported mass; this is not causal grounding proof.

## Decision and limits

Preserve both checkpoints and the C CIDEr gain as useful evidence. Do not label G a validated alignment contribution or launch longer training automatically. Request Astra decision on the smallest justified confirmation/change. Existing controls do not isolate the contributions of SCST, KL and preservation inside C.

One seed, reused dev1000, no test used. Image bootstrap does not establish seed robustness or negate development-set reuse. Weak negatives mean unrecorded by annotation/reference support plus detector veto, not verified visual absence. Checkpoint/HDF5 binaries were not included or loaded; source texts and hashes, reports, predictions and score alignment were checked.

Reproduce metadata audit with `audit_results.py` after caching each ZIP text once into ignored `review/C_loaded.json` and `review/G_loaded.json`.
