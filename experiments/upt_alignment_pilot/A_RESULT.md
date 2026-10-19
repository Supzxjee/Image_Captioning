# A: continued-training control, val1000

User-run Kaggle ZIP received and checked on 2026-10-19. **Artifact audit PASS.** This run adds no new UPT region/text branch; it measures continued training of the preserved SCST+CE+object0.5 baseline. D must be compared with this trained control, not only with the starting checkpoint.

| Metric | Starting checkpoint | A end checkpoint |
|---|---:|---:|
| CIDEr | 1.189066 | 1.205513 |
| BLEU-4 | 0.350305 | 0.351719 |
| METEOR | 0.285374 | 0.287266 |
| ROUGE-L | 0.572772 | 0.571341 |
| CHAIRs | 5.80% | 5.20% |
| CHAIRi | 3.5452% | 3.2475% |
| Object recall | 45.6809% | 45.8039% |
| Captions with hallucinated objects | 58 /1000 | 52 /1000 |

CIDEr increases .016447 (+1.38%). Conditional paired bootstrap95% interval is **[-.003994,+.036571]**, so this sample does not establish a reliable positive population gain. It uses saved per-image CIDEr and fixed val-IDF, not independent retokenization or seed uncertainty. 377 captions change;14 hallucinated captions become clean and8 clean captions gain hallucinations. Hallucinated mentions58->53; object mentions1636->1632, so the CHAIR change is not just denominator inflation.

For the269 relation-mode images, CIDEr1.331000->1.322335 decreases. The722 object-only images improve1.141482->1.167390. These are subgroups defined by UPT output mode, not relation-correctness annotations. Do not claim relation alignment from A or from whole-set CIDEr alone. The9 no-object images are too few for strong conclusions.

Both T4 GPU ranks completed1250 synchronized updates (global10000 images, batch8). First200 updates are CE-only; remaining1050 use CE+.05SCST(CIDEr-D+.5object_count_v2). Recorded training loop time is526.82s (8.78min), excluding setup, staging, IDF preparation and evaluation. Both ranks begin from the same model digest; all logged pre-clip gradient norms agree. Saved-source hashes, protocol/target hashes and cohort identity match the published pilot; model/DDP/metric preflight tests PASS in the recorded setup log.

Audit replays Karpathy split and all5000 val references from a pinned local dataset, verifies two non-overlapping prediction shards and1000 unique IDs, reaggregates saved per-image CIDEr/CHAIR outputs, and checks finite training traces and objective/reward arithmetic. It does **not** rerun inference or independently recalculate caption tokenization/CIDEr/BLEU/METEOR/ROUGE. The review ZIP excludes weights and HDF5, so the recorded checkpoint hash and final cross-rank parameter identity cannot be independently checked from this ZIP.

Preserve `/kaggle/working/upt_alignment_pilot_A/checkpoint.pth`; recorded SHA256 `cd65c49afa77081ccb5355150fe614626ad601dde397c179b875515fb602ea0d`. No checkpoint upload is needed yet. Keep D's original starting checkpoint unchanged; do not warm-start D from A. When D arrives, first compare its origin predictions, input/cohort/target identity and runtime against A, then compare D-on with A-end and D-off.

No test used. This is one seed on historically reused val1000 with raw greedy decoding; do not compare these scores directly with old beam/reranked test scores or promote A as the new best model on this evidence alone. B/C remain optional attribution runs after A/D, not completed experiments.

Reproducible artifact checks and exact values: [audit script](audit_review.py), [audit output](A_result.json). The assistant audited artifacts only and did not launch a notebook.
