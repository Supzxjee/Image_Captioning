# Training object reward: completed train20k / validation1000

Date:2026-10-16. **Conditional CIDEr improvement over the matched anchored control was observed. Object-hallucination reduction was not demonstrated; the prespecified expansion gate failed.** Do not call this a completed semantic-alignment method or promote it automatically to full training/test.

Both independent jobs restarted from original QFormer epoch10, with the same train20k/validation1000 cohorts, batch8,2,500 updates, AdamW5e-6, FP32, dropout disabled and seed42. Control: CE+.05SCST(CIDEr). Treatment: CE+.05SCST(CIDEr+object reward). Object coefficient1; object reward=.5salient unique recall minus unsupported unique fraction. Only training-image labels entered reward construction. Executed core794034c; manual launcher15924f2.

## Same1,000 validation images

| Generator / policy | BLEU4 | METEOR | CIDEr | CHAIRs ↓ | CHAIRi ↓ | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original rank0 | .365078 | .284004 | 1.218230 | 3.60% | 2.4523% | 44.0113% |
| Anchored control rank0 | .372126 | .288218 | 1.235533 | 3.70% | 2.4933% | 44.3571% |
| Object reward rank0 | .377448 | .289429 | 1.256351 | 3.90% | 2.6702% | 44.5772% |
| Original fixed | .384152 | .294604 | 1.275819 | 3.80% | 2.5828% | 44.9544% |
| Anchored control fixed | .371848 | .294023 | 1.262949 | 3.50% | 2.3669% | 45.3002% |
| Object reward fixed | .381192 | .295484 | 1.287529 | 3.50% | 2.3483% | 45.6146% |

Fixed is the unchanged decoder/CLIP/OCC .3/.4/.3 policy. No UPT or new reranking weights. These are subset scores, not directly comparable to full-val5000/test5000.

Raw treatment versus matched control: CIDEr+.020818, BLEU4+.005321, METEOR+.001211 and recall+.2201 percentage points. However hallucinated captions37→39 (4 removed,6 introduced), and hallucinated mentions37→40. This does **not** establish that the object penalty reduced raw hallucination.

After fixed reranking: treatment CIDEr+.024580 over matched control; both have35 hallucinated captions and36 hallucinated mentions. Five caption errors are removed and five introduced. Correct unique objects increase1441→1451 out of3,181; recall rises by.3144 percentage points. CHAIRi falls slightly because recognized mentions rise1521→1533 while hallucinated mentions stay36. Do not describe that denominator change as fewer hallucinated objects.

Versus original-fixed, treatment CIDEr+.011710, CHAIRs3.8%→3.5% (38→35 captions;12 errors removed,9 introduced), recall+.6602 percentage points. BLEU4 is still lower (.381192 versus .384152). These point estimates do not constitute an across-metric or repeated-run improvement.

## Conditional uncertainty and repeat stability

Paired5,000-resample bootstrap over the1,000 images, validation IDF held fixed:

| Comparison | CIDEr delta | Conditional95% interval |
|---|---:|---:|
| Object raw minus original raw | +.038120 | [.012173,.063246] |
| Object raw minus current anchored control raw | +.020818 | [.002419,.039795] |
| Object fixed minus original fixed | +.011710 | [-.014116,.037691] |
| Object fixed minus current anchored control fixed | +.024580 | [.003546,.045764] |

The intervals against the **current matched control** exclude zero, providing a clearer CIDEr signal on this fixed sample than previous pilots. They are not across-seed confidence intervals, are not adjusted for repeated method exploration, and use previously inspected validation. The final-policy comparison against original-fixed still includes zero.

The anchored control is not an exact repeat of R119:247 raw and354 fixed captions differ, despite matching declared seed and settings. Raw CIDEr is close (1.235314→1.235533), but fixed CIDEr changes1.282620→1.262949. The cause is not established from the ZIP; do not assert a specific CUDA or library cause. Use the current control for the current treatment contrast, retain this reproducibility limitation, and assess repeated-run stability before claiming a paper-level gain.

## Gate and next decision

Raw CIDEr and recall gates pass; fixed preservation gates against both original and matched anchored control pass. The raw CHAIR gate fails:3.9%/2.6702% exceed original3.6%/2.4523% and control3.7%/2.4933%. Keep the recorded gate failure unchanged.

Preserve both current checkpoints and the original reference. The object-reward checkpoint is a development candidate with improved caption metrics, not a validated hallucination-control solution. The next useful decision is to verify training/evaluation repeat stability and understand why the raw object penalty did not reduce held-out hallucination, before increasing epochs, sweeping coefficients or adding relation terms. No new training or test run is authorized/executed by this result audit alone.

## Artifact audit

ZIP SHA256 `cbe39cb26f07f9218a60e6b0f5bfa310541b215d30c202106b58adb1e5956035`,5,816,965 bytes. Protocol SHA256 `80f00af9c99387f200bbdf5f308181c38ca7cfe74e46338063869a987ed3b97a`.

- Protocol/input identities, original predictions and train/validation cohorts match the previous review exactly. Source dataset JSON and full detector file absent locally; full split provenance and fixed-rerank formula were not independently replayed.
- Both histories contain2,500 finite updates through20,000 images. Logged weighted losses and sampled/greedy CIDEr+object reward identities match. Their mean differences match logged advantages; this is arithmetic verification, not replay of actual gradient updates.
- Target bundle SHA256 `6301d13e9d242c8c5930bb7d747040aa721b0f592c116180807a7dfc7b675750`. Exactly20,000 training rows, no validation rows.19,885 reward-eligible images;115 have no supported class after reference union and receive neutral object reward.121 annotation sets were empty before union. Support/salience/neutral states validate against the COCO80 schema; source annotation cache identity matches the inspected pinned cache. Raw training references and detector exemptions were not independently reconstructed locally.
- Three ordered1,000-image beam pools and2x500 journals per pool match exactly; CLIP attachment preserves candidates and language scores. Predictions and CHAIR caption identities match image cohorts.
- Official PTB CIDEr, BLEU1–4 and ROUGE-L recomputed for all six policies and agree to1e-10. METEOR cross-checked only. CHAIR reaggregated from supplied per-image annotations, not rerun against independent external ground truth.
- Weights absent from ZIP: recorded checkpoint hashes were not independently loaded or rehashed. Training summaries report841.2s control/840.9s treatment for this run.

Training rewards share COCO object ontology with CHAIR; annotation absence is not physical-absence proof, and this reward does not verify instances, attributes, negation or relations. Reproduce the read-only audit with `audit_scst_comparison.py --source <object-review.zip> --prior-review <anchor-review.zip> --output <audit.json>`. Structured result: `audit/scst_ce_object_train20k_val1000_result.json`.
