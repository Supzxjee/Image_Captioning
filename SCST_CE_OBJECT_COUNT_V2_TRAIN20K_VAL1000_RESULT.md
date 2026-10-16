# Object Count v2: fewer hallucinations, CIDEr not above the original fixed baseline

Status: completed user-run Kaggle comparison; local artifact audit PASS. Expansion gate FAIL because fixed-reranked CIDEr is below the original QFormer fixed baseline. No test evaluation or automatic notebook launch.

## Results on the same validation1000

| Model / selection | BLEU4 | METEOR | CIDEr | CHAIRs | CHAIRi | Object recall | Hallucinated captions / mentions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original / raw | .365078 | .284004 | 1.218230 | 3.60% | 2.4523% | 44.0113% | 36 / 36 |
| SCST+CE / raw | .369736 | .286245 | 1.230521 | 3.80% | 2.5710% | 44.1056% | 38 / 38 |
| Count v2 / raw | .371786 | .287431 | 1.230817 | 3.30% | 2.2373% | 44.1685% | 33 / 33 |
| Original / fixed | .384152 | .294604 | 1.275819 | 3.80% | 2.5828% | 44.9544% | 38 / 39 |
| SCST+CE / fixed | .372259 | .293657 | 1.260458 | 3.80% | 2.6264% | 45.2373% | 38 / 40 |
| Count v2 / fixed | .375178 | .293413 | 1.264003 | 3.20% | 2.1164% | 45.1430% | 32 / 32 |

The raw count-reward treatment removes8 control hallucinated captions and introduces3:38→33. Fixed selection removes9 and introduces3:38→32. Fixed hallucinated mentions drop40→32 while total recognized mentions drop1523→1512; the CHAIRi improvement therefore reflects fewer errors, not merely a larger denominator. Fixed correct unique objects decrease1439→1436 versus the control, a small recall tradeoff; they exceed the original fixed baseline1430.

Raw CIDEr versus matched control is effectively unchanged (+.000296). Fixed CIDEr improves+.003545 over matched control, but remains−.011816 below original fixed. This is a useful hallucination-reduction signal in this run, not an across-the-board model-quality win.

## Paired CIDEr uncertainty

5000 paired image bootstrap resamples, conditional on these checkpoints and fixed validation IDF:

| Treatment minus baseline | CIDEr delta | Conditional95% interval |
|---|---:|---|
| Raw vs original raw | +.012586 | [−.014575, +.039865] |
| Raw vs matched SCST+CE raw | +.000296 | [−.017760, +.017804] |
| Fixed vs original fixed | −.011816 | [−.039442, +.014857] |
| Fixed vs matched SCST+CE fixed | +.003545 | [−.014799, +.021173] |

All intervals include zero. No statistically established CIDEr gain is claimed. Hallucination counts are descriptive paired outcomes from one reused validation subset/seed; they are not a multi-seed significance result.

## Verification

- Input hashes, train20k/validation1k cohort IDs and original predictions match the prior object-reward review. Each arm completes2500 finite updates; logged CE/SCST and object-reward arithmetic checks pass.
- Count penalty and aliases_v2 are explicitly versioned. Training target rows:20000, eligible19885, neutral115, held-out target rows0.
- Both arms report the same strict runtime and initial model hash. Control first32 updates match every common recorded field from the successful same-GPU replay; runtime and initial-model identities also match. Full-run histories do not contain a post-step model hash, so this is a history/trace comparison, not a full2500-step weight replay.
- Official PTB/CIDEr/BLEU/ROUGE recomputed for all6 policies; METEOR cross-checked only. CHAIR reaggregated from supplied per-caption annotations. Candidate pools, journals and prediction identities checked.
- Checkpoint file identities recorded: control `ee09f4f3861a9057722fa76951c323641f67148beecfd1a1e640638113dd1696`; treatment `acaa5c6492e45449bd1cbaaed0439585b4ec56b6187391152712e88c693d32cf`. Weights absent from ZIP.
- Training summaries:1193.62s control,1049.57s treatment; excludes notebook bootstrap and subsequent evaluation.
- Source ZIP5862955bytes, SHA256 `ab28c4f7927d5c17daad908cfb011ddc1ccab500d327cc62ee1d02f3eba0bf9d`.

The earlier fraction-penalty trial used different runtime controls. Its scores are historical context, not an isolated count-versus-fraction ablation. This trial changes both penalty and parser, so neither contribution can be isolated. COCO annotation support is not proof of physical object absence; no relation-grounding improvement is established. Original source datasets/detections and trained tensors are not present for independent end-to-end replay.

## Decision

Retain this variant as a lower-hallucination candidate, preserve its checkpoint, and do not promote it as the best overall model or increase training budget yet. The next focused experiment should lower the object-reward weight from1.0 to0.5, keeping count penalty, parser, strict runtime, budget and comparison protocol fixed, to examine the quality/hallucination tradeoff. This would be exploratory validation tuning, not evidence that0.5 is optimal. Do not change fixed reranking weights simultaneously. No follow-up training has been launched or notebook prepared as part of this result review.
