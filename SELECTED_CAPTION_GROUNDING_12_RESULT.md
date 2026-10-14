# Selected relations reach the caption model, but barely affect decoding

The completed 12-image probe changes model memory and candidate log probabilities, but changes neither the top caption nor the ordered wording of the three returned candidates. The current historical E3 checkpoint does not meaningfully use the new selected target evidence in this small development probe. No new caption-quality metric or improved checkpoint is produced.

## Verified execution evidence

Uploaded ZIP SHA256: `90f5334d94257ba76ac7808f478732f7757a2581760be25c7eeee6048ca08688`.

12 images were the first lexicographic eligible filenames from 23 same-class-alternative images in the inspected full100 pool. On contains 27 relation slots; off has zero active slots; swaps retain the original number of slots. The same saved caption checkpoint, original generic prompt, visual features, beam5 and candidate_count3 were used. Checkpoint SHA256: `7a9afaf9861ad57f0f5c4a05d3fc4e51ae94c36f014b30aa6c832cc9c00adae7`.

`analyze_selected_caption_probe.py` recomputes counts from the returned records and checks summary agreement, unique filenames, masked-off counts, retained swap counts and finite measurements. Two analysis tests pass, including separation of score changes from wording changes and rejection of inconsistent summaries/masks.

| Compared with on | Off | One target-box swap |
|---|---:|---:|
| Top captions changed | 0 / 12 | 0 / 12 |
| Ordered three-candidate wording lists changed | 0 / 12 | 0 / 12 |
| Images with changed candidate log probabilities | 12 / 12 | 12 / 12 |
| Mean relative full-memory norm change | 0.001112324 | 0.000002476 |
| Same quantity expressed as percent | 0.111232% | 0.000247594% |
| Mean absolute candidate logprob change | 0.001149808 | 0.000004215 |
| Maximum absolute candidate logprob change | 0.003759455 | 0.000021055 |

The memory statistic includes concatenated generic-prompt and visual memory, rather than a relation-only vector. It is a sensitivity statistic, not percentage accuracy or percentage causal contribution. Mean on evidence weight is .023309 across 27 slots (range .009500–.033128). These are learned module weights, not calibrated probabilities or measurements of the separate query gate. Accepted masks are null for this checkpoint; there is no V3 accepted-relation result here.

For image040423, all conditions return `a man talking on a cell phone while holding an umbrella.` For image130024, all return `a group of people standing on top of a snow covered slope.` These examples demonstrate unchanged wording only; the probe does not human-adjudicate caption quality.

## Interpretation and decision

The bridge is functioning in the limited sense that masking/changing its input changes downstream memory and decoding scores. The resulting perturbation is insufficient to change any returned wording here. More reliable relation annotations alone have no demonstrated route to a caption improvement through this frozen fusion path.

Stop further target-hinge training and manual annotation expansion for now. The next method to develop is a dedicated, instance-grounded relation memory that the decoder can attend to directly, with reliability weighting and optional absence handling. This is a proposed architectural experiment, not implemented or validated by this result. It must retain image evidence and avoid forcing unverified relations into captions. A subsequent small training pilot should show relation-memory use AND supported caption changes before scaling or claiming novelty.

Do not multiply the old gate just to force changed captions or promote a head from these inspected examples. Keep the existing baseline and E2 results for comparison. No CIDEr/BLEU/CHAIR improvement is claimed.

## Limits

- Small inspected development sample, selected for same-class alternatives, not a representative heldout benchmark.
- New selected relations are predictions; alternative targets have unknown correctness and may both support the relation.
- The original generic V2.1 prompt contains relation information in every condition; off does not remove all semantics.
- Historical checkpoint and new selected relations differ in distribution. The result does not prove another architecture cannot use relations.
- No repeated on/on numerical reproducibility condition was returned; tiny score changes alone are weak evidence. The systematic off memory changes and recorded masks are consistent with a live branch, but no per-word causal attribution is established.

## Artifacts and reproduction

`python analyze_selected_caption_probe.py --review <selected_caption_grounding_12.zip> --output <new analysis directory>`

`python -m unittest tests.test_selected_caption_probe_analysis`

Local preserved ZIP and analysis: `C:/KhoaLuanTotNghiep/Code/output/selected_caption_grounding_12_result`. Public aggregate evidence: `audit/selected_caption_grounding_12_result.json`. No source human annotation or checkpoint is published.
