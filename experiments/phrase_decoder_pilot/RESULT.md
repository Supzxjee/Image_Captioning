# Phrase decoder CE versus CE+CROSS — 20 October 2026

The matched technical pilot completed. Both arms improve held-out template preference from 10/14 to 13/14 events. CROSS does not improve the number of correct preferences or demonstrate better free generation than CE. Do not promote CROSS to full caption training on this evidence.

## Artifact validity

`compare_reviews.py` returns `MATCHED_ARTIFACT_COMPARISON_PASS` on the user-provided archives. Source bytes, locked manifest, input checkpoint, staged features, vocabulary, starting predictions, runtime and optimizer configuration match. Both ranks finish 300 updates with unchanged original parameters. Both arms have identical model and optimizer hashes after the common 60-update CE warmup. Saved scores, margins and summary statistics were independently recomputed by the auditor. No training or inference was rerun locally.

Input archive hashes and the machine-readable comparison are in `COMPARISON_RESULT.json`. Checkpoint adapters are intentionally absent from these review ZIPs; saved provenance checks do not constitute independent re-execution of the checkpoints.

## Results

The 7 adaptation holdout groups contain 14 events; none of their image IDs occur in the 14 fit groups. These are previously human-reviewed train/restval examples, not an independent benchmark.

| Holdout measurement | Before, evidence on | CE after | CE+CROSS after |
| --- | ---: | ---: | ---: |
| Correct preference between the two supplied phrases | 10/14 | 13/14 | 13/14 |
| Groups with correct preferences in both directions | 3/7 | 6/7 | 6/7 |
| Mean correct-minus-incorrect log-probability margin | 0.299 | 0.481 | 0.721 |
| Positive phrase token NLL, lower is better | 2.421 | 0.659 | 0.740 |

Both arms reach 28/28 preference decisions and 28/28 normalized exact template matches on fit events. On holdout, both have 5/14 normalized exact template matches and only 1/7 groups matching both templates. Exact matching is only a string check: acceptable paraphrases may differ, and this is **not semantic accuracy**. Free generation differs between CE and CROSS in only two holdout events.

The geometry-only nearest-neighbor control obtains 9/14 preferences and 2/7 groups correct in both directions. This weaker control does not isolate image evidence from geometry in the learned models. Evidence-on/off sensitivity also does not establish correct visual grounding.

CROSS-minus-CE agreement is zero on every holdout group. Its descriptive bootstrap interval is consequently [0,0]; this says nothing about equivalence on a population or statistical power.

## Image-grounded qualitative inspection

All 14 held-out outputs were read alongside saved images and subject/object boxes. These are researcher observations, not new human adjudication labels.

- **208095, two people in the same image:** one is standing on a surfboard in the water, while the selected other person is carrying a board beside the water. Both arms generate `a person standing on a surfboard` for both events. The carrying event's correct-minus-incorrect margin worsens from -0.738 with CE to -1.138 with CROSS. Increasing the average margin can also increase confidence in a wrong case.
- **70642, two skateboard events in the same image:** for the child carrying a board, CE generates `a person standing on a skateboard in a skate park.` CROSS changes this to `a person is standing with a skateboard`. This removes the explicit standing-on error, but still omits the intended carrying relation. Both supplied-phrase preferences count as correct, with small positive margins of 0.015 and 0.038. This illustrates the difference between teacher-forced ranking and free generation.
- **377570, selected phone event:** CE mentions looking at a phone; CROSS instead generates `a person sitting in a chair with a laptop`, omitting the target phone relation. The image does contain a laptop, so this is target omission rather than evidence of object hallucination. Both still prefer the correct supplied phone phrase.
- **16737 / 335594, phone near ear:** both arms generate generic holding-in-hands/hand text, omitting the discriminating near-ear detail. Do not mark all such generic descriptions logically false; the missing distinction is the problem.
- **107584 / 162520:** both arms correctly generate the intended standing-on versus carrying surfboard phrases. The result is not uniformly negative.

## Decision and next diagnostic

Retain CE as the simpler reference; do not claim CROSS superiority. The pilot supports that direct phrase supervision can alter the decoder, but does not yet establish robust instance-grounded visual-semantic alignment, full-caption gains, or architectural novelty. The perfect fit outputs and imperfect holdout details are consistent with template memorization; they do not prove its cause.

Before more data collection, SCST, or full training, a focused evaluation-only intervention should hold the image/global memory fixed and swap selected person-object evidence, while separately testing visual features versus geometry. Check whether the generated relational phrase follows the specified event, especially in 70642 and 208095. This would distinguish instance selection failure from broad scene/template preference. No additional experiment has been launched or notebook created as part of this review.

Single seed, tiny selected holdout, GT boxes, restricted phrase families and globally contextual CLIP patch features remain material limits. No CIDEr/CHAIR or independent full-caption test was evaluated here.
