# Pair-prefix diagnostic — 20 October 2026

Both saved adapters pass restoration/replay on Kaggle. An independent local saved-record auditor returns `SAVED_ARTIFACT_AUDIT_PASS`: archive CRC, source bytes/hashes, protocol seal, locked groups, staged receipts, original pilot scores/captions, reconstructed model hash receipts, per-token score means and prefix branch probabilities were checked. No inference or training was run locally. Receipt validation is not independent execution of the checkpoints.

## Main result

Both arms correctly prefer the intended first relation token on all four fit-control events. On the four previously inspected holdout events, both prefer the intended relation token in only two events: the standing-on targets. Both carrying targets still prefer standing over carrying.

| Carrying target after `a person` | CE P(standing) | CE P(carrying) | CROSS P(standing) | CROSS P(carrying) |
| --- | ---: | ---: | ---: | ---: |
| Surfboard, image208095 | 47.54% | 1.52% | 41.62% | 0.25% |
| Skateboard, image70642 | 16.09% | 13.72% | 12.17% | 9.85% |

For CROSS on the carrying skateboard target, `is` is the actual top1 token (13.35%); standing and carrying rank2/3. This leads to `a person is standing with a skateboard`, an insufficiently discriminating description, not the same error as explicitly saying standing **on** the board.

All eight natural greedy outputs per arm are identical whether generated from BOS or continued after `a person`. The models already generate that prefix naturally. Consequently, this equality is largely expected for deterministic autoregressive decoding and **cannot identify or rule out exposure bias**. The common-description-prefix condition duplicates `a person` for all four groups. These are not two independent interventions; a broader claim that fixing the opening fixes generation is unsupported.

## Why phrase ranking hid the error

For the carrying skateboard target, the saved CE score prefers the correct whole phrase by +0.0152, although the first relation word prefers standing: logP(standing)=-1.8272 versus logP(carrying)=-1.9860. CROSS has the same discrepancy: whole-phrase margin+0.0385, first-word logP -2.1065 versus -2.3174.

The ranking score averages log-probabilities over words of complete supplied phrases. It includes easy continuation words, uses different lengths for `standing on` and `carrying`, and excludes EOS. It need not agree with the next-token argmax in unrestricted greedy decoding. Conditional on a supplied carrying phrase, EOS logP is also low (-4.5045 CE/-4.6767 CROSS); EOS is learned by positive CE but absent from the ranking metric. This is an observed surrogate/generation mismatch, not proof that length normalization or EOS alone causes all errors. Simply adding EOS to a metric would not fix the learned model.

## What the interventions do and do not show

- Natural target selection does change probabilities, even in failed holdout images. For example CE P(carrying) in70642 is13.72% for the child carrying the board versus4.42% for the child standing on a board. Thus the evidence path is not entirely ignored or pair-invariant; its response is insufficient for the correct generation decision.
- Evidence-off makes both events in an image receive identical outputs. On fit examples, evidence-on recovers the two intended descriptions. This validates an active conditional path but does not prove image-semantic generalization: training memories, masks, membership and geometry all vary with the target.
- Explicit geometry swap preserves the intended fit captions, but changes scores and some holdout text. Holding the region masks/membership fixed does not remove all geometry, and the swap is inconsistent with real boxes. These findings cannot establish that visual features dominate geometry, or the reverse.
- On208095 the carrying-target first-word odds worsen with CROSS, even though its average phrase margin across the larger holdout improved. There is no evidence to prefer CROSS here.

## Decision

Keep CE as the simpler reference and stop scaling CROSS under this configuration. Do not build a phrase-routing module on the assumption that the current memory is already semantically correct. The favorable branch of the earlier proposal — correct first relation decision under a shared prefix, wrong only in BOS generation — is not observed in these two failure groups.

The next research decision concerns the representation/learning of the selected pair and its interface to the decoder. The current evidence does not isolate encoder failure, decoder failure, global-scene dominance or small-template overfitting. A new proposal should explain and test that distinction instead of adding SCST or another gate by default. No extra experiment or notebook is launched by this result review.

Only two holdout groups and two fit controls, all previously reviewed; GT boxes, one seed, restricted templates, globally contextual CLIP patches. The rates here are descriptive diagnostics, not independent semantic accuracy, full-caption CIDEr/CHAIR gains or a demonstrated new alignment method.
