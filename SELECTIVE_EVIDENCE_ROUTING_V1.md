# Selective instance evidence routing: first implementation

## Question and scope

Can instance-bound verification select useful relation prompts, and can its reliability explicitly bound a causal decoder's use of them? This prototype keeps the current CLIP/Q-Former/Transformer architecture. It does not replace it with GPT-4 or claim attention/gating itself is novel.

The first paper motivates structured object/position context; DFCap motivates adaptive context fusion. Existing component-verification work such as FoCUS (https://arxiv.org/abs/2609.00709) overlaps the broad objective. The testable distinction proposed here is carrying the same ordered-instance evidence and abstention policy through prompt selection and decoding. Novelty remains unverified.

## Implemented

`captioning/evidence_routing.py` contains:

1. `EvidenceRelationVerifier`: frozen subject, object, union CLIP features, their ordered interaction, ten geometric values and predicate-conditioned text. A learned correction retains the frozen union/text cosine as a prior. This is a small learned verifier, not an oracle.
2. Independent binary supervision for predicates. Labels are supported=1, contradicted=0, unknown=-1. Unknowns contribute no classification loss. Simultaneous riding/holding is allowed. No predicate swap or absent caption word becomes an automatic negative.
3. Temperature and selective tails fitted on image-disjoint calibration rows. Source training images are split deterministically 75/25 before training. Evaluation labels never fit the model, temperature or thresholds. The policy requires observed 90% known-label precision and five known selections per tail; insufficient support causes abstention. This is not a statistical precision guarantee.
4. `verified_prompt_memory`: prompt claims and decoder evidence use exactly the same selected instance IDs/boxes/predicates. Textually identical same-class objects remain different instances. Invalid localization must be masked upstream. In the cached development run, localization is conditional on the existing class/IoU matching.
5. `StepwiseEvidenceRouter` and the opt-in `VerifiedEvidenceCaptionDecoder`. Decoder hidden states attend to accepted evidence; a per-token learned gate is bounded by the attention-weighted verification probability. The ordinary-gate ablation uses the same accepted set without that bound. Frozen verifier scores/features receive no caption-loss gradient. The output projection starts at zero, preserving the original decoder at initialization. With no accepted evidence, its residual is exactly zero.

The existing training/evaluation CLI is unchanged. This adapter is deliberately not silently enabled in old checkpoints. The current experiment trains the verifier only; router weights have functional tests, not caption adaptation results.

## Reproduce the cached CPU experiment

Only the already produced `triplet_detector_training_features_v1_review.zip` is needed. No MSCOCO image mount, visual HDF5 copy, model download or GPU is required.

```powershell
python run_evidence_routing_cached.py --source "C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip" --output "C:/KhoaLuanTotNghiep/Code/output/selective_evidence_routing_v1"
```

The source contains 1,141 matched training pairs and 310 development evaluation pairs. The new fit/calibration partitions have 863/278 pairs. These are reused development pools, not an independent benchmark. Known actor-action absence labels are valid for pair negatives, but do not by themselves provide all same-actor/wrong-target hard negatives.

Outputs: verifier checkpoint, policy/temperature, row-wise predictions, structured prompt cache, compact decoder evidence, loss history and summary. Probabilities/decisions are predictions; unknown selected claims remain unverified. No negative-tail decision is exported as a positive prompt.

## Decoder integration API

```python
from captioning.evidence_routing import VerifiedEvidenceCaptionDecoder

# Load the original checkpoint into the original model first.
# Match evidence tensors and adapter to the model device before use.
adapter = VerifiedEvidenceCaptionDecoder(model.decoder, evidence_dim=64).to(device)
memory, padding = model.build_memory(images, prompt_tokens, prompt_mask)
# evidence must come from the matching image's verified_prompt_memory output.
logits, gates = adapter(target_ids, memory, memory_key_padding_mask=padding,
                        evidence=evidence, return_gate=True)
```

Do not also append heuristic direct-relation memory or leave unverified relation prompts in the baseline prompt stream when measuring this route: the verifier would then be bypassed. The paired caption trial must use the same object-only baseline memory and encoder weights in both arms, the same verified evidence slots and candidate budget, and change only reliability conditioning. Freeze the verifier, fit/calibrate it without caption evaluation images, then adapt routing/decoder parameters using train captions. Candidate evaluation must use the adapter at every decoding step; passing it only during teacher forcing is insufficient.

Save adapter weights separately with verifier hash, policy, source checkpoint hash and evidence dimension. They are not interchangeable with old decoder-only checkpoints. The current CLI does not yet implement caption adaptation, beam generation with this adapter, or claim-aware reranking. No lexical cooccurrence heuristic is introduced as semantic supervision.

## Next measured experiment

First establish supported-prompt precision/coverage and same-actor wrong-target behavior on labels not used for calibration. Then compare ordinary gate vs reliability-bound gate with identical input evidence, training and candidate count. Measure actual caption relation errors separately from CHAIR (which measures object hallucination), and report CIDEr/CHAIR only after caption evaluation. Never infer that more changed captions or larger gate values mean better captions.
