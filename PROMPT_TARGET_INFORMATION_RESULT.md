# Target information audit after the first review batch

The selected-prompt serializer produces identical text for 21 of the 34 eligible human-reviewed target contrasts. Keeping the selected subject/object IDs and normalized boxes distinguishes all 34. This is a representation check on training cases, not evidence that a new caption method improves metrics.

## Executed check

`audit_prompt_target_information.py` calls the real `build_selected_prompts` serializer twice per reviewed case. The subject, predicate, selection score and threshold are fixed; only the target changes. Artificial selection scores force the single claim through serialization and are not model predictions or an accuracy experiment.

| Check | Result |
|---|---:|
| Eligible reviewed contrasts | 34 |
| Same-class alternatives | 21 |
| Identical resulting prompt text | 21 / 34 |
| Distinct structured grounded payloads | 34 / 34 |

Example: selecting either of two bicycles produces `a photo showing a person riding a bicycle`. A deterministic CLIP text encoder receiving the same text and tokenizer settings cannot recover which bicycle was selected. This does not imply identical visual grounding: the visual branch can still distinguish boxes.

Two tests pass: a same-class swap keeps the text identical but changes the grounded payload, and a cross-class swap changes the text. Actual per-case payloads remain local; public evidence contains aggregate counts and the filtered-source hash only.

## Code path and integration boundary

The selected JSON already preserves IDs and boxes in `objects` and `relations`; information has not been lost from that file. However, `embed_relation_prompts.py` reads only `entry['prompt']` and produces text tokens/masks. That route discards instance metadata from its tensor cache.

The caption model's `Encoder.forward` in `captioning/models.py` supports a separate relation entry. `captioning/inference.py` passes its text tokens, subject/object boxes, confidences and mask to the pair-grounded alignment module. Thus existing E3 grounding is not text-only and is not shown to be broken by this audit.

The new selected JSON is not directly compatible with `build_grounded_relation_cache.py`: selection emits `relations` with `subject_instance`/`object_instance` and normalized boxes, while the older builder reads `triplets` with `subject_id`/`object_id` and divides pixel boxes by image size. Simply feeding the selected JSON into that builder can yield empty relation lists; renaming fields alone would double-normalize boxes. Its counterfactual map also lacks sitting_on/eating/talking_on. No such unsupported conversion was run or silently patched here.

The audit exports an explicit language-plus-ID-plus-normalized-box interchange contract. It is NOT an executable tensor cache, and does not change the trained caption pipeline. Model logits/margins are not relabelled as calibrated confidence; a cache bridge must define that policy and counterfactual semantics explicitly.

## Decision

Pause further manual annotation and target-loss tuning. First test caption dependence on verified relations, using a fixed caption checkpoint and a small fixed image subset: compare the existing relation input, a zero relation mask, and a target-box swap with identical relation text. Keep visual tokens, generic prompt tokens, decoding policy and candidate count fixed. Record accepted relation masks/weights, memory change, caption candidates and relation errors. All-zero/unchanged results must be reported, not interpreted as an improvement. Wrong-box swaps are interventions, not verified negative training labels.

The full caption checkpoint and its compatible visual/prompt/grounded caches are not available locally. No actual decoder intervention, CIDEr/BLEU/CHAIR evaluation or causal bottleneck conclusion is reported. This audit identifies an integration risk and a falsifiable next test; it does not prove that serialization caused the prior pilot's poor transfer.

## Reproduce

`python audit_prompt_target_information.py --package <original annotation directory> --filtered <human_batch1_filtered.json> --output <new output directory>`

`python -m unittest tests.test_prompt_target_information`

Local output: `C:/KhoaLuanTotNghiep/Code/output/prompt_target_information_audit_v1`.
