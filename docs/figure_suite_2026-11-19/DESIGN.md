# Six-figure suite — approved user request, 19 November 2026

Deliver M1, M2, M3, baseline, full expanded pipeline, and full pipeline using M1/M2/M3-only module boxes. One consistent restrained serif/white/thin-line style, concise labels, no floating tensor notation, straight connectors, prior parameter-state icons. Local Downloads folder and ZIP are primary; GitHub is backup. No model/scorer/Kaggle runs or formal report edits.

## Scope and provenance

- M1: spaCy en_core_web_sm extraction/filtering (min count5), noun/verb templates, frozen CLIP ViT-B/16 text encoding, L2 normalization and fixed banks5269/1594 by512. Learned aggregation stays in M2. Follow existing corrected bank and preparation code; templates must not silently change.
- M2: same concise topology as latest approved expanded plan: noun/verb aggregation, two object self-/cross-attention stages then object-guided relation stage, gated fusion, two memories. Norm/FFN/cosine details remain in implementation/module explanation, not removed from model.
- M3: concise existing five-operation object/relation rows, both pools read original Vo, object query/residual Vo and relation query/residual Vr; metadata bypasses text encoder. Standalone can treat the named memory/prompt records as external inputs and omit wires from off-figure preprocessing.
- Baseline: actual `baseline_final_pipeline/baseline/train/patch_model.py`: Linear768/512 then LayerNorm; four decoder layers; final LayerNorm before vocabulary head; sinusoidal positions; beam5. Final pipeline has three parallel-attention decoder layers and its own implementation, not the baseline decoder.
- Full expanded figure includes M1 internals as well as M2/M3; compact source preprocessing, decoder, generation and reranking remain. Modular version keeps other pipeline components but module boxes say only M1/M2/M3.
- Examples are illustrative, not claimed outputs for the photo. Preserve existing versions. Native editable SVG/draw.io and PNG required for each; one six-page draw.io and ZIP can ease delivery.

## Validation

Parse every SVG/draw.io, inspect all six actual renders, check geometry/orthogonality/source identities, and hash copied Downloads assets. Final bundle uses short numbered filenames, a brief README and an index. No new experiment registry row for drawing-only work.
