# Compact connected pipeline — 07/11/2026

## Decision before drawing

User approved replacing the eight-panel technical atlas with one connected figure, following the supplied reference's arrangement. Preserve the actual model; keep TRAIN banks above, image/visual flow at left, image-specific prompts below, two adapters and caption selection at right. Only semantic fusion receives a shared mechanism inset. Show Slot MLP, masked cross-attention, and residual inside each adapter. All prompt/matching outputs must connect to their consumers. No experiments, model execution, metrics or registry changes.

Inputs: `experiments/csa_paper/model.py`, `prompt_adapter_model.py`, and the previously verified training/reranking configuration. Clean example image is COCO TRAIN263974 from the existing repository; no invented boxes on the input image. Detector version remains unspecified.

## Deliverables

`pipeline-compact.svg`, `.png`, `.pdf`, `.html`; source `build.py`. Rounded boxes are operations; square boxes are data. Blue snowflake = frozen, orange flame = trainable. Orange and cyan indicate object/relation semantics. The shared fusion detail is a zoom, not another forward stage; the three fusion instances have separate parameters.

The figure deliberately omits per-operation norm/residual wiring in the shared fusion detail, numerical dimensions inside slot MLP, exact aggregation equations, decoder internals and training loss graph. Those are implementation details in the technical atlas, not changes to the model. Both slot builders use original Vo. TRAIN banks are prepared once, but aggregation parameters are learned. Inference uses no reference captions. PDF/PNG rendering and checks are recorded in `qa.json` after inspection.


## Verification completed

SVG unique IDs, 11 required data-flow connectors, one clean photo, absent banned method names and approximate Arial box-text widths checked. Final PNG and Poppler rendering of the one-page PDF visually inspected. Selectable PDF text verified with pypdf. This is not a claim of no wire crossings: crossings without a junction are separate streams. Source and outputs copied to Downloads. No model, scorer or notebook executed. No result sheet row needed for a diagram-only task.
