**Latest V2:** [Decision and QA](V2_DECISION.md) · [SVG](pipeline-clean-v2.svg) · [PNG](pipeline-clean-v2.png) · [PDF](pipeline-clean-v2.pdf). Fusion is expanded at its point of use; no detached detail panel. Earlier files below remain intact.

# Clean redraw — user feedback 07/11/2026

Decision before implementation: replace the boxed, verbose compact figure with a sparse architecture drawing that follows the reference's visual grammar. Remove surrounding section frames and duplicate explanations; reduce operation/data boxes, use small section headings and short edge labels. Keep one shared fusion detail only. Preserve three fusion instances, global bank distinction, image-specific prompt generation/matching, two memory streams and reranking. Keep previous files untouched. Do not run or modify models, notebooks, metrics, report or Sheet.

Source is `build.py`; deliver `pipeline-clean.svg/png/pdf/html`. Frozen/trainable icons refer to learned parameters, not inference-time optimization. Within the combined ROI/projection box the flame refers only to projection. The shared adapter slot-builder creates separate object/relation slots from text, original Vo feature(s), boxes and valid links; it does not merge the two memories. Slot outputs carry their ROI masks. Fusion norm/FFN details and decoder internals are intentionally compressed; the model is unchanged. Crossings without dots are not junctions. There is no all-crossings-free claim.


## Verification

Final PNG and Poppler PDF render inspected; one PDF page with selectable text. Approximate Arial text-width checks, unique SVG IDs and nine required source/consumer connectors PASS. Headings kept clear of prototype wires. No model/scorer/Kaggle execution; no metric or Sheet changes. SVG/PNG/PDF copied to Downloads as Pipeline_clean_07_11_2026.*.
