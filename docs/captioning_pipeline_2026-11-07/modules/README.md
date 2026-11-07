# Three-module architecture — user-approved scope

Decision: exactly three detailed modules, with identical names in the main overview: M1 Semantic bank construction (offline TRAIN-only); M2 Bank–region alignment (learned aggregation, two object stages and one relation stage); M3 Prompt–region alignment (image-specific UPT encoding, ROI matching, slot construction, masked attention and residual updates). The overview hides internal steps only inside these modules. Model semantics checked against experiments/csa_paper/model.py and prompt_adapter_model.py; no model execution. Prior drafts remain unchanged.

Visual plan: established serif font, thin orthogonal wires, ivory nouns/bank and blue prompt/relation; left→right within each module, labeled inputs/outputs. Standard attention sublayers may use conventional Add & Norm / FFN grouping, with caption identifying abbreviations. No frozen/trainable footer or training note.

## Deliverables

- 00-overall: names of M1/M2/M3 only; identical names match detailed figures. Auxiliary image/YOLO box symbols refer to the same original image/detections. Decoder, beam and reranker remain visible.
- 01-bank: TRAIN captions → POS tagging → lemmatization → CLIP text → two fixed banks. Caption-to-text formatting templates are abstracted.
- 02-bank-region: top panel is the exact cascade, with separate Vo and Vr outputs; lower panel opens one alignment stage. Semantic selection uses soft relevance, not hard labels. The second object stage consumes selected T1, not the raw noun prototypes again. Standard dropout, FFN internal residual/norm and attention skip paths are abbreviated; visible Add & Norm blocks denote residual normalization. The learned gate uses visual-only H and cross-attended features. Selected T is also returned to the next object stage.
- 03-prompt-region: preparation results connect physically to both lanes through one prompt-package data bus (not a new neural layer). Input boxes combine text, original Vo regions and geometry by concatenation; plus symbols there are descriptive, while circled + is actual residual addition. Object/relation lanes have separate weights. Mask only allows linked ROIs/endpoints; unmatched ROIs receive zero delta.

All SVG/PNG and four one-page selectable-text PDFs inspected after final edits. Native SVG text/shapes remain editable. Unique IDs, text widths, orthogonal wires, and no operation-body crossings checked. Same scientific model; no experiment/metric/Sheet changes. Prior drafts preserved. These are review drafts, not user-accepted final figures.

## Label cleanup

User requested M1 title suffix (offline, TRAIN only) and output note removed, M2 two explanatory captions removed, M3 bottom explanatory caption removed. These annotations are removed from the drawings only; scope and architecture remain as documented above. Node positions, arrows and diagram dimensions unchanged. Three affected PDF renders inspected.

## Overview annotation cleanup

User requested small lines removed from the main figure: all free-standing edge labels and explanatory footnotes removed; title, named input/output blocks, M1/M2/M3 and all10 arrows retained. Module detail figures unchanged. Updated PNG/Poppler PDF inspected.


## Week13 annotation edit

Removed exactly the two Only slots linked to... labels in M3 at user request. Mask semantics retained in report prose. Same nodes, geometry and arrows; updated M3 SVG/PNG/PDF and embedded Word image.


## Visual extraction detail 08/11

build_visual.py creates 04-visual-extraction.svg/png/pdf: same image branches to frozen CLIP ViT-B/16 patch features and existing YOLO boxes/classes, both feed ROI pooling, pooled N×768 regions pass through learned768→512 projection to V for M2; B/classes also feed M3. It expands the existing overview block, not a fourth semantic module or new detector execution.10 orthogonal connectors/unique IDs/textwidth/no body transit PASS; PNG/Poppler PDF inspected. YOLO version deliberately unspecified.
