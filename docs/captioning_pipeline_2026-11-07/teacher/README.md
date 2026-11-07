# Teacher-facing overview — 07/11/2026

User requests a compact, understandable presentation figure after rejecting implementation-level V1–V4. Use academic-diagram: DFCap for grouped mechanism boundaries and local arrows, LoftUp for feature stacks and a short main axis. Do not copy either method. Main takeaway: region features are enriched by TRAIN semantic banks and image-specific, region-linked UPT prompts, then used to generate/select a caption.

Use a consistent overview level. Keep separate meaningful steps, not multiple operation lists in one box; omit standard implementation internals instead of flattening them into dozens of nodes. Show CLIP/YOLO, ROI/projection, two bank aggregation branches, two object stages as a stack, relation alignment, two masked attention updates, slot building, decoder, beam, parallel scores and selection. Standard Norm/FFN, precise slot gather/indexing, masks, positional encoding, vocabulary projection and softmax are not expanded. Two small Linear nodes are the actual output projections before prompt residual addition. A repeated Vo feature stack denotes the original object memory used by both slot MLPs, not Vr or adapted Vo′. Beam returns five candidates; scoring uses no references/CHAIR.

Single editable SVG is the main source, PNG preview and one-page PDF are exports. Real photo is the existing clean COCO TRAIN263974 example; all token stacks are schematic, not measured attention maps. Retain prior versions; do not modify models, metrics, reports or Sheet. Only the user's own Kaggle notebooks run models.


## Delivery and presentation notes

- [Editable SVG](pipeline-teacher.svg), [PDF](pipeline-teacher.pdf), [PNG](pipeline-teacher.png), [1440px slide preview](slide-preview.png). Downloads: Pipeline_trinh_bay_voi_thay_07_11_2026.*.
- Source: `build.py`. SVG shapes/text/connectors are vector; original photo is raster.
- References: supplied DFCap paper Figure2, PDFp7, for local grouping/QKV/gating clarity; academic-diagram's original LoftUp Figure3 reference for feature-stack notation. Borrow visual relationships only; this model has no DFCap retrieval or bidirectional DF module.
- Checked approximate Arial box widths, unique IDs, one clean photo, 14 necessary routes, no operation-body transit. Final SVG PNG, Poppler PDF and 1440px slide preview visually inspected; private DFCap side-by-side QA kept outside Git. PDF one page, text selectable. No zero-crossing or native PowerPoint claim.

### Explain the figure to the supervisor

1. **Regions:** CLIP supplies image features; YOLO supplies boxes. ROI pooling and projection produce one vector per retained region.
2. **Alignment:** TRAIN noun/verb banks supply global semantics. Object alignment uses two cascaded learned stages; relation alignment follows it. UPT supplies image-specific object/relation prompts. The frozen text encoder creates their embeddings; class/IoU matching links them to YOLO regions. Separate slot MLPs use text, original Vo feature(s) and geometry. Masked attention updates object/relation memories, followed by learned output projection and residual addition.
3. **Caption:** the decoder reads both updated memories, beam search returns five captions. Fixed decoder/CLIP/object-consistency scores select the output. No test references or CHAIR are used for selection.

Object/slot stacks mean repeated structure with **separate parameters**, not shared weights. The isolated Vo feature stack is the same original object memory, never adapted Vo′. Learned attention is a mechanism for combining signals, not a guarantee that each attended region is correct. Current training is XE then SCST+CE; no extra object-reward loss is added by this figure.
