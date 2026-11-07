# Reread original CSA-Net figures — 07/11/2026

Source inspected directly: user-supplied `C:/KhoaLuanTotNghiep/Paper/Cascade-Semantic-Prompt-Alignment-Network-for-Image-Captioning.pdf`, 16 pages. Figure 2 on PDF page 4 / printed5269; Figure 3 on PDF page5 / printed5270, and related Method III-A/B. Both pages rendered with Poppler and inspected; no model/scorer execution. Source figure crops are local review artifacts, not committed/re-published in the repository.

## What the source actually does

- Figure2 has a short main path across the top: detector regions, object alignment stage, relation alignment stage, two grounded feature streams into the language generator. It uses compact, aligned operation bars inside module boundaries.
- Two dashed lower frames expand the object/relation semantic alignment in place by short zoom leaders. The object/verb pools, prompt templates and frozen text encoders are located inside their respective semantic frames, close to their consumer. Zoom leaders are not tensor cables.
- Colored feature bars, region thumbnails and representative words explain *what* is aligned, reducing the need for implementation prose. They are conceptual illustrations, not evidence that every shown region-word match was measured.
- The language generator retains Self-Attn / Parallel Cross-Attn / FFN / Softmax. Figure2 does not unpack every gate, residual, mask, checkpoint, cache, preprocessing or training loss.
- Figure3 is a separate depth level: aggregation, a word-region similarity/relevance diagram, then S-MHA / C-MHA fusion. Detailed formulas are in the body text rather than dozens of extra boxes in Figure2.

## Consequence for our drawing

The previous unconditional rule “every operation in the entire implementation must become a box” was too rigid. Preserve one short operation per visible box, but choose a consistent abstraction level and group modules with a real technical meaning. Do not put unrelated sequential steps into a box to save space, and do not flatten all internal mechanics into one enormous wiring diagram.

Our extra image-specific UPT prompt branch and reranking really add complexity compared with the source. They must remain distinguishable from the TRAIN global banks. The original figure is a layout reference, not permission to rename, remove or claim identical components. Main figure: region extraction → bank-based alignment → image-specific prompt conditioning → generation → selection. Local mechanism detail should focus on the region/text alignment; standard details can be shown in a separate mechanism figure or explained in prose. No newly drawn version was authorized by the user's current request to reread the source; this turn records the reading and explains it. V1–V4 are completed artifacts, not accepted final layouts.
