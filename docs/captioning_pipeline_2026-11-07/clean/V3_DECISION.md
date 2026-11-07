# V3 — one operation per node

User explicitly rejected combining unrelated operations into one box. Reinspect the supplied Tam reference and redraw while preserving the actual model. Separate CLIP from fixed banks, separate noun/verb banks and aggregations, patch data/ROI/projection, object/relation Slot MLPs, masked attention/output projection/residual sum, token data/word embedding/position sum, vocabulary projection/softmax/beam search, and three scorer operations/weighted selection. Keep a standard Transformer decoder node as in the reference. Expand Object fusion 2 with separate semantic selection, self-attention, cross-attention and gate; standard norm/FFN internals remain omitted rather than relabeled as a single new operation. No detached mechanism atlas. Original Vo feeds slot building; separate Vo′/Vr′ to decoder. Previous drafts preserved; no model or experiment changes.


## Verification and delivery

Approximate Arial box-width and unique SVG-ID checks PASS. All requested split-operation nodes exist. No connector crosses an operation-box interior. The only collinear edge overlaps are intentional original-Vo branches; corrected a draft overlap of Vo and Vr. Final PNG and Poppler PDF render inspected; PDF one page with selectable text. Downloads Pipeline_tach_tung_buoc_V3_07_11_2026.*. Figures still use ordinary abstractions: standard Transformer decoder and omitted norm/FFN internals; no model was changed or executed. Previous drafts not removed; no experiment registry row.
