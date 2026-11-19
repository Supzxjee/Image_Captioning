# M3 and connected overview — 19 November 2026

User requested M3, followed by the overall connection of M1/M2/M3. Keep the approved palette and individual operations, straight/orthogonal arrows, no CSA name, no training/frozen legend or training footer. Deliver local SVG/PNG and native draw.io files; do not modify reports or execute models.

## Scientific scope

- Current implemented model is `baseline_final_pipeline/final_pipeline/xe/csa_patch_model.py`, with `PromptMemory` in `semantic_patch_model.py`, cache interpretation in `semantic_data.py`, and core encoder/decoder in `bank_layers.py`.
- M3 preparation uses image-specific UPT phrases and class/IoU matching with YOLO boxes (IoU >= 0.5). Frozen CLIP ViT-B/16 text features are L2 normalized.
- Box weights are normalized fractional patch overlaps. Objects use matched YOLO ROI boxes for pooling/support and original UPT box coordinates for the 4-D geometry input. Relations use two pooled features from ORIGINAL M2 object memory Vo and 8-D geometry.
- Object input [text512, pooled512, geometry4] -> MLP1028/512/512. Relation input [text512, subject512, target512, geometry8] -> MLP1544/512/512. ReLU between linear layers. Up to 8 object and 3 relation slots.
- Object attention Q=Vo; relation attention Q=Vr. K/V=their respective slots. Only spatially linked, valid slots are allowed. Residuals yield Vo'/Vr'; no supported slot gives an unchanged patch.
- M2/decoder stay as in implementation: CLIP patch196x768, linear projection768/512, two memory streams196x512, three decoder layers with parallel object/relation cross-attention and gated fusion. Do not import baseline's four-layer decoder or final LayerNorm.
- Overall beam5 and fixed 0.3 decoder/0.4 CLIP/0.3 detector-based object score. Detector support is a weak consistency signal, not proof of visual absence or a verified hallucination detector.
- M1 remains the existing TRAIN-only noun/verb bank construction, not image-specific prompts. M1 and M2 diagrams remain untouched.

## Verification plan

Parse SVG/draw.io XML, inspect actual Chrome renders of M3 and overview, check arrow endpoints and original-Vo versus adapted-Vo semantics, and deliver in Downloads. No Torch/Kaggle/model runs.
