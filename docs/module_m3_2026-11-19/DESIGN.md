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

## Revision requested by user: follow Tâm reference layout

The user found the initial figures confusing. Preserve all scientific content and previous drafts; revise presentation only. Borrow the user-provided reference's small operation boxes, white tensor boxes, thin orthogonal connectors and two-row visual-generation path. Put the bank branch above M2, image-specific prompt extraction above M3, visual flow through the middle and generation right-to-left below. M3 detail becomes two vertical adapter columns: pooling, concatenation, MLP, masked cross-attention, residual, output. Keep original Vo for both slot builders and distinct Vo/Vr queries. Main diagram may use module names because details have separate figures. No report/model/training changes.


## User revision: all data notation inside blocks, with concrete examples

Move V/Vo/Vr, prompt vectors, Q/K/V, pooled features, slots and residual names into corresponding data or operation blocks; keep connectors unlabelled. Add larger bank, prompt and beam-candidate data blocks following the actual user-provided Tâm Figure4 extracted from weekly13. Bank words person/bicycle/motorcycle and ride/stand/park are checked against the corrected bank. Image-specific phrases and captions will be labelled illustrative examples, not claimed outputs from a model or this photo. CLIP patch features remain patch tokens, not falsely called per-object tokens. Preserve separate object/relation memories, geometry support, matching threshold and exact model mechanisms. No new inference/training/metrics or report edits.

## User rejected the example-rich version aesthetically

Make one compact overview layout proof before expanding other figures. Use the actual Tâm reference: small normal-weight serif labels, fine dark lines, near-white operations, two restrained example plates with title rules, compact two-row generation flow and short unlabelled orthogonal connectors. Retain tensor names inside blocks, source examples in bank/prompt only, and small freeze/train icons. Do not change model content or present this layout as user approved. Previous M3 drawings remain available, not redrawn in this revision. Prompt/ROI metadata travels together in the prompt record and branches before CLIP text; box links bypass the text encoder.

## Approved: expose M2 and M3 mechanisms within the overview

User agreed to expand both modules in the main figure. M2: separate noun/verb semantic aggregation, two sequential object attention-alignment stages, then relation attention alignment; explicit gated updates; output both Vo and Vr. Attention-alignment blocks summarize existing cosine reweighting and visual self-/cross-attention; standard normalization/FFN details remain in the existing module figure, not falsely removed from the model. M3: two parallel rows with separate pooling, concatenation, Slot MLP, masked attention and residual-add operations. Both pools read original Vo; Q=Vo/Vr and K/V=corresponding slots, validity/spatial masks. Shared named memory/prompt records distribute the required fields; no reference captions enter inference. Keep serif/white/thin-line style, restrained examples, notation inside blocks, compact CLIP/decoder and straight orthogonal connectors. A single crossing may use a straight gap, never a curve or false junction. This changes presentation only, not the implemented model, training, reports or numerical results.

## User revision: only essential box labels

Keep the expanded M2/M3 topology and all connections. Prune implementation prose, Q/K/V assignments, gate/concatenation/MLP formulas, Output/Contexts lines and repeated tensor sizes from processing boxes. Retain operation names (aggregation, object/relation attention, gated fusion; pooling, concatenate, Slot MLP, masked cross-attention, residual add), branch identity, original-Vo pooling, essential tensor identities in data boxes, model names and beam5. Bank and prompt examples remain brief as requested earlier. Shorten process boxes and remap edge endpoints; do not reintroduce floating symbols or change model computations. No report/model/training changes.
