# Scope: M3, image-specific prompt–visual alignment

Create a single scientific architecture figure for the M3 module of an existing image-captioning pipeline. Explain how linked object/relation prompts update two visual memories before the caption decoder. Do not invent new components, losses, datasets, results, or claims. This module is not itself a caption generator. Do not label it CSA.

## Exact mechanism from implementation

Inputs are the original object memory Vo and relation memory Vr, each 196 x 512, plus cached prompt embeddings and region links. The image-specific object/relation descriptions come from frozen UPT, and were linked to YOLO regions using same-class IoU matching at 0.5 plus ambiguity/collision filtering. Text is encoded by frozen CLIP ViT-B/16; all of this is upstream preparation, outside the focus of M3. Linked ROI weights express fractional overlaps with the 14 x 14 visual patch grid, normalized per region. UPT box geometry remains a separate slot input; do not imply the box coordinates are CLIP-encoded text.

There are two horizontal branches:

OBJECT: pool original Vo using linked object ROI weights; concatenate the 512D object text embedding, 512D pooled appearance, and 4D object geometry; Slot MLP is Linear(1028,512), ReLU, Linear(512,512). The resulting object slots are key/value of masked multi-head cross-attention. Query is original Vo. Add attention delta back to original Vo to produce updated object memory Vo'.

RELATION: pool subject and target features separately, BOTH from original Vo. Concatenate the 512D relation text embedding, 512D subject feature, 512D target feature and 8D pair geometry; Slot MLP is Linear(1544,512), ReLU, Linear(512,512). Relation slots are key/value of masked multi-head cross-attention. Query is original Vr. Add attention delta back to original Vr to produce updated relation memory Vr'.

Attention uses 8 heads. Valid-slot and linked-region support masks restrict each patch query to linked prompt slots. Relation support is the union of subject and target support. If a query has no supported valid slot, its entire delta is zero and its original memory remains unchanged. Both outputs are 196 x 512. The downstream decoder reads both updated memories, but is outside this drawing.

The user wants concise visible labels, with explanations in the caption. Preserve every essential source and arrow, but do not cram dimensions/formulas into each operation. In particular do not draw relation pooling from Vr or updated Vo'.

## Visual specification

Use a restrained, professional paper figure, white background, regular serif font like Times New Roman, black/dark gray thin lines, pale peach for the object row and pale blue for the relation row. No gradients, shadows, 3D blocks, clipart or decorative icons. No giant title, footer, legend or training sentence. No CSA text. No frozen/trainable badges in this trial. All arrows must be straight or right-angle polylines; no curved arrows. No arrows crossing text or unrelated boxes. Align corresponding operations on the same vertical axes with ample consistent gaps.

Use two clear left-to-right rows with five short operation labels each: ROI pooling, Concatenate, Slot MLP, Masked cross-attention, Residual add. Use a small plus-circle for residual addition if it saves space. Show separate compact input data boxes near their actual consumers rather than huge shared input boxes with long perimeter wires. A shared small data box 'Original object memory Vo' may feed both pooling branches via a short left-side bus; put the memory symbol inside the box, not as floating wire text. The relation row also has its own 'Original relation memory Vr' source for attention and residual. Avoid ambiguous shared buses that suggest Vo and Vr are interchangeable.

The object input data plate can say 'Object text + geometry' and contain a tiny illustrative example 'person'. The relation plate can say 'Relation text + pair geometry' with 'person riding bicycle'. These are illustrative text examples, not detector claims about a supplied photograph. Both plates feed Concatenate, not ROI pooling. Show a small 'ROI links + valid slots' data plate feeding the attention masks, with a routing style that cannot be mistaken for text/value features. Q from memory and K,V from slots should be legible; prefer putting short Q/K,V annotations inside the attention box or on its ports, not littering other wires with tensor names. Keep only the needed 196 x 512 dimension in input/output data boxes if space permits.

Output plates should be named 'Updated object memory' and 'Updated relation memory'. Deliver a wide 16:9 diagram, compact enough to read at report width. Keep scientific mechanism exact while simplifying layout. Do not add a Transformer decoder, beam search, reranking, image extraction or semantic bank to this module-only figure.
