# CSA mechanism recheck — 10/11/2026

Scope: user asked to revisit CSA after weak CIDEr gains. Read original local 16-page PDF, method §III-B–E (Eq3–19), implementation §IV-B and ablation discussion §IV-D; checked current semantic_patch_model.py and patch_model.py as text. No model/scorer execution, no new training approved. PDF source: C:/KhoaLuanTotNghiep/Paper/Cascade-Semantic-Prompt-Alignment-Network-for-Image-Captioning.pdf, DOI10.1109/TCSVT.2023.3343520. This review is text-based, not a fresh visual/equation-render audit.

## Correction to preceding chat recommendation

CSA does NOT require a separate word-region alignment loss in its stated objective. Its alignment learns through caption XE, followed by CIDEr-D SCST. A direct grounding loss is an alternative proposal, not a missing CSA component or an established remedy for our CIDEr stagnation. Stop treating lack of such a loss as proof of incorrect alignment.

## What original CSA does

- Region features from a detector; default Swin-B + deformable DETR, detector pretrained on four datasets (VG, OpenImages,Objects365,COCO with held-out duplicates excluded).
- Global TRAIN vocabulary pools (4764 nouns,975 relation words), templated and encoded by frozen RegionCLIP text encoder. These prompts are bank concepts, not UPT per-image triplets.
- Learned semantic aggregation into K prototypes (Eq3–5). Cosine prototype-region scores, softmax-weighted pooling across regions, then softmax relevance over prototypes (Eq6–8).
- Deep fusion has visual self-attention, visual-query/text-key-value cross-attention, normalization and adaptive self/cross mixing (Eq9–11).
- Two object alignment layers, then one object-guided relation layer using enhanced object features (Eq12, §IV-B).
- Keep object and relation memories separate. At every decoder layer, two parallel cross-attentions read them and separate gates mix contexts conditioned on the previous word (Eq14–16).
- 15 XE epochs then30 SCST epochs with CIDEr-D; beam5. This does not mean SCST is a separate alignment loss.

## Current patch model differs materially

Current SemanticPatchCaptioner is an adaptation, not complete CSA: fixed CLIP196patches; CLIP-text banks; one object BankMemory and one relation BankMemory; each BankMemory is aggregation/alignment plus residual cross-attention (no complete CSA fusion self-attention/postnorm/FFN stack); additional UPT per-image masked prompt adapters; predecoder mixture into ONE memory, with a per-patch strength gate in latest version; standard four-layer decoder reads one memory. Original CSA uses2object+1relation layers and dual memories with token-dependent decoder fusion.

Historical ROI-based paper-core model was a different experiment with closer CSA structure. Do not say this latest patch implementation retained that complete architecture, and do not transfer current findings to original CSA.

## Interpretation and recommendation (not an approved run)

The latest gate controls how much semantic residual is injected into one memory. It does not restore CSA's two-stage object refinement or per-word access to distinct object/relation contexts. These are verified structural differences; neither is proven to cause the weak gain. Original stronger regional features, RegionCLIP pretraining, detector finetuning and longer training also confound comparison. Directly comparing our10XE120.91 with their final145.2 after15XE+30RL is invalid as an architecture-effect estimate.

If pursuing CSA inheritance, first specify a faithful structural adaptation on the improved patch backbone:2complete object alignment/fusion blocks →1relation block → preserveVo,Vr → parallel gated decoder; retain image-specific prompt adapters if required by our topic. Keep baseline/data/training comparisons explicit. This is more grounded in CSA than adding another free-standing gate or an unvalidated alignment loss. No guarantee of CIDEr improvement; no full run, new loss or experiment budget authorized by this review.
