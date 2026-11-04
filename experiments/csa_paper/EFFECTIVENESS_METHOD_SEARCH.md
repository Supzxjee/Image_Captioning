# Mechanisms to improve current CSA adaptation

04/11/2026. Literature search and recommendation, NOT a new approved experiment. User explicitly prioritizes CIDEr improvement and lower hallucination, with inherited visual-semantic alignment; novelty is not required by the user. Journal acceptance requirements are not assessed here. No model, notebook or scorer run.

## Recommendation

First candidate for practical caption quality: retain CSA alignment and fine-tune the trained XE model using SCST with a CE anchor and the existing training-only object reward. Compare SCST+CE against SCST+CE+object from the identical CSA checkpoint, batches and budget. This is a proposed comparison, not implementation authorization. Do not add the previously weak word-region loss by default. Coefficients and budget still require a protocol decision. SCST is a second training stage; it need not and should not be interpreted as requiring random initialization.

Evidence boundary: original SCST has caption-metric evidence; CIDEr-only optimization can worsen hallucination. Our historical QFormer full recipe improved TEST raw CIDEr 1.188245 -> 1.212024 and CHAIRs 4.32% -> 3.98%; fixed rerank 1.221815 -> 1.247342 and 3.48% -> 3.44%. That comparison did NOT isolate object reward because its SCST+CE selection fell back to the original model. It does not establish a gain on CSA. See ../../SCST_OBJECT_W05_LOCKED_TEST5000_RESULT.md and ../../SCST_FULL_TRAIN_W05_RESULT.md.

If the user instead wants an additional alignment mechanism, the strongest fit found for a focused feasibility design is cyclical decoding-localization-reconstruction, not another claim that attention mass supervision must work. It needs a genuine shared-decoder reconstruction path and adaptation to CSA's two streams; it cannot be represented as a ready one-layer drop-in. Expected extra training passes must be benchmarked. Existing contextual CLIP/CSA features and linguistic shortcuts can still permit reconstruction without correct localization. No guaranteed CIDEr/CHAIR gain.

## Primary sources inspected and applicability

1. Rennie et al., CVPR2017, Self-Critical Sequence Training for Image Captioning.
   https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html
   Primary indexed abstract/IBM author page inspected; direct CVF page retrieval returned403 in this search. Self-critical reward baseline supports direct sequence-level optimization, not visual grounding by itself.
   https://research.ibm.com/publications/self-critical-sequence-training-for-image-captioning

2. Ma et al., ECCV2020, Learning to Generate Grounded Visual Captions without Localization Supervision.
   https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630341.pdf
   Full searchable PDF method and Tables1-3 inspected. Decode -> word-conditioned localizer -> shared-decoder reconstruction. No word-box annotations required by this method; train-only localizer removed at inference. Flickr30k Entities Table1 averaged5runs: unsupervised baseline CIDEr59.6 -> cyclical61.4, grounding F1all4.08 ->4.98 and F1loc11.83 ->13.53. These are NOT MSCOCO/CSA or CHAIR results. Table3 grounding-biased detector has CIDEr52.4 ->52.0: grounding gains are not universally CIDEr gains.

3. Dai et al., EACL2023, Plausible May Not Be Faithful: Probing Object Hallucination in Vision-Language Pre-training.
   https://aclanthology.org/2023.eacl-main.156/
   https://aclanthology.org/2023.eacl-main.156.pdf
   Full searchable PDF Section6 and findings inspected. ObjMLM masks whole object names/phrases and reconstructs them using image/text inputs. Reported relative CHAIRs reduction17.4% in their VLP setting; not percentage points and not established for CSA. This is token-object alignment, not an explicit word-to-box attention target. Requires a specified auxiliary masked-object task, masking/tokenization and attention policy for our causal decoder. Remaining text can still reveal the object; masking is not proof of visual dependence. Also reports SCST can improve CIDEr while worsening faithfulness; generic global ITC/ITM did not substantially reduce hallucination in their experiments. Patch features are not categorically inferior to detector features.

4. Zhou et al., CVPR2020, More Grounded Image Captioning by Distilling Image-Text Matching Model.
   https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_More_Grounded_Image_Captioning_by_Distilling_Image-Text_Matching_Model_CVPR_2020_paper.html
   Indexed official abstract only; full page/PDF403 here. POS-SCAN teacher regularizes captioner word-region attention. More relevant than generic contrastive alignment, but a trained teacher and verified region-text interface are additional costs. Not selected as the fastest near-term implementation.

5. Huang et al., ICCV2019, Attention on Attention for Image Captioning.
   https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_Attention_on_Attention_for_Image_Captioning_ICCV_2019_paper.html
   Indexed primary abstract only; full page/PDF403. Filters attended information using query-conditioned gating. CSA already has fusion and decoder gates, so benefit on top of CSA is unestablished; not prioritized simply because standalone AoANet improves CIDEr.

## Decision boundary

No guaranteed gains, invented compute estimate, new model execution, training code or notebook. Current recommendation is CSA + staged SCST/CE/object for practical quality; cyclical reconstruction is the additional grounding alternative to discuss if needed. A from-scratch controlled comparison is appropriate for a changed module or auxiliary XE loss; it is not a universal rule for every training mechanism. Existing weak supervision pilots remain evidence, not overwritten by the search.

Academic-search MCP tools named in the skill were unavailable; built-in web search and primary conference/publisher/author sources used. Failed full-text access is explicit above. This is a targeted search, not an exhaustive systematic review.
