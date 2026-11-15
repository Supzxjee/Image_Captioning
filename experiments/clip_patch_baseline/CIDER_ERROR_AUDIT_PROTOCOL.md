# VALID caption error audit

Use the completed SCST1500 v94 CPU recovery ZIP. Join scores, selected ranks, image metadata and all five beams by COCO ID and verify the selected captions against the beams. Rank by saved final CIDEr with COCO ID as the tie-breaker. Sample 30 images uniformly from the lowest decile and 10 from the remaining 4,500 without replacement using seed 1500. Shuffle the review order. The comparison group is drawn from outside the low-score group.

Review the final caption against the image first, without scores, group, references or reranker details. Then review the alternatives in shuffled order and identify any clearly better caption. Keep uncertainty explicit. Low CIDEr is a sampling criterion; correctness comes from image review. The design is descriptive and cannot estimate population error prevalence or establish metric gains.

No new decoding, training or TEST. The outcome will guide the choice between studying errors in the candidate generator and studying errors in the selector.

## Returned review and findings

The user returned a 40-case export on 2026-11-15 (export SHA-256 `b11c0db2aeba1ca4cf46cfe4f544f83ea22727206b7bd6d708eee93044395d58`). All three fields on all 40 cases match the saved assisted labels. The returned export records no edits and does not record which cases were independently checked.

| Primary judgment | Bottom decile sample | Comparison sample |
| --- | ---: | ---: |
| Correct and adequate | 9 | 4 |
| Missing salient content | 9 | 1 |
| Incorrect or added object | 6 | 0 |
| Incorrect action or relation | 2 | 2 |
| Attribute or count | 1 | 1 |
| Language | 3 | 1 |
| Unclear | 0 | 1 |

Among the 21 low-score cases with an identified error, 8 had a clearly better saved alternative and 13 did not. In the comparison group the corresponding counts were 3 and 2 among 5 identified errors. The alternative can be an improvement while still incomplete. The groups were sampled by score and the comparison sample is small.

Suggested next check: for the 15 low-score cases with missing content or wrong objects, trace the visually verified object through detector output, prompt/ROI linkage, and all five beams. Use this to distinguish missing upstream evidence from available evidence the generator fails to verbalize. No model run or modification in this analysis.

## Fifteen-case evidence trace

Reuse the 15 low-score cases with a primary missing-content or wrong-object judgment. Inspect the exact v94 saved captions and reranker detection labels. A class label is evidence of a detection at image level, with instance identity and fitted boxes still to verify. Join the exact v175 prompt-link rows by VALID COCO ID. Absence of a linked slot does not imply absence of a visual ROI. No inference or training.

### Prompt–ROI join complete

[Extracted v175 rows](CIDER_ERROR_VALID15_PROMPT_LINKS.json); returned export SHA-256 `3d9f227b28123970db43c756850da5fbbd8ee0328cdb2581895c3032ba3bf75d`. The declared source digest matches the v175 pin. All 15 VALID IDs, object ROI bounds and relation endpoints verified. Fourteen images have no retained relation. The only two retained phrases are on image 29358 and refer to a donut, whose identity remains uncertain by inspection.

- 239894 and 487438: teddy bear and bed appear in saved reranker detection labels respectively, but the matching prompt classes are absent. The export contains retained links, so it cannot distinguish missing UPT proposals from filtered links.
- 436349: bus and truck both have slots; the review favors bus, while all five beams use truck.
- 82174: two bird slots and two chair slots are present; none of the five beams expresses both pairs. Slot counts alone do not prove correct instance or relation use.
- 72024 and 468858: boat and couch slots are present. A saved alternative mentions the omitted object, while the selected caption leaves it out.

These are selected error cases. The metadata localizes failure modes without measuring causal prompt effects. Proposed next check: inspect raw UPT proposals and YOLO ROI rows on the teddy bear and bed cases, then test adding one verified object–ROI link at a time on the same frozen checkpoint if a usable ROI exists. Keep other inputs fixed and use the saved original beam/caption baseline. No new inference or training performed.

### Two-case source check approved

Read the exact detector and UPT prompt caches that fed v175. Export the source boxes and class evidence for COCO 239894 (teddy bear) and 487438 (bed), reconstructing detector ROI order with the historical confidence sort, clipping and cap. Compare the count and all retained v175 object ROI boxes. The frozen UPT cache exposes previously selected proposals, so an absent class can be attributed to absence in that cache, with raw UPT model output still unknown. No model execution or prompt injection in this phase.

### Two-case source check complete

[Export](CIDER_ERROR_TWO_SOURCE_BOXES_RESULT.json). Exact pinned source IDs and the two retained v175 records match. In both cases ROI 1 is a usable detector box by visual overlay. For COCO 239894, teddy bear confidence is 0.9260 and the cached UPT object list contains only person. For 487438, bed confidence is 0.8526 and the cached UPT list has suitcase and backpacks. The target class is absent from both cached UPT lists, so there is no same-class UPT–YOLO IoU to compute. Raw UPT output before cache selection is unknown. Proposed frozen two-image check: append only the verified detector-derived noun slot and compare the five original versus modified candidates. No inference has run for that intervention.

### Frozen single-slot intervention approved

Treat these two selected VALID images as a mechanism check. Replay exact v94 with pinned v175 links, embeddings and VALID patches. Append one noun prompt to the next empty object slot, using the verified YOLO rectangle for both normalized box and patch support. Keep every other packed slot, decoder, beam width and max length fixed. Compare all five captions and report target noun mentions, beam rank and any baseline replay differences. Historical reranker output is for context only. No reranking, training, reference CIDEr or TEST.

### Frozen single-slot intervention result

[Result](CIDER_OBJECT_SLOT_VALID2_RESULT.json). User ZIP SHA-256 `43cfe472185bb84099df7ed2faca42424d1e538734a6439adf33f6112c5eeff5`. Offline verification confirms both cases, checkpoint identity recorded in the export, verified ROI coordinates, and exact replay of the five archived beams per case. The added slot changes the five-beam list in both images and the top beam only for 239894. Target noun mentions remain 0/5 before and after on each image. The teddy bear case still says bag; the bed case keeps the same top beam and introduces bookshelf in one beam. This two-image probe does not estimate CIDEr, CHAIR, reranking utility or the direction of the target noun score. Defer a full fallback run. A follow-up, if justified, would compare teacher-forced target phrase margins under a fixed caption prefix using the same model and inputs.

### Fixed-prefix score check approved

Reuse both v94 arms from the single-slot probe. On COCO 239894, score teddy bear versus bag after "a woman with a red". On COCO 487438, score bed versus chair after "a room with a". Teacher-force each continuation and report token log probabilities, their sum, and the shift in the correct-minus-confusion margin. Also report per-token mean margin because the teddy-bear alternatives differ in length. Neither score alone measures general caption quality. No new search, reranking, metric or training.

### Fixed-prefix score result

[Result](CIDER_OBJECT_PHRASE_VALID2_RESULT.json). Verified both exact prefixes, checkpoint identity, token counts, and all aggregates offline. Correct-minus-foil total log probability margins: teddy bear vs bag -5.6344 to -2.0023 (+3.6321); bed vs chair -2.8530 to -2.0501 (+0.8029). Both target scores increase. In the teddy case, mean log probability per token margin turns positive (+0.1214), while the total margin remains negative. Prior five-beam decoding still had zero target mentions in both cases. This selected two-image, fixed-prefix probe establishes sensitivity to the added text-and-region slot, with regional specificity, generalization and corpus-level gains untested. Proposed next check: compare matched ROI and wrong ROI while holding the added noun text fixed, using both supportive and confusable cases before modifying decoding or training.

### Matched-region control approved

Run the same v94 fixed-prefix check in three arms: original, one added prompt at the verified object box, and the same added text at a visually checked distractor box. Use handbag ROI 6 for COCO 239894 and chair ROI 7 for COCO 487438. These boxes are disjoint from the corresponding target boxes and unused by the original prompt slots. The primary readout is matched minus distractor in the target-minus-foil summed log-probability margin. Also return both arms relative to original. Keep slot identity, text embedding, masks and other slots fixed. The box changes both absolute geometry and patch support, and the distractors are visually different in size and position. This measures sensitivity to the linked ROI for two cases. No decoding search, training or TEST.

### Matched-region control result

[Result](CIDER_OBJECT_ROI_CONTROL_VALID2_RESULT.json). Both baseline and matched scores replay the earlier result exactly. Matched minus wrong ROI in total target-minus-foil margin is +3.5831 (teddy bear) and +0.8153 (bed). Wrong ROI minus original is +0.0489 and -0.0123 respectively. Correct linkage yields a higher margin for both selected images, while the correct noun remains below the foil in total log probability. The ROI intervention changes both geometry and patch features. Recommend testing on a broader selected VALID cohort with comparable distractor size and position before modifying decoding or training. No corpus metric estimate.

### Broader VALID matched-region preparation

Exclude the 40 already reviewed images. Reuse v94 archived VALID beams. Screen for a YOLO class with confidence at least .50 absent from all five saved captions and from existing v175 object prompts. Require a distinct different-class detector ROI unused by the existing slots, around an object mentioned in the archived final caption. Compare the same target prompt at two boxes. Keep box area ratio at most 2, absolute normalized center distance at most .35, and IoU at most .10. Candidate boxes must be visually confirmed before scoring. Sample one pair per image, prioritize similar geometry, and limit repeated target classes in the review shortlist. Aim for 20–30 visually accepted cases. Two prior examples remain outside the cohort.


### Broader VALID screening result

912 images screened; 48 eligible pairs and 26 unique shortlisted images. Verified pair geometry and all JPEG hashes. Visual correspondence and caption salience remain to be judged before scoring. If fewer than 20 pass, report the yield before considering new cases. [Candidates](CIDER_ROI_COHORT_CANDIDATES.json).


### Broader VALID visual review

Assistant judged all 26 image overlays: 4 accept, 20 reject, 2 unclear. Most rejections are background/minor details or object mismatch. The proposed 20–30 case probe is not ready. User review pending. [Labels](CIDER_ROI_COHORT_ASSISTED_LABELS.json).

### Reference-based candidate screening approved

Before checking geometry, retain only target labels mentioned in at least two separate saved VALID reference captions. The reference payload SHA-256 is `219cbe174a62e04b75c4f110242e12c51edb32a4c42210713db93b330340e75e`. This retains 141 of the earlier 912 images. Preserve the existing detector, ROI and prompt pins, foil eligibility and geometry thresholds. Report mention counts for shortlisted nouns. References select the diagnostic sample offline. Visual checks and later scoring must use the verified image/ROI, with references excluded from model inputs. No GPU in screening.

### Reference-based screening result

[Result](CIDER_ROI_REFERENCE_COHORT_RESULT.json). The 141-image reference-first pool gives two eligible pairs, both exact repeats of candidates rejected in assisted visual review: COCO 456969 (the proposed pizza box covers salad) and COCO 22420 (the motorcycle is printed on a book). No accepted pair under this filter. Do not use these pairs to score the decoder. The reference mention criterion applies to the whole image and does not validate a particular detector box. If this experiment is revisited, visually verify target boxes first and consider same-size within-image controls without requiring another detector label mentioned in the saved caption.

### RegionCLIP fixed-box feasibility

Use the official pretrained RegionCLIP RN50 recognition backbone at source commit 4b8513b56e24827e3d6468e1f2105869f35c2d0b. Freeze the source. Start with two previously verified VALID images and their four exact target/control YOLO boxes. Use the official recognition preprocessing, C4 ROI pooling, layer4 and attention pooling. Do not generate boxes. Record original normalized boxes and features with checkpoint digest. This is a technical extraction smoke only; features in different spaces must not be substituted into v94 without training/projection. If cache extraction works, design a matched trained comparison of feature sources on the same boxes.
