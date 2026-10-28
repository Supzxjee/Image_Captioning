# Fixed32-record visual audit — assistant draft, pending human verification

28/10/2026. No model runtime or GPU. Sample locked before seeing images: four hash-first pairs in each split×stratum cell, giving16pairs/32records/32distinct images. Exact membership and source labels are in `AUDIT32_SELECTION.json`, bound to V2manifest `6ca6af23...` and audit selection `a86f9f84...`.

The assistant inspected all32images via eight contact sheets and examined originals for small/ambiguous cases. Proposed labels: **17supported,13contradicted,2uncertain**; all32S/O boxes were judged to identify the intended visible instance. This is **not human verification or a training approval**.

## Issues requiring review

- **#32 / COCO104510:** source says FALSE for “the teddy bear is touching the cat”. The teddy's extended arm appears to touch the cat's body/tail in the original. Assistant proposes supported, pending human checking. Do not relabel or remove this record automatically.
- **#16 / COCO497579:** a person rides a horse; source says FALSE for “person next to horse”. Ordinary usage may prefer “on”, but riding is not automatically a logical negation of proximity. Assistant marks uncertain pending an explicit interpretation of next-to.
- **#22 / COCO324483:** cat inside an upright suitcase; source says FALSE for “suitcase below cat”. Enclosure and overlap make below ambiguous. Assistant marks uncertain.
- Small objects (#5person, #6airplane) can be recognized in the original but may not remain discriminable at224px. Uniqueness/IoU does not guarantee CLIP can resolve the relation.
- Some categories are figurines/decorations (#20bird, #26elephant). These are not automatically wrong annotations, but the evidence domain should not be described as exclusively real animals.

Of30definite assistant judgments,29agree with the source; two uncertain cases are not counted as disagreements or silently accepted. No accuracy or population noise estimate is claimed from this small, stratified assistant audit. These observations do not yet establish systematic annotation failure or approve the label gate.

## Human review

`review/assistant_review.html` contains the draft. `review/index.html` stays blank, with a separate localStorage key. Both show official original photos, S/O predicted boxes and the source statement/label. Each record requires its own explicit human confirmation; editing an answer resets confirmation. Export remains pending/incomplete unless all32records and a reviewer name are supplied. Every export keeps `training_approved=false`.

The human should especially inspect #16/#22/#32, correct any other errors and export `vsr_frame_audit32_review.json`. Preserve the frozen training manifest while this review is pending. `ASSISTANT_AUDIT32.json` is explicitly `assistant_proposed_not_human`, with zero human confirmations.

Image downloads use the official COCO S3 HTTPS endpoint and preserve native pixels/dimensions. The first COCO hostname failed TLS hostname validation; no verification was disabled. Source hashes/URLs are saved in `AUDIT32_IMAGE_PROVENANCE.json`. Detector cache has no raw RGB hashes, so no byte-parity claim against its historical extraction is possible. Contact-sheet bounding boxes are visual overlays, not regenerated image evidence.

## Cache and next preparation

Read-only HDF5header/ID inspection found all612V2images in the three original CLIPshards (172/229/211), each marked complete with197×768FP16 tokens and the original dataset/preprocessing metadata. No feature values or full36GBshard hashes were read; see `CACHE_COVERAGE.json`.

Original prompt embedding coverage cannot be checked locally because the `.pt` cache is not present. The user-run CPU preparation checks its pinned SHA and every image key, copies exact existing tokens/features and refuses substituted GT-name/VSR-text prompts. It creates no encoder, detector or captioner. Its runtime cache-copy checks remain unexecuted until the user runs Kaggle; metadata and a label-audit draft are not model-performance results.
