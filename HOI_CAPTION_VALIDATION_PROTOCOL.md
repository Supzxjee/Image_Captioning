# Full Karpathy validation: Q-Former beams + frozen UPT

Run **5,000 validation images on Kaggle T4 x2**, with the original Q-Former 32-query/2-layer epoch-10 checkpoint. No training and no test split in this run.

The implementation compares the same distinct beam-5 candidate pool under three policies:

1. Original rank 0 (accumulated decoder log probability).
2. Historical fixed reranking: decoder 0.3, CLIPScore 0.4, OCC 0.3.
3. That same fixed score plus `lambda * UPT_relation_feature`, with `lambda` chosen from **0, 0.05, 0.1, 0.2** by validation CIDEr. Ties select the smaller lambda; zero is a valid outcome. Baseline weights, parser, pooling, contrast formula and grid are fixed before execution.

## Relation evidence

Use the pinned official UPT-R50 HICO checkpoint and implementation from the [frozen transfer audit](PRETRAINED_HOI_TRANSFER_RESULT.md). Run the detector/interaction model on each whole image. Do not restrict its proposals to labeled source pairs or use human boxes. No ground-truth caption, object label or relation annotation enters inference or claim extraction.

A conservative lexical parser extracts explicit `person -> verb -> COCO object class` claims for hold, ride, sit_on, carry, eat and talk_on. It requires a nearby preceding person mention, rejects explicit negation, attaches the next explicit object before another recognized action/clause, and leaves unsupported constructions unparsed. The rules are in `captioning/hoi_caption_reranking.py`; they are a coverage limitation, not a general semantic parser. Attribute, count, passive-voice and coreference verification are outside this version.

For each trained `(verb, object class)`, keep the highest raw UPT logit over actual detected person/object instance pairs of that class. Captions usually do not identify a unique person instance: this is **existential class-level support**, not proof of unique caption-instance binding. Saved evidence retains the selected subject/object boxes and indices.

For each parsed claim with evidence, compare its logit `z` against the strongest **different trained object class** for the same verb in that image, `z_other`. The contribution is `tanh((z-z_other)/2)`. Average contributions over parsed claims. A missing detection, unsupported object/verb combination, unparsed caption or absent competing class contributes zero. This bounded relative score is not a calibrated correctness probability. It avoids treating a small HICO sigmoid value as a negative merely because it is below 0.5; the source audit showed that correct `sit_on` scores can be much smaller.

No per-candidate minmax normalization is applied to the relation feature. The raw UPT model and inherited decoder/CLIP/OCC scores remain unchanged. The full validation experiment evaluates this new caption scoring rule; earlier 20/20 target ordering and 80-true/16-false acceptance results are not results for this rule.

## Execution and verification

- Two independent GPU processes each own alternating images; no duplicated DDP inference. Both beam generation and UPT inference use the two GPUs. CLIPScore text scoring uses one GPU after beam generation.
- Each image completes an append-only JSONL journal. A resume requires the same protocol/input hashes and ordered shard IDs. Only an incomplete final write can be repaired automatically.
- Merge requires exactly the expected validation ID set with no duplicates and matching COCO IDs/filenames. UPT enrichment must preserve every candidate caption, decoder score, ordering and CLIPScore.
- Original prompt cache: `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt`. Relation V2.1 or E3 caches are not substituted into the Q-Former baseline.
- Original checkpoint: `/kaggle/input/datasets/ducanh2403/qformer-epoch10/qformer_epoch10/model_h1_2_crossattn_epoch_10.pth`.
- Visual cache, MS COCO and detection paths use the existing dataset inputs. No output from a mutable notebook version is required.
- Record SHA256 of checkpoint, prompt cache, dataset JSON, detection cache and UPT weights; record reference/project revisions, visual-shard sizes and all image IDs. Frozen UPT weight SHA256 must match `12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c`.
- Check strict baseline architecture/vocabulary when loading the checkpoint. Backbone weights are unnecessary for Q-Former inference because the existing visual cache is used.

## Metrics and interpretation

Tune only CIDEr on validation; report BLEU-1..4, METEOR, ROUGE-L, CIDEr, CHAIRs, CHAIRi and object Recall for all three policies. Verify grid CIDEr agrees with final scoring. Report CHAIR hallucination removed/introduced transitions relative to rank 0 **and** fixed reranking. These flags do not establish that an entire caption is correct.

Report caption length, explicit object mentions, parsed-action coverage and changed-caption count to reveal gains obtained by dropping content. Convert eval IDs to actual COCO IDs for CHAIR and verify its cached annotations cover every image.

Validation metrics are used for selection and are not an independent test result. Freeze `locked_validation_selection.json` before a separate test run. Historical **test** CIDEr 1.221815 from fixed reranking must not be numerically compared to new **validation** CIDEr as if they were the same split. HICO/COCO pretraining overlap and partial lexical coverage remain limitations.

The compact review archive contains predictions, candidate evidence, metrics, transitions, protocols, journals and logs. It excludes model checkpoints and visual caches. A run is complete only when `run_summary.json` reports `VALIDATION_COMPLETE`; a created notebook or active session is not a completed result.

## Empty-output correction and manual rerun

Version1 completed beam generation and CLIP scoring for all5000 images, then failed in UPT. When no valid human/object pair exists, the pinned interaction head omits attention maps; upstream postprocessing can consequently return an empty image-output list. The wrapper previously indexed `[0]` unconditionally.

The corrected wrapper returns correctly shaped empty relation tensors **only when the captured pair inventory is also empty**. It keeps the image/detections and assigns unknown/neutral relation evidence; it never invents a negative label or drops an image. Missing output with nonempty pairs remains an error. Other model exceptions propagate, and hooks are removed in every path. Workers record pair counts and empty-image status; the final summary reports coverage.

The updated notebook writes to `qformer_upt_val5000_fixed`. The user starts it manually; it has not been resubmitted automatically. If previous completed scored candidates and their protocol are present at the explicit old working directory, it reuses them automatically. Otherwise the user can fill `REUSE_SCORED_CANDIDATES` and `REUSE_SOURCE_PROTOCOL` with the two recovered JSON paths. The runner verifies the source protocol hash, checkpoint/prompt/dataset hashes, full image identities, visual-shard sizes and pinned CLIP revision before reuse. It then starts at UPT instead of recomputing the beams and CLIP scores. Leaving both paths unset with no previous output starts the complete pipeline.
