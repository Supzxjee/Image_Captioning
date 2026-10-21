# UPT hidden-pair caption pilot — A/H, two GPUs per notebook

Prepared21October2026 at the user's request, following Astra's research direction and protocol review. **READY_NOT_RUN.** The assistant creates notebooks, performs pure data/syntax checks and audits returned files; the user runs Kaggle. No claim of improved CIDEr, verified phrase–pair grounding or novelty.

## Run in this order

1. Import [UPT_HIDDEN_CACHE_11K_2GPU.ipynb](UPT_HIDDEN_CACHE_11K_2GPU.ipynb). Internet ON, **T4x2**, **Save&RunAll**. Attach MSCOCO2014 (JSON plus original images). The notebook downloads the pinned official UPT-R50 HICO checkpoint with SHA verification; optionally provide its mounted path at `LOCAL_UPT_CHECKPOINT`. This is not the Q-Former checkpoint.
2. Preserve the **whole** `/kaggle/working/upt_hidden_pairs_11k` folder as a Kaggle dataset, or mount the completed notebook Output containing that folder. Keep both `features_rank*.h5`, both `pairs_rank*.jsonl`, `cohort.json` and `cache_receipt.json` unchanged. Send `upt_hidden_pairs_11k_review.zip` for auditing. The ZIP excludes HDF5; it is not the dataset. Binary arrays are standard HDF5, not ZIP-based PyTorch serialization.
3. Import [UPT_HIDDEN_CAPTION_A_2GPU.ipynb](UPT_HIDDEN_CAPTION_A_2GPU.ipynb) and [UPT_HIDDEN_CAPTION_H_2GPU.ipynb](UPT_HIDDEN_CAPTION_H_2GPU.ipynb). They can run concurrently on two accounts; **each notebook uses both T4 GPUs for its single arm**. Attach the same inputs and edit `HIDDEN_CACHE` to the directory that actually contains `cache_receipt.json`. The default dataset slug is an example until you create it. Share private datasets with the other account or copy them with identical file contents.
4. Return `upt_hidden_caption_A_review.zip` and `upt_hidden_caption_H_review.zip`. Keep each `checkpoint.bin` separately. These checkpoints contain the complete base plus optional hidden branch, with a versioned wrapper; they are not plain baseline checkpoints. No Kaggle API token or automatic notebook launch is used.

Training inputs, both A and H:

| Input | Default mounted path |
|---|---|
| Preserved caption baseline | `/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth` |
| Existing legacy prompt tokens | `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt` |
| Fullframe CLIP cache | `/kaggle/input/datasets/ducanh2403/visual-cache` |
| COCO dataset JSON | `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json` |
| Existing object detections for training reward | `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json` |
| New hidden-pair cache | Your completed `upt_hidden_pairs_11k` dataset folder |

Only extraction needs original images. Training does not download CLIP or re-encode UPT. Neither old full-COCO UPT score cache, UPT prompt JSON, paircrop shared cache nor phrase/crop adapters are inputs to this pilot. The legacy prompt cache remains part of both arms; A is not literally prompt-free.

## Locked design

Same train10k/val1000 cohort as the previous caption pilot; raw cohort identity `0c65467f3907644a8674fe3eff71455d65bde944c30e13a71914617243cf988d`. The dataset JSON, baseline weights, legacy prompt file and reward detections are SHA-pinned. These are reused development images, not a new independent whole-system test. Test references/annotations are not used for selection or rewards.

A continues the preserved baseline. H adds `Linear(1024,512)→LayerNorm(512)` to map each cached UPT pair vector into caption memory. H has525824additional trainable parameters and additional external pretraining information. No new gate, text encoder, geometry embedding, role embedding, LoRA or supervised relation loss. Bbox/identity/label/score are audit/selection metadata, not additional decoder inputs. Training inherits the caption loss/SCST/optimizer loop that completed prior A/B/C/D pilots; only factories for model/loading/prediction and explicit backward-compatible runner options change.

Both: global batch8 (4/rank),1250updates,seed42,checkpoint at end only. First200updates CE, then `CE+.05SCST`, reward `CIDEr-D+.5object_count_v2`. Base LR5e-6, projection LR1e-4. AdamW: betas(.9,.999),eps1e-8,weight_decay.01,amsgradFalse,maximizeFalse,capturableFalse,differentiableFalse,foreachNone,fusedNone; actual optimizer groups are exported. Clip gradient norm1. Model.eval() disables dropout while retaining gradients. Same base initialization and deterministic data schedule; reset sampling RNG42+rank after origin evaluation, before training. Sampled trajectories can diverge after policies diverge.

The inference encoders are frozen. CLIP features are read from the existing cache; UPT is absent from the training graph. Caption CE/SCST gradients reach the projection and the baseline's trainable caption components. No pseudo-verb labels/logits are passed to the decoder or used as reward. Natural captions do not provide hard pair/phrase alignment labels; this is implicit adaptation, not proven grounding.

## Extraction and cache contract

Official UPT/pocket/DETR/HICO code revisions and UPT checkpoint are pinned. Each image runs through **all native detections and interactions**, including min_instances fallback and competitive attention, before limiting memory size. A hook captures input to `interaction_head.box_pair_predictor` (1024dimensions). Classifier output from the same call must equal the raw head logits before sigmoid/prior/FP16; this is exact parity of the same computed tensor, not a cross-backend re-computation. Human-first proposal order and exact native(human,other) row enumeration are checked. Native human-human rows are not fed to caption memory.

Astra's fixed selection policy: human→nonhuman; **both endpoint detection scores strictly >0.2**; sort by decreasing score product, then(s,o,native_row); retain at most8. No verb threshold or reference-dependent choice. If no pair survives, no evidence is fabricated. Weak native proposals still influence full UPT attention before this filter. Detection confidence is not relation truth.

Each rank owns disjoint ordered images and writes one HDF5 plus a JSONL audit. Main arrays per image:

| Array | Shape/dtype | Meaning |
|---|---|---|
| features | 8×1024 FP16 | Selected native pair rows; zero padding |
| valid | 8 bool | True selected slots, false padding |
| pairs / native_rows | 8×2 int16 / 8 int32 | Proposal endpoints / original head row; padded−1 |
| boxes | 8×2×4 FP32 | Subject/object xyxy normalized to original image; native out-of-frame extents retained |
| labels / confidence | 8×2 int16 / FP32 | Native class IDs and detection scores |
| image_ids / dataset_indices | int64 | COCO ID versus Karpathy JSON index, kept distinct |

Record native pair count, human→nonhuman count before confidence filtering, eligible count after filtering, selected count, original/resized dimensions and original JPEG SHA. Feature rows, endpoint order, labels, padding and finiteness are validated. A complete receipt hashes all payload files; training verifies receipt and relevant extraction source policies. FP16 values are read as FP32 without another encoder pass.

Rough vector payload:11000×8×1024×2≈180MB, plus metadata and logs. Original images/UPT checkpoints/code checkouts stay outside Saved Output. Training stages only11000 CLIP/legacy rows in/tmp, requiring several GiB; free-space checks run first. No GPU runtime estimate is claimed before measuring on Kaggle. Saved partial shards are not treated as complete; this first bounded extraction version does not auto-resume partial HDF5.

## Evaluation and decision

Raw greedy original and end captions, identical val1000/reference IDF, CIDEr plus CHAIRs/i and recall. Export per-image CIDEr, predictions, CHAIR numerators/denominators and caption length. H_off removes the auxiliary tokens from memory, not merely zeroing attended values; it is H after adaptation, not A. An all-empty batch uses the unchanged baseline memory; mixed empty rows use attention masks. Pair-swap uses same-split donors with exactly the same valid-token count; donor IDs are saved. Singleton buckets are self donors and are not independent counterfactuals. Swaps measure sensitivity, not correctness.

Numeric gate for further confirmation: H−A CIDEr>=.01 and paired-bootstrap95% lower bound>0; H beats original under the same evaluator; CHAIRs/i do not increase versus A; recall drops by at most.01 on0–1scale. A numeric pass alone does not promote the model. Human review must show supported relational information without increased event/instance/additional-fact errors. No requirement to mention an unreferenced pair. Missing or uncertain facts are not forced negatives.

The32-image human cohort is fixed from cache coverage before any model output: up to16without eligible pairs plus remaining with eligible pairs, deterministic hash filling, excluding568known phrase-audit image IDs. This is a stratified diagnostic sample, not an estimate of development-wide error prevalence or proof of noninferiority. Allow uncertainty; assess object/instance, asserted relations, extra false facts and useful information separately.

Technical failure is invalid, not a negative result. Small/bad valid effects stop the current configuration. Positive but uncertain effects are inconclusive; no automatic extra seed, threshold/loss search or test-based checkpoint selection. No full training without further decision. Success would still need controls for additional encoder/context/capacity; the projection+memory combination is not claimed novel.

## Validation status and logs

Local: five pure NumPy selection/identity/sample tests, Python/notebook syntax and embedded payload/source identities. No local torch/h5py/model/DDP/metric smoke tests or real inference were run. User-run preflights cover hook/raw-logit alignment, HDF5 round-trip, masks, CE/SCST gradients, real attention decoder, and DDP with nonempty/one-rank-empty/both-empty batches. A pass on synthetic tests is not a claim of semantic correctness.

If setup fails before Output exists, run `print(LOG.read_text(encoding='utf-8')[-12000:])` and return the first FAIL/ERROR. Output-level failures preserve logs and a small review ZIP. All source/model downloads use pinned public sources; the notebook does not need KAGGLE_API_TOKEN. Keep the completed cache folder and each checkpoint.bin; the review ZIP intentionally excludes them.
