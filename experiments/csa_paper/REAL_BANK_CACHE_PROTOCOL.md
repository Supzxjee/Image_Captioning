# Frozen CLIP text cache + real-input CSA smoke

02/11/2026. User supplied mount `/kaggle/input/datasets/ducanh2403/csa-real-v126-lemma-fixed`. This confirms the reported mount, not independent local access. No additional lexical audit, architecture, frequency weighting or long training. User runs Kaggle; agent only metadata/NumPy/static/mock locally.

## Input gate

Verify all8original126 payload SHA +2corrected overlay SHA from the reviewed manifests. Require `corrected_assets_receipt.json` to select `bank_words_corrected.json`/SHAcec76a02… and identify source126; compare all original/overlay manifest maps. Validate original+corrected word counts/delta and immutable noun/unrelated-verb policy. Verify NPZ `allow_pickle=False`: featuresFP16[2180,768], boxesFP32[2180,4], scoresFP32[2180], offsetsINT[321], IDsINT[320]; finite, monotonic offsets,1..150regions/image, exact IDs/order/positive dimensions/box bounds/metadata kinds.54token sequences valid BOS/EOS/PAD/range. Groups256TRAIN/64val/0test. This first notebook requires the original regions.npz bytes. If Kaggle expands it, stop and preserve logs; no automatic reconstruction or shape-only/hash-bypass fallback. Never write input mount.

## Frozen text cache

- Official `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. That revision has `pytorch_model.bin`, not safetensors. Official blobSHA `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`, ~599MB. Download only model/config/tokenizer files to temporary cache, not output. Fully verify model bytes before loading, weights_only=True, trust_remote_code=False; require Torch>=2.6. No model download/load locally by agent.
- transformers4.57.1/huggingface-hub0.36.2/tokenizers0.22.1/ftfy6.3.1 in temporary pipless venv with system-site Torch/NumPy; no global Torch/vision/HF mutation. Record package versions and every downloaded file SHA. Full `CLIPModel` loads checkpoint without inventing text-head weights; only use `get_text_features`, no vision forward. Missing/mismatched/error keys fail; only two legacy nontrainable position_ids buffers may be ignored, recorded explicitly.
- eval/frozen/inference_mode/FP32, batch64. Templates unchanged: noun `a photo of a {lemma}`; verb `a photo contains the action of {lemma}`. No caption references enter cache construction. Tokenizer exact revision, max77; do not silently truncate, fail if overlong. Projected512D outputs checked finite/nonzero then L2normalized, saveFP32 arrays plus exact ordered word strings in `semantic_banks.npz`; no pickle/object arrays. Frequencies not model inputs.
- `cache_receipt.json` status SEMANTIC_BANK_CACHE_COMPLETE pins bank file/manifest and all input identities. Keep this cache even if downstream smoke fails. Text cache is the requested deliverable, not a trained caption checkpoint. Unknown visual-cache revision and CLIP-vs-RegionCLIP adaptation remain declared.

## Real-input smoke, no quality evaluation

After successful cache, free CLIP weights/VRAM and run a separate worker. Use four fixed TRAIN images (include a genuine fallback if available) and two fixed val images from original320metadata. New CSA D512/8heads/K150/3layers/vocab10000, same core equations. Two XE updates, batch2, LR8e-6, Adam(.9,.99), clipnorm1, FP32, seed42; TRAIN first reference only. Check finite logits/loss/all gradients, both decoder gates/projection/aggregation connection, bank buffers frozen. Check actual-bank aggregation row-permutation invariance in tolerance. Beam5/max20/raw cumulative score on two val images; val references not fed as prefixes/targets. Save decoded token/caption records explicitly labeled smoke-only; no CIDEr/CHAIR/reward/SCST/DDP or benchmark claim. No caption checkpoint saved. No full15XE+30RL scheduled.

GPU0 only, T4-compatible; no use of second GPU. Installcap900s, cache/downloadcap1800s, smokecap600s are stopping caps, not ETA or billing estimates. Review ZIP contains metadata/logs, excludes binary cache; preserve full output folder for subsequent Dataset save. Separate cache and smoke status prevents recomputing successful embeddings after a smoke failure.

## Sources checked

- [Official revision file list](https://huggingface.co/openai/clip-vit-base-patch16/tree/57c216476eefef5ab752ec549e440a49ae4ae5f3).
- [Official checkpoint metadata/SHA](https://huggingface.co/openai/clip-vit-base-patch16/blob/57c216476eefef5ab752ec549e440a49ae4ae5f3/pytorch_model.bin).
- [Transformers4.57.1 CLIP implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/clip/modeling_clip.py): library API reference, not author CSA code reuse.
