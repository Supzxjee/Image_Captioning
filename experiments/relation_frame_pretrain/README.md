# RelPos / RelContrast — fixed frame pretraining

User-run only. No local Torch/model execution. The completed v105 cache and separate human audit are prerequisites; cache prep's old pending-audit flag is superseded by the completed audit, not edited retrospectively.

## Run

Import one notebook per account, choose T4 ×2, Internet ON, Save & Run All:

- `PRETRAIN_RELPOS_2GPU.ipynb`: positive statement reconstruction only.
- `PRETRAIN_RELCONTRAST_2GPU.ipynb`: same reconstruction + .5 observed-predicate truth BCE + .5 same-statement image-pair softplus loss.

Add Input `ducanh2403/relation-frames-v2-cache-v105` and the dataset holding A at `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`. Cache path is prefilled exactly as confirmed by user. No visual-cache/prompt/UPT/COCO-image dataset or API secret is needed for this phase. Full SHA guards reject substitutions. Original cached prompt arrays stay untouched and are not consumed by frame pretraining.

Endpoint200 only, seed42, global batch16 assertion pairs (8 per rank), AdamW1e-4/weight decay.01, gradient clip1. This is12.5 passes over256training pairs. Both arms process the same ordered positive/negative images with the same model initialization; RelPos does not train on negative-image losses. Holdout64pairs128uniqueimages, 32/32strata; no checkpoint selection on holdout.

Each notebook caps GPU phase (including preflight/evaluation) at .5 device-GPU-hour, equivalent to15wall minutes on two GPUs; not a runtime prediction. Total two-arm cap1device-GPU-hour. CPU hashing/downloading is outside this measured phase and Kaggle's own accounting may differ. If incomplete, send review; do not change budget/test tolerances silently. Current remaining account quota has not been reverified.

## Implementation choices

Frozen raw CLIP grid14×14×768, CLS excluded. Normalize predicted boxes by original width/height under the original direct224×224resize; use fractional patch intersection area, so small valid boxes need no nearest-token fallback. Shared entity projection256; ordered S/O role-area channels in union attention keys/values; union rectangle with log fractional-area attention prior. Signed8Dgeometry and80class embeddings32D. MLP840→512→512 + LayerNorm gives frame512. No GT boxes/classes are model inputs; manifest GT exists only as verification lineage.

GRUhidden256 receives frozen Aword embeddings512 and frame-conditioned initial hidden state; a trainable Linear256→10000 head, identical between arms. No decoder/Aforward or Aoutput-head transplantation. Positive loss is mean of three lexical-span meanCEs (subject, predicate, object), excluding articles/copula/EOS from direct loss. No negative sentenceCE. Only the annotated predicate logit gets truth supervision; other predicates remain unknown.

Local static/metadata checks validate all320frozen pairs and1280predicted referent bindings; no neural execution. Kaggle preflight tests actual frozen embedding/vocab, finite logits/losses, batch-vs-single tolerance1e-5absolute/relative, gradient paths/unknown-head-zero, frozen state, identical rank initialization and optimizer update, then restores the exact initialization and fresh optimizer before train. FP32 only; deterministic kernels and no TF32/AMP. No toy model check replaces the real preflight.

## Decision

Both arms score each heldout statement on I+ and I− with the **same lexical-normalized GRU log likelihood**, not an untrained RelPos truth head. Ties count incorrect and are reported. CPU comparison (`compare_frames.py`) uses10,000paired bootstrap draws, fixed seed20261028, separately resampling32pairs in each stratum.

Pass requires RelContrast accuracy≥.70, improvement overRelPos≥.10, paired95%CI lower(diff)>0. Only then implement caption controls Tune/RelPos/RelContrast. Failing a numerical/budget check is incomplete, not a semantic result. Passing pretraining is not proof of caption improvement, independent generalization, or paper novelty. VSRimages are adaptation-heldout, potentially seen by A; the human audit was assisted, not independent gold.

Send two small reviewZIPs; preserve both checkpoint.bin. The code never starts a second phase or creates a Dataset automatically.
