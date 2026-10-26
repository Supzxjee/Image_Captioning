# Stale prompt C/T: bounded caption pilot

User authorized packaging for manual Kaggle runs on26 October2026 after Astra's
research decision. Preparation evidence is in `../stale_prompt_preparation/RESULT.md`.
Independent Argon response is still pending; this is explicit in both notebooks.
No model or training was executed locally.

## Run

Use **one notebook per arm, T4×2 each**, Internet ON, Save & Run All:

- `STALE_PROMPT_C_2GPU.ipynb`: matched control.
- `STALE_PROMPT_T_2GPU.ipynb`: same recipe plus0.1 stale loss.

Attach these inputs (edit paths if another account mounts a different slug):

1. Full preparation output `/kaggle/input/notebooks/ducanh2403/stale`.
   Recursive exact-receipt matching finds nested cache roots; reject ambiguous copies.
   Requires every receipted root file, including `aux_rank0.h5`, `aux_rank1.h5`,
   `benchmark_inputs.h5`, `protocol.json`, `manifest.json` and `receipt.json`.
   A review ZIP alone is insufficient. Flat upload outputs and original folders work.
2. Exact A checkpoint `pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`.
3. Original `prompt-cache/prompt_clip_tokens_cache.pt`.
4. Original CLIP `visual-cache`.
5. MSCOCO2014 `dataset_coco.json`.

No original pixels, UPT cache or prompt-text JSON is needed: the audited auxiliary
features already include the four conditions. No CLIP re-encoding takes place.

## Locked recipe

Main TRAIN10k, five references each;1,250 updates/globalbatch8, fixed seed42.
An independently fixed permutation consumes each rank's1,000 auxiliary rows once
in250 batches of4, one per5updates. Neither arm filters on hinge activation.

The **unchanged preparation implementation** of `profile_stale.Objective` is shared:

`C = mainCE + .1KL(A||student) + .1neutralCE + .1visualContrast`.

`T = C + .1 ReLU(s01 − stopgrad(s00) − .1)`.

Main/neutral CE and KL use global valid-token normalization. Last QFormer layer,
prompt projection/MHA/gate/LN and entire decoder train; other weights and A anchor
remain frozen. Eval mode with gradients, FP32, AdamW5e−6,betas(.9,.999),eps1e−8,
weightdecay.01,gradientclip1; no scheduler,AMP,SCST,sweep or interim dev selection.
AdamW decay was not specified during preparation (which had no optimizer); its
standard default.01 is now locked identically for C/T before either runs.

Only end checkpoint saved as legacy-serialized `.bin`. No automatic promotion.

## Integrity, cost and evaluation

Preparation receipt/protocol/manifest and all root payload hashes are verified.
Auxiliary shape,dtype,complete flags,COCO IDs and finite values are checked.
Main data are staged in physical source order into a local `/tmp` HDF5 (~7.1GB;
need~8GiB free). Stage time is logged separately and was **not** measured by the
preparation's~6minute arm projection. The two-hour staging cap is separate from the
two-hour training/end-greedy cap. On failure do not silently resize data/batch/steps.
All2,000 factual auxiliary inputs must exactly match staged original main inputs.

Before training, run dependency/metric smoke, original synthetic gradient tests,
and a two-GPU **synthetic** global-token normalization test. No extra pretrained
model diagnostic/probe is introduced. Frozen state/anchor hashes are checked every
25steps; end tensors must agree across ranks and differ from A. Nonfinite values,
OOM,wrong hashes or measured projected throughput over2hours stop the arm.

Raw greedy A and end on **the same locked dev1000** using the same original inputs
and batchsize16. Standard PTB/CIDEr uses the fixed full dev1000 references (same
policy as prior A/H runs), not batch-local IDF. CHAIR uses pinned annotation cache,
checked COCO IDs and explicit reaggregation of numerators/denominators.

Send `stale_prompt_CT_C_review.zip` and `stale_prompt_CT_T_review.zip`.
Keep both `checkpoint.bin` files and full input cache. ZIPs exclude large stage and
checkpoint files; they include predictions, per-image CIDEr, CHAIR, protocol/logs.

Only after both arms: paired image-bootstrap T−A and T−C. Gate: T−A>=.01CIDEr,
both95%CI lower bounds>0;CHAIRs/i not worse than A;Recall>=A−.005. A single arm's
summary has **no GO decision**. Reused development set/one seed are limitations.

Masks retain silhouettes/context; `object` may be a shortcut; alias probability
can be escaped by synonyms. Caption improvement would justify more investigation,
not establish semantic alignment or novelty without appropriate mechanism checks.
