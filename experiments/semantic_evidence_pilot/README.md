# Semantic evidence pilot: two notebooks, two GPUs per notebook

Completed: [A/E results and artifact audits](AE_RESULT.md). E has not improved the matched control; do not scale this version to full training. Original notebooks remain unchanged for reproducibility.

## What to run

- `SEMANTIC_EVIDENCE_A_2GPU.ipynb`: matched continued-training control, no new UPT branch.
- `SEMANTIC_EVIDENCE_E_2GPU.ipynb`: text-query image patch attention plus a pseudo-pair alignment loss.

Run A on one account and E on the other, each selecting **GPU T4 x2**, Internet on, **Save & Run All**. The notebooks are self-contained; all new Python source is embedded. This is not two different arms sharing two GPUs inside one notebook. No notebook is launched by the authoring scripts.

Both start from the same best preserved checkpoint, NOT from completed pilot A/B/C/D. Keep the legacy prompt cache and new UPT JSON separately. No new dataset is required. The six existing inputs are checkpoint `qformer-scst-object-w05-legacy`, legacy `prompt-cache`, `upt-prompts-full-all-actions-v2`, `visual-cache`, MSCOCO JSON and `objectdetectionecache`. Copy the actual mounted paths into the first code cell if your friend's account mounts them differently; private datasets must be accessible to that account.

The exact paths and expected SHA256 identities are embedded in both notebooks. No raw UPT cache, COCO images or V-COCO data is required. If the MSCOCO dataset is attached, Kaggle may prepare the whole input before the first print.

## Fixed protocol

Same deterministic Karpathy train10k/val1000 cohort as the preceding pilot, seed42; global batch8 = 4 per GPU, 1250 synchronized updates. Base LR5e-6, new branch LR1e-4. First200 updates CE, next1050 CE+.05SCST, with unchanged CIDEr-D+.5object_count_v2 reward and full TRAIN IDF. E adds .1 times alignment loss from the first update. All five training references per image are retained. No test images or held-out reference targets enter the loss. New branch and base train jointly; no additional frozen-base arm is included.

Compare E against the newly run A, not only against the starting checkpoint. End checkpoint only; no val-based checkpoint search. Primary caption policy is raw greedy on the same1000 validation images. Score CIDEr/BLEU/METEOR/ROUGE and CHAIR; do not compare these point scores directly against historical beam5/reranked test5000 scores.

## Module

UPT supplies up to three selected triplets, keeping subject and object instance boxes. Each triplet gives S, O, R queries. S uses the subject label, O the object label and R the full relation phrase, rather than a standalone verb. Images without selected relations use object queries; no-object images mask all extra tokens.

Frozen CLIP text states are projected to queries, with learned S/O/R role embeddings. Cached spatial CLIP image tokens supply keys and values. Keys additionally encode patch coordinates and membership in the subject/object boxes. S attends inside its subject box, O inside its object box, and R inside the union. Tiny boxes use the nearest patch. This preserves the existing assumption that cache spatial coordinates correspond to direct Resize(224,224), without crop offsets.

The attended values contain visual features, with no text residual added to them. These evidence tokens are appended to the existing Q-Former/legacy-prompt memory for the caption decoder. There is no learned scalar gate like D. Attention is restricted by boxes; it cannot recover an object outside a wrong UPT box.

## Loss

`L = CE + scheduled(.05)*SCST + .1*L_align` for E. A omits `L_align` and the extra branch.

Matching score is cosine between a projected text query and attended image evidence. `L_align` averages two margin hinges: matched pair versus wrong text on the same region, and matched pair versus the same text on a donor region, margin .2. Donors come only from the current four-image local-rank training batch, have the same role and a different image. Any exact phrase appearing in either image's selected valid slots is excluded symmetrically. One deterministic first-eligible donor is used per eligible query. If there are no eligible donors, the alignment term is differentiable zero; images still contribute CE/SCST.

Local replay of the locked train batches found eligible negatives for 29325/31139 object slots and 3161/5307 relation slots; no local batch lacks all negatives. Subject slots are all `person`, so there are **zero subject contrastive negatives** under this rule. Subject attention learns through caption loss and box restriction, not a claimed subject-instance contrastive objective.

These are **UPT pseudo-positives and potentially noisy negatives**, not verified false relations. In particular an image may support multiple actions and UPT may omit one. The exclusion removes known duplicates, not all semantic overlap. No unlabelled relation is assumed globally false, no calibrated truth probability is claimed, and no direct verb-only discrimination is established by this loss. This whole-method pilot tests utility; explicit grounding/novelty claims would require independent annotations later.

## Checks and artifacts

Local data validation checks all11000 images' slot ordering, allowed bounds, phrase hashes and locked cohort. Synthetic CPU tests check text-sensitive attention, absence of a direct text-value shortcut, tiny-box fallback, finite CE/SCST/alignment gradients, duplicate exclusion and all-empty evidence. Kaggle preflight additionally checks two-GPU synchronization, including an empty-evidence rank, before any real caption training.

Every25 updates report CE/SCST/alignment, eligible pseudo-pair count/margin, and separate query/key/value/position/role gradient norms. These are diagnostic numbers, not relation accuracy on ground truth. End evaluation includes on/off/text-swap/region-swap interventions, reference NLL and attention entropy. Swaps measure sensitivity, not guaranteed false claims. Off masks the added branch of the same trained model; it is not the separately trained control A.

Return `semantic_evidence_pilot_A_review.zip` and `semantic_evidence_pilot_E_review.zip`. Keep `checkpoint.pth` separately in saved Output; ZIPs exclude weights/features. Checkpoint format `semantic_evidence_pilot_v1` requires this experiment's PilotModel loader. The scripts refuse existing Output/stage paths; use a fresh session or new paths rather than overwrite a completed run.

## Decision before results

Do not expand into many ablations now. A useful E should improve CIDEr versus matched A without a clear hallucination regression, and show evidence of conditional image/text use. A loss decrease or caption change alone is insufficient. Single seed and reused validation limit conclusions; no success claim or full-training recommendation is made before reviewing the outputs.
