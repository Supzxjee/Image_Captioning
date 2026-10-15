# Training object reward on top of CE-anchored SCST

Status: PREPARED_NOT_RUN. The user launches Kaggle manually. No performance claim.

## Matched experiment

Two independent GPU jobs restart from the **original QFormer epoch10**, not from a selected validation checkpoint. Keep the same deterministic train20k/validation1000 cohorts, prompts, frozen visual cache, all previously trainable model parameters, seed42, batch8, one pass/2,500 updates, AdamW5e-6, FP32 and dropout disabled. R119 provides the rationale for retaining CE; adding an object term is now the only treatment difference.

| GPU | Arm | Objective |
|---|---|---|
| 0 | scst_ce | CE + .05SCST with CIDEr reward |
| 1 | scst_ce_object | CE + .05SCST with CIDEr + object reward |

No new detector, VLM, prompt cache, alignment module or reranker. This is an object-consistency training control, not a claimed novel relation-grounding method. The object term changes the sampled-minus-greedy reward entering policy gradients; it is not an extra input at inference.

## Labels and states

The inspected annotation cache is from CHAIR-metric-standalone commit4087a26211aa2339b9a76307cb8f0321ef691d0a. Its source builds support sets from COCO2014 instance annotations plus reference-caption object labels. Cache SHA25673ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3,2,057,907 bytes. **Reward generation reads labels only for selected train/restval images**, checks the source dataset split and exports a20k-image target file. Validation/test support labels are never used for training rewards. The downloaded source cache itself contains all splits, so this filtering is explicit rather than a claim that the entire cache is training-only.

For each training image:

- Supported classes are the cache's annotation support set, union any extra classes parsed from that image's five training references. References are positive evidence only; this union avoids treating alias differences between parsers as negative evidence.
- Salient classes are the unique recognized classes mentioned across those five training references. This gives a coverage term without requiring a caption to list every annotated background object.
- A class absent from support but detected at confidence>=.9 is neutral: neither rewarded nor penalized. Detector confidence is not calibrated truth; this is a conservative exemption only. Detector absence alone does not create a negative label.
- The penalty covers recognized classes in the COCO80 ontology that are absent from support and not neutral. Call them **annotation-unsupported**, not definitively physically absent. Annotations and lexical mapping can be incomplete or wrong.
- Outside-ontology words are neutral. If support remains empty after reference union, the entire object reward is zero; do not treat every generated noun as false.

Local inspection verified annotation coverage20,000/20,000 for the exact cohort, no classes outside COCO80, and no held-out overlap.121 selected images have an empty cache support set before the training-reference union. The final eligible count depends on the real training references and is saved at runtime; do not claim all121 remain unavailable. Full raw training references/detector data are not present locally, so actual20k target construction is still a Kaggle preflight step.

## Fixed reward formula

Let P be unique recognized classes in a generated caption, G the support set, S the salient classes and U the detector-neutral classes:

`coverage = |P intersect S| / |S|` (zero if S empty).

`unsupported_fraction = |P minus G minus U| / max(|P minus U|,1)`.

`R_object = .5 * coverage - unsupported_fraction`.

`R_total = R_CIDEr + 1.0 * R_object` for treatment; control uses CIDEr alone. CIDEr remains the same training-IDF CIDEr-D x10 with Treebank normalization. Object coefficient1 and coverage coefficient.5 are one provisional fixed setting, not a tuned optimum. R_object lies in[-1,.5]. Both sampled and greedy captions receive the same scoring policy; SCST uses their total-reward difference.

The unique-class formulation prevents repeating the same noun for extra reward. Neutral classes cannot dilute the penalty denominator. A caption with no recognized objects gets zero object reward, rather than full correctness credit. Example with person+bicycle salient: correct person+bicycle .5; generic person .25; person+unsupported motorcycle -.25. These are unit-test examples, not qualitative model results. Coverage reduces an incentive to omit all objects but does not guarantee the model cannot exploit the parser or become verbose.

The noun matcher is the existing COCO alias parser. It does not verify negation, exact instances, relations or attributes. A person carrying a bicycle can receive the same object score as a person riding it. Optimizing a reward sharing the CHAIR ontology may improve that metric without improving general visual grounding; that is a stated evaluation limitation.

## Execution and logs

Use SCST_CE_OBJECT_REWARD_TRAIN20K_VAL1000.ipynb or kaggle_scst_ce_object_reward.py, replacing the old experiment cell in **QFormer UPT Val5000 20261015**. GPU T4x2, Internet on, same five inputs; no new Kaggle dataset. The already pinned annotation cache is about2MB. HDF5 features are read in place. Preflight runs original SCST11, anchor3 and object reward10 tests; two logging-helper tests also passed locally.

After targets are built and hashed, training starts. Save only training labels in train_object_targets.json. Treatment logs separate CIDEr/object/total rewards, coverage, unsupported fraction, available-supervision fraction, CE and SCST losses. Accumulate RL and CE gradients, clip their sum once and take one optimizer step per batch. Keep the same raw beam5 and fixed .3 decoder/.4 CLIP/.3 OCC policies; no UPT. No reward-label lookup is used at inference.

Evaluate original, anchor control and object-reward treatment on the same1,000 validation images. Report official PTB BLEU/METEOR/ROUGE-L/CIDEr and CHAIRs/i/Recall, caption lengths and paired caption changes. Primary exploratory gate: treatment raw CIDEr exceeds both controls, raw CHAIRs/i no worse than either, raw Recall within1 percentage point of the better control. Additionally, fixed-rerank CIDEr/CHAIRs/i must be no worse than both original-fixed and anchor-fixed, with Recall within1 percentage point. Preserve the original reference even if the control regresses. A point-estimate gate does not establish statistical superiority; this is one seed on previously inspected validation.

Return `/kaggle/working/qformer_scst_ce_object_train20k_val1000_review.zip`. It excludes weights. Keep separately:

- `/kaggle/working/qformer_scst_ce_object_train20k_val1000/scst_ce/checkpoint.pth`
- `/kaggle/working/qformer_scst_ce_object_train20k_val1000/scst_ce_object/checkpoint.pth`

No notebook was launched by the assistant. No new train/evaluation results exist yet.

References: [COCO dataset](https://cocodataset.org/#download), [pinned annotation-cache implementation](https://github.com/Maxlinn/CHAIR-metric-standalone/blob/4087a26211aa2339b9a76307cb8f0321ef691d0a/chair.py).
