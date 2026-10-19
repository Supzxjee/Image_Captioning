# Frozen E matcher diagnostic

Use [FROZEN_E_MATCHER_PROBE_2GPU.ipynb](FROZEN_E_MATCHER_PROBE_2GPU.ipynb). This notebook only scores candidates using the **completed E checkpoint**, with its trained text-query/visual-value cosine matcher frozen. It does not train, generate captions, select a checkpoint, or alter an existing output. Both GPUs perform independent inference shards of this one diagnostic; there are no DDP collectives.

## Run manually on Kaggle

1. Import the notebook and select T4 x2, Internet on.
2. Set `CHECKPOINT` to E's exact `checkpoint.pth`, SHA256 `651f411f589c5dd7beb0fae98fafaa7d9ea06a7fd7f2b49400ac878f60227b18`. The default `/kaggle/working/semantic_evidence_pilot_E/checkpoint.pth` works only in the still-active E session. In a new notebook, attach E's saved output or upload that single checkpoint as a dataset and paste its actual mounted path. Do not substitute A, the original SCST checkpoint, or an independently reserialized file.
3. Keep `VISUAL_CACHE=/kaggle/input/datasets/ducanh2403/visual-cache`, or paste its actual mounted path on the other account.
4. Run it yourself. Download `/kaggle/working/semantic_evidence_matcher_probe_review.zip` afterward. Keep E's checkpoint separately; the ZIP contains scores, summaries, provenance, source and worker logs, not model weights.

Only the E checkpoint and existing visual HDF5 dataset are required. The diagnostic manifest is embedded; there is no raw-image, prompt-cache, COCO JSON or V-COCO dataset download. The pinned CLIP text encoder is fetched automatically. Visual entries are read directly from the cache, not copied in full to `/tmp`. The code repository goes to `/tmp` so it does not clutter saved Output. Each worker prints progress in its log; the parent displays the latest line every ten seconds.

## Locked diagnostic: 62 pairs on 59 images

| Task | Pairs | Label construction |
|---|---:|---|
| Wrong object instance | 12 | Existing human judgments: all three boxes clear, same object class, exactly one supported and one contradicted claim |
| Wrong predicate | 12 | Fixed subject/object boxes; V-COCO positive role versus explicitly zero actor/action for the alternate verb |
| Wrong subject instance | 38 | Same object and predicate; positive actor versus another person explicitly labeled zero for that action |

Holding and riding may coexist. Missing edges and unknown actor labels are never treated as negatives. Predicate-swap positives are balanced **within each object class**: bicycle 5 holding/5 riding; motorcycle 1/1. Thus the verb check does not reward simply associating an object class with whichever action is more frequent in this set. Other tasks report their actual action mix rather than claiming balance they do not have. Candidate order is fixed by ID hash, without model scores. Ties count as half credit.

Every pilot-training image was removed before selection. Nevertheless, **all 59 images are Karpathy train/restval**, and the captioning backbone has previously trained on this split. Existing human judgments were also reused in earlier research. This is a diagnostic held outside the latest E updates, not an unseen validation/test benchmark. No Karpathy test labels or images enter the probe. GT/human boxes also differ from UPT detector boxes used in E training. V-COCO labels follow their annotation ontology, not exhaustive truth for every natural-language meaning of a verb.

## Scores and interpretation

The trained R-role cosine scorer compares the same-image positive and negative candidate. Three inference conditions share exactly the same frozen weights:

- `boxed`: the trained attention with subject/object membership and allowed union box.
- `global_view`: same image/text, but remove instance membership and allow the full image. This asks whether box restrictions help.
- `zero_visual`: zero the patch features while retaining the query/weights. This checks lexical and learned bias; it is an out-of-distribution control, not a fair standalone model.

Report tie-aware pair accuracy, image-macro accuracy and 2000 image-cluster bootstrap resamples. No cosine threshold is tuned. These scores are not confidence probabilities. Only twelve predicate contrasts exist after strict filtering, so their uncertainty will be large; do not use the result to claim general relation understanding.

If boxed scores cannot distinguish confirmed alternatives, the training pseudo-pair accuracy was insufficient evidence of the required fine distinctions. If boxed scores are useful while earlier caption interventions remain weak, that motivates examining decoder use of those signals. Neither outcome alone identifies the cause: limited cases, box/cache resolution, GT-versus-detector box shift, label ambiguity and metric choice remain possible explanations. A good score is not proof that caption quality improves.

## Local preparation and checks

`prepare_probe.py --local-output-root C:/KhoaLuanTotNghiep/Code/output --output probe_manifest.json` reproduces the manifest from the pinned caption JSON, existing grounded V-COCO development role labels and the user's previous human review. Source hashes, the excluded pilot cohort identity, counts and candidate labels are saved. Reviewer name and free-text comments are not copied into this public manifest.

Five synthetic data tests passed locally: explicit-negative handling, training-image exclusion, class/action balancing, human-label eligibility, ties and invalid boxes. Python and notebook code cells parse, and all notebook cells remain unexecuted. See [local_validation.json](local_validation.json). A synthetic model test runs as Kaggle preflight. **No GPU inference/checkpoint replay has been run locally**, because the E checkpoint is not present locally and the current local runtime lacks torch.
