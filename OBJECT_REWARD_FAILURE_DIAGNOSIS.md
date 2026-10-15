# Object-reward failure diagnosis

Date:2026-10-16. This is a local code/artifact diagnosis, not a new training experiment. Official R119/R120 metrics remain unchanged. Evidence: `audit/object_reward_failure_diagnosis.json`, reproduced by `diagnose_object_reward_failures.py` against the two user review ZIPs.

## 1. Confirmed training-reward alias gaps

The legacy object parser misses two names in actual validation error captions:

| COCO ID | Caption excerpt | Missing mapping |
|---|---|---|
| 568205 | a woman putting a **turkey** into an oven | turkey → bird |
| 333772 | a cat laying on top of a wooden **desk** | desk → dining table |

The treatment has40 CHAIR hallucinated mentions. The legacy parser recognizes36; these two alias gaps account for2 more. The remaining2 mentions are context-ambiguous mother/baby mappings discussed below. The two alias misses occur in the matched control too; they do not explain the entire treatment-control regression.

Fix implemented: an **opt-in** `aliases_v2` reward parser adds turkey/turkeys and desk/desks. It is used consistently for training-reference salience and generated-caption reward when selected by `--object-reward-parser aliases_v2`. The default remains `legacy`, so old published launchers and scores are not silently redefined. The fixed-rerank object parser is deliberately unchanged to preserve that baseline. The parser choice is recorded in protocol and target metadata. This vocabulary repair is not a claim of context-complete or instance-level understanding.

## 2. Confirmed lack of bitwise control replay; exact GPU cause unconfirmed

Compare R119's anchored control with R120's anchored control, which share declared data, seed, objective and initialization source:

| Field | First exact difference | Previous | Current |
|---|---:|---:|---:|
| All common step1 fields except elapsed time | none | identical | identical |
| CE loss | step2 | 1.7735275030 | 1.7735276222 |
| SCST loss | step2 | -.4873808622 | -.4873812199 |
| Pre-clip gradient norm | step3 | 4.7032666206 | 4.7032217979 |
| Sampled-caption mean reward | step25 | .5067337155 | .5515753627 |
| Greedy-caption mean reward | step29 | 1.0633796453 | 1.0783262253 |

CE loss differs by1.1920929e-7 at step2, before a sampled reward difference is observed. This supports a numerical divergence followed by a changed sampling trajectory; rewards alone do not reveal the exact first token difference. Old artifacts contain neither per-batch feature/token hashes nor enough runtime state to identify the responsible kernel/library/environment change. Do not assert cuBLAS, SDPA, data order, dropout or a wrong source checkpoint as the proven root cause.

Code inspection confirms a reproducibility gap: old training sets seeds and disables TF32, but does not enforce deterministic algorithms or a fixed attention backend. Seed equality alone is not sufficient; PyTorch also does not guarantee cross-release/device reproducibility. [Official PyTorch2.10 reproducibility guidance](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

Fix/instrumentation implemented behind `--strict-reproducibility`:

- Worker environment sets PYTHONHASHSEED42 and CUBLAS_WORKSPACE_CONFIG=:4096:8 before process CUDA initialization.
- Enforce deterministic algorithms with errors rather than warn-only fallback; disable cuDNN benchmarking; require deterministic cuDNN; disable TF32.
- Restrict attention to math SDPA and disable the MHA fastpath for the strict diagnostic run, limiting distinct no-grad/grad backend paths. Apply the same setting to beam workers. This deliberately changes numerical execution paths and must be applied to both comparison arms; it is not an assertion that old inference was incorrect.
- Save Python/package/CUDA/cuDNN/device/backend metadata in runtime.json and initial model tensor identity in strict mode. Hash batch indices, visual inputs, captions, prompts and sampled/greedy tokens for the first32 steps. These diagnostics do not enter model inputs or rewards.

The CPU sampled-update replay test is exact for32 small synthetic steps. **No T4 deterministic replay has been executed.** A same-environment short GPU repeat is still needed before claiming this resolves the observed control drift. Strict execution can be slower or fail on an unsupported operation; no silent fallback is allowed. No new notebook was launched or published to rerun the full experiment.

## 3. Potential evaluator false positive; keep official metrics

COCO309499 changes from “a couple of zebra standing next to each other” to “a mother zebra and her baby in a field.” CHAIR maps both mother and baby to person even though these words can refer to the zebra family. This is a context-ambiguous lexical mapping, not evidence of two distinct human objects in the sentence. No image-level human adjudication was performed here.

Do not blindly add mother/baby→person to the reward dictionary just to match CHAIR. Do not remove this case from published CHAIR or replace the official3.9% result with an adjusted number. Keep it as an explicit qualitative review item alongside the standard metric.

Of the six new raw CHAIR-error captions versus control, three add an unsupported table; one introduces scissors; one introduces knife; one is the zebra mother/baby case. The other five are recognized by the current noun parser. Fixing turkey/desk alone therefore cannot explain or repair all six new errors.

## 4. Confirmed reward-design counterexample, not a demonstrated training exploit

The fraction penalty can be diluted while the same unsupported object remains. Synthetic support={person,sink}, salient={person}, no neutral labels:

| Caption | Unsupported class | Object reward |
|---|---|---:|
| person next to a motorcycle | motorcycle | 0 |
| person next to a sink and a motorcycle | motorcycle | .166667 |

Adding a supported but non-salient sink improves the score solely by changing the penalty denominator. No erroneous motorcycle has been removed.11,120/20,000 training target rows contain at least one supported class outside the salience set, so this design possibility is not limited to a nonexistent target structure. This does not prove the model exploited it in the actual run. A non-dilutable count or per-object penalty is a candidate follow-up objective, not silently substituted here.

The reward also permits CIDEr gain and positive coverage to offset a penalty; it is not a hard hallucination constraint. That is a design tradeoff, not a reversed gradient or arithmetic bug. Recorded reward/loss identities and gradient-sign tests remain correct.16/20,000 target rows have detector-neutral exemptions, and458 have empty salience; neither count by itself establishes a causal explanation for the validation result. No evidence here justifies removing conservative neutral handling.

## What is fixed versus pending

Implemented and tested locally: versioned alias repair, opt-in strict execution, runtime/model/batch/token trace evidence, and regression tests retaining legacy behavior.34 distinct local tests passed, including the existing SCST/anchor/object/preflight suite and8 new diagnosis tests. No new task-specific model metrics exist.

Pending: same-GPU short-run replay with strict settings; image/context review of the flagged zebra caption; a separately controlled decision on non-dilutable object penalties. Prioritize validating control replay before another full training run. Do not combine a parser change, new reward formula and runtime change into an unexplained single treatment.
