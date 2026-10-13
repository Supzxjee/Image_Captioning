# Separate actor compatibility: local development result

Two actor verifiers were implemented and actually trained locally. Neither is promoted to the full captioning pipeline. The nonlinear verifier fixes the inspected child's wrong-actor food links, but performs poorly on the labelled development actor contrasts. The compact verifier rejects all relation prompts under the train-fitted acceptance rule.

## Supervision and method

Training reuses V2 frozen CLIP features: 1,024 training pairs and 256 development pairs with image-disjoint splits. Only endpoints of confirmed same-image/same-object actor contrast edges supervise the verifier. There are 151 training edges in 120 images, 298 unique labelled endpoint claims. Unknown links remain unknown. Negatives are annotated absence of that actor's action, not exhaustive incorrect object binding labels.

Per-predicate train positive/negative endpoint counts: holding85/87, riding10/10, sitting_on24/24, carrying18/18, eating8/8, talking_on3/3. This is a very small dataset for learning actor/action compatibility, especially eating and phone actions.

- Nonlinear verifier: subject/object CLIP vectors, their product/difference, and ten continuous normalized box/offset coordinates; MLP128→64→six logits. Union crop is excluded. Approximately272k parameters.
- Compact verifier: ten box/offset coordinates, six subject/text cosines, six object/text cosines, subject/object cosine and squared center distance; linear24→six logits,150 parameters. Union crop is excluded.
- Each run uses seed42,30 fixed epochs, AdamW lr0.001/weight decay0.01, equal-weight known-endpoint BCE and same-object actor softplus ranking. No epoch selection from dev. The compact model was tried after observing nonlinear failure, so these are sequential development trials, not a pre-registered architecture benchmark. The same optimization settings do not establish that either architecture has been optimally trained.
- Verifier thresholds are fitted on **train endpoints only**, requiring >=5 known accepts and >=90% empirical train precision. Veto is enabled only for predicates with >=5 train positives and negatives; otherwise baseline passes through unverified. Sigmoid activations are not calibrated probabilities.
- Selection requires both the frozen R50 relation threshold and verifier threshold before the existing cap3. This allows multiple actors; it does not force a winner per object. Original R50 scores, thresholds and predictions remain unchanged.

## Results

| Measure | Frozen R50 | Nonlinear actor head | Compact actor head |
| --- | ---: | ---: | ---: |
| Same development actor edges correctly ordered | 36/40 | 19/40 | 17/40 |
| Selected claims on20-image detector audit after verifier | 5 | 4 | 0 |
| Relation-bearing images | 3 | 3 | 0 |

Actor ordering compares the base scorer with verifier logits, not fused score AP. There are only40 labelled actor edges in32 dev images; eating has one dev contrast and riding none. Verifier endpoint AP must not be compared directly with R50 macro AP over all known claims: their evaluation masks differ.

The nonlinear head accepts the foreground child's eating-sandwich and eating-hot-dog links at activation~0.8837, and rejects both background-person links at~0.1355 (train-fit eating threshold0.7749). This fixes one visually inspected error pattern but does not establish generalization. It still accepts holding a worn backpack and the ambiguous sitting-on-bed label. The conflicting sandwich/hot-dog detection labels remain unresolved.

The nonlinear head has poor threshold transfer on labelled dev endpoints: holding precision~51.5%, sitting50%, carrying33.3%. The compact head finds no train threshold satisfying the acceptance rule, so it abstains for every supported predicate and all20 audit images are object-only. Zero selected false relations caused by zero selected relations is not a successful method.

## Decision

Preserve both trials, do not replace R50, and do not train caption fusion on these verifier outputs. Capacity reduction alone has not solved the problem. Overfitting/data scarcity is a plausible explanation, not proven from these results; the compact trial may also need different optimization.

The next bounded experiment expands **confirmed actor supervision**, rather than arbitrarily tuning thresholds on the child image. Earlier V2 metadata reports658 actor edges in the full eligible official V-COCO train pool versus151 cached. The full endpoint extraction enumerates all actor counterparts without the old four-negative-per-positive cap, so its exact count is determined and logged after preparation.

A new development exclusion manifest contains1,388 images: previous diagnostic/VLM image IDs plus the388-image R53 cohort. It excludes these from new official V-COCO val development selection; all20 detector-audit images are already in that cohort. Official V-COCO train remains the training source; caption validation/test images are filtered out and train/dev image overlap is checked. GT boxes are used for supervision, so expanded training alone does not solve detector distribution shift. The old20-image set stays a development failure audit, not an untouched benchmark.

## Local outputs and reproduction

Outputs are under `C:/KhoaLuanTotNghiep/Code/output/separate_actor_compatibility_v1` and `C:/KhoaLuanTotNghiep/Code/output/compact_actor_compatibility_v1`. Both contain head checkpoints, train histories, labelled dev predictions, verifier veto decisions and structured prompts. Original frozen baseline is preserved.

```powershell
$env:PYTHONPATH='C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps'
& 'C:/Users/leduc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' run_actor_compatibility_cached.py --train-source 'C:/Users/leduc/Downloads/instance_contrast_alignment_v2_diagnostic_1024_review.zip' --candidate-source 'C:/Users/leduc/Downloads/residual_detector_full_candidates_20_features_review.zip' --scored 'C:/KhoaLuanTotNghiep/Code/output/residual_detector_full_candidates_20_scored' --output 'C:/KhoaLuanTotNghiep/Code/output/separate_actor_compatibility_v1'
# Compact run: add --compact and use a separate output directory.
```

Five actor checks and16 existing supervision/protocol tests pass. They verify unknown actors are never negative, actor input response and union exclusion, invalid contrast rejection, compact parameter count/union exclusion, and exhaustive counterpart enumeration. No caption metric, independent human precision, or novelty claim is made.
