# Direct decoder relation memory: 12-image routing pilot

R91 showed that the historical E3 path produced unchanged candidate wording under relation on/off/target-swap. This implementation adds an explicit group of relation slots to the decoder cross-attention memory. It is a new routing hypothesis, not a validated caption improvement or a novelty claim.

## Implemented behavior

For each existing instance-grounded relation, take the projected text vector t, the pair vector p (subject/object/union patches plus geometry), and detector confidence c. Append `c * LayerNorm(t + c * p)` to the usual generic-prompt + visual memory. The normalization/projections reuse the existing pair module. No additional parameters or pretrained model are introduced.

This path is independent of the old learned visual-query update gate. It preserves pair box evidence instead of only appending generic relation text. Confidence is a detector weight, not a calibrated probability of the relation. Invalid and zero-confidence slots are zeroed and attention-masked. The old residual remains present in both arms; the comparison isolates the appended decoder slots.

Default behavior remains disabled. `--decoder-relation-memory` enables the experimental training/evaluation architecture for V1/V2 only. V3 is deliberately unsupported because bypassing its accepted-relation filter would violate its contract. Saved checkpoints include the architecture flag; strict evaluation rejects a mismatch. Explicit training warm start can upgrade a legacy checkpoint with the same tensors. Relation slots are excluded from return_visual/ITC visual features.

14 local tests pass: the five new tests cover real encoder/decoder memory dimensions, masks, no-relation row equivalence, zero-confidence behavior, decoder-logit response, gradient to pair projection, visual-output boundaries and checkpoint/config guards; nine existing cached encoder/grounding tests pass. CLIP is stubbed and features/weights are synthetic in these local engineering checks. No trained-caption result is available yet. Diagnostic inference-only caches are now rejected by the training loader.

## First run: same checkpoint, no training

Run the same 12 images and input files as R91, with six conditions: legacy on/off/target_swap and direct on/off/target_swap. New direct slots are untrained for this decoding route. This checks whether the proposed route provides usable sensitivity before paying for adaptation training. It cannot establish quality improvement: altered captions may be worse or hallucinate predicted relations.

The same generic V2.1 prompt remains in every condition. Alternate boxes are unknown labels, not negatives. Compare image-supported content after reviewing the returned captions; do not tune gates or choose favourable images from this result.

## Kaggle cell

Use the notebook that just produced `selected_caption_grounding_12.zip`. It already has the working SELECTED, CHECKPOINT, PROMPT_CACHE and VISUAL_CACHE paths. Paste this as a new cell after those definitions. On a fresh session, first copy those four path assignments from the previous working cell; no new input upload is required.

```python
import os, subprocess, sys, shutil
from pathlib import Path

for name in ('SELECTED', 'CHECKPOINT', 'PROMPT_CACHE', 'VISUAL_CACHE'):
    assert name in globals(), f'Copy the working {name} = Path(...) assignment from the previous probe.'
for path in (SELECTED, CHECKPOINT, PROMPT_CACHE):
    assert Path(path).is_file(), f'Missing file: {path}'
assert Path(VISUAL_CACHE).is_dir(), VISUAL_CACHE

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = 'cc4dfc6'
CACHE = Path('/kaggle/working/direct_decoder_relation_probe.pt')
OUTPUT = Path('/kaggle/working/direct_decoder_relation_memory_12')
assert not CACHE.exists() and not OUTPUT.exists(), 'For a rerun, choose new CACHE and OUTPUT names.'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTHONUNBUFFERED'] = '1'
print('Direct decoder relation memory probe started.', flush=True)

def run(args, cwd=None):
    print('Running:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], REPO)
run(['git', 'checkout', '--detach', COMMIT], REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], REPO)
run([sys.executable, '-u', 'build_selected_grounded_cache.py', '--selected', SELECTED, '--output', CACHE, '--clip-model-path', 'openai/clip-vit-base-patch16'], REPO)
run([sys.executable, '-u', 'probe_selected_caption_grounding.py', '--checkpoint', CHECKPOINT, '--selected', SELECTED, '--grounded-cache', CACHE, '--prompt-cache', PROMPT_CACHE, '--visual-cache', VISUAL_CACHE, '--output', OUTPUT, '--sample-size', '12', '--beam-size', '5', '--candidate-count', '3', '--compare-decoder-memory'], REPO)
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('Download review:', archive, flush=True)
```

One GPU suffices. No COCO image attachment, full visual-cache copy, new manual labels, caption training or checkpoint export. The pinned CLIP encoder may reuse the previous download. Return `direct_decoder_relation_memory_12.zip`, containing records and summary only.

## Decision after the run

If the new route is sensitive, inspect whether changes correspond to supported objects/actions and whether the incorrect predicted relations are copied into captions. Only then prepare a matched, small caption-adaptation pilot with a training cache and an untouched evaluation subset. If wrong-box sensitivity remains negligible, investigate pair representation rather than forcing higher gate values. No trained result, CIDEr/BLEU/CHAIR gain or final model promotion is claimed at this stage.
