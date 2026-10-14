# Selected relation → caption: first 12-image intervention

The selected detector JSON now has a dedicated bridge into the existing pair-grounded caption input. The bridge consumes normalized boxes directly and preserves subject/object instance IDs. It neither invokes the old pixel-box builder nor changes model weights.

Local validation: the real 100-image JSON contains 128 selected claims in 73 images; all entries pass preparation checks across six predicates. Six unit tests pass, including cache shape validation, normalized coordinates, detector-confidence weighting, rejected mismatched labels/pixel boxes, the alignment-module on/off/box-swap path, and an intervention that leaves the original tensors/text intact. CLI help and syntax compilation pass. These use synthetic text/visual vectors for module tests; no trained decoder or CLIP encoding has run locally.

## Input and scope

The input JSON is the R88 same-label control's inspected full100 output, not ground truth and not a promoted model. The first 12 lexicographic images with a different same-class, low-overlap target are used. Selection does not consult caption references or relation correctness. The alternate is UNKNOWN, not a negative label. This is a development sensitivity probe, not validation/test accuracy.

Three conditions share image features, the original generic prompt, checkpoint, beam size5 and up to3 distinct captions:

- on: newly selected instance-aware relation input;
- off: identical relation tensors, but all relation mask slots zero;
- target_swap: change just one target box, with relation text/confidence/mask fixed.

The output records captions and raw/average log probabilities, full-memory relative changes, relation masks, evidence weights and verified acceptance where available. Changed captions are not automatically better captions. Candidate-change flags include changed scores with unchanged words.

Weights are strictly loaded from the checkpoint's saved architecture and vocabulary; no random missing heads or warm start. Generic V2.1 relation text remains in all conditions. The first recommended checkpoint is the historical E3 pair-grounded epoch3 at the path below; testing its response to newer selected claims is a distribution-change probe, not a fair caption improvement comparison. V2/V3 checkpoints can also be explicitly supplied and their architecture is inferred strictly from metadata.

Relation confidence uses min(subject detector confidence, object detector confidence), not the region model logit/margin as a probability. Contrast predicates are diagnostic phrases and can coexist with positives. The cache is marked inference_only: it must not be used as verified negative supervision or evidence that the V3 gate is calibrated.

## Kaggle input

Upload the supplied `selected_relations_full100.json` as a JSON dataset input, then copy its exact mounted path into SELECTED below. It is small and has no `.pt` upload/extraction problem. Keep your checkpoint-store, generic prompt-cache and visual-cache inputs attached. No COCO images/dataset attachment is needed for this probe; filenames provide COCO IDs and the saved checkpoint provides the vocabulary.

One GPU is sufficient for 12 images. The three visual HDF5 files are read directly for the chosen images; there is no full visual-cache copy. Internet is needed for clone/dependency installation and the pinned CLIP text encoder unless it is already locally available. Outputs contain no checkpoint copies.

```python
import os, subprocess, sys, shutil
from pathlib import Path

SELECTED = Path('/kaggle/input/datasets/ducanh2403/CHANGE_ME/selected_relations_full100.json')
CHECKPOINT = Path('/kaggle/input/notebooks/ducanh2403/e3-checkpoint-store/e3_checkpoint_store/model_h1_2_crossattn_epoch_3.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
CLIP_MODEL = 'openai/clip-vit-base-patch16'
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '83980b3'
CACHE = Path('/kaggle/working/selected_grounded_probe.pt')
OUTPUT = Path('/kaggle/working/selected_caption_grounding_12')

print('Selected-caption probe bootstrap started.', flush=True)
for path in (SELECTED, CHECKPOINT, PROMPT_CACHE):
    assert path.is_file(), f'Missing file: {path}'
assert VISUAL_CACHE.is_dir(), VISUAL_CACHE
assert not CACHE.exists() and not OUTPUT.exists(), 'Use fresh output paths for a rerun.'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTHONUNBUFFERED'] = '1'

def run(args, cwd=None):
    print('Running:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], REPO)
run(['git', 'checkout', '--detach', COMMIT], REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], REPO)
run([sys.executable, '-u', 'build_selected_grounded_cache.py', '--selected', SELECTED, '--output', CACHE, '--clip-model-path', CLIP_MODEL], REPO)
run([sys.executable, '-u', 'probe_selected_caption_grounding.py', '--checkpoint', CHECKPOINT, '--selected', SELECTED, '--grounded-cache', CACHE, '--prompt-cache', PROMPT_CACHE, '--visual-cache', VISUAL_CACHE, '--output', OUTPUT, '--sample-size', '12', '--beam-size', '5', '--candidate-count', '3'], REPO)
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('Download review:', archive, flush=True)
```

Return the small `selected_caption_grounding_12.zip`. No further human annotation or caption training is required for this probe. Actual trained-checkpoint results are pending; no BLEU/CIDEr/CHAIR improvement is claimed.
