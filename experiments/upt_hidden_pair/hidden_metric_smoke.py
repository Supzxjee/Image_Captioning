"""User-run metric dependencies before staging/training; no caption model loading."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from captioning.scst import TrainingCider
from captioning.object_reward import _AnnotationReader
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider


def main(chair):
    refs = {1: ['a person holding an umbrella'] * 5, 2: ['a dog running through grass'] * 5}
    reward = TrainingCider(refs, reference_cache_size=2).score([1, 2], [r[0] for r in refs.values()])
    assert np.isfinite(reward).all() and (reward > 0).all()
    tok = PTBTokenizer()
    gt = tok.tokenize({i: [dict(caption=s) for s in captions] for i, captions in refs.items()})
    hyp = tok.tokenize({i: [dict(caption=captions[0])] for i, captions in refs.items()})
    score, rows = Cider().compute_score(gt, hyp)
    assert np.isfinite(rows).all() and abs(score - np.mean(rows)) < 1e-12
    with (Path(chair) / 'chair.pkl').open('rb') as stream:
        annotation = _AnnotationReader(stream).load()
    ids = sorted(annotation.imid_to_objects)[:2]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / 'input.json').write_text(json.dumps([dict(image_id=i, caption='a person') for i in ids]))
        subprocess.run([sys.executable, '-u', str(Path(chair) / 'chair.py'), '--cap_file', str(root / 'input.json'),
                        '--cache', str(Path(chair) / 'chair.pkl'), '--save_path', str(root / 'output.json')], cwd=chair, check=True)
        assert 'overall_metrics' in json.loads((root / 'output.json').read_text())
    print('TRAIN CIDEr-D, fixed-IDF caption CIDEr/PTB Java, and CHAIR smoke PASS', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
