"""Fail on metric/tokenizer dependencies before staging or training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from captioning.metrics import score_files
from captioning.scst import TrainingCider
from captioning.object_reward import _AnnotationReader


def main(chair):
    references = {1: ['a person holding an umbrella']*5, 2: ['a dog running through grass']*5}
    reward = TrainingCider(references, reference_cache_size=2).score([1,2],[r[0] for r in references.values()])
    assert np.isfinite(reward).all() and (reward > 0).all()
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)
        gt = dict(info={},licenses=[],type='captions',images=[{'id':i} for i in references],
                  annotations=[dict(id=i*5+j,image_id=i,caption=text) for i,refs in references.items() for j,text in enumerate(refs)])
        (p/'gt.json').write_text(json.dumps(gt))
        (p/'pred.json').write_text(json.dumps([dict(image_id=i,caption=refs[0]) for i,refs in references.items()]))
        result = score_files(p/'pred.json',p/'gt.json',p/'metrics.json')
        assert np.isfinite(result['CIDEr'])
        with (Path(chair)/'chair.pkl').open('rb') as stream: annotation = _AnnotationReader(stream).load()
        ids = sorted(annotation.imid_to_objects)[:2]
        (p/'chair_input.json').write_text(json.dumps([dict(image_id=i,caption='a person') for i in ids]))
        subprocess.run([sys.executable,'-u',str(Path(chair)/'chair.py'),'--cache',str(Path(chair)/'chair.pkl'),
                        '--cap_file',str(p/'chair_input.json'),'--save_path',str(p/'chair_output.json')],cwd=chair,check=True)
        assert 'overall_metrics' in json.loads((p/'chair_output.json').read_text())
    print('CIDEr reward, official caption metrics and CHAIR smoke PASS',flush=True)


if __name__ == '__main__': main(sys.argv[1])
