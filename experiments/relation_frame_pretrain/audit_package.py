"""Local static + real cohort metadata audit; never imports Torch/model/runtime."""
import ast
import json
from pathlib import Path
import tempfile
from frame_data import metadata, save


def main():
    here=Path(__file__).resolve().parent;root=here.parents[1];inventory=root/'experiments/relation_frames_inventory'
    texts=json.loads((inventory/'review/cache_loaded.json').read_text(encoding='utf-8'))['texts']
    with tempfile.TemporaryDirectory(prefix='pretrain-metadata-',dir=inventory/'review') as temp:
        stage=Path(temp)
        for n in ('images.json','detector_boxes.json','receipt.json'):(stage/n).write_text(texts[n],encoding='utf-8')
        (stage/'relation_frames_manifest.bin').write_bytes((inventory/'V2_MANIFEST.json.gz').read_bytes())
        (stage/'inputs.h5').touch()  # metadata audit only, no feature reads or integrity claim.
        vocab,images,packed=metadata(stage,root/'experiments/semantic_grounding_method/data/gain_metadata.json.gz',verify=False)
        assert len(vocab)==10000 and len(images)==612
        assert sum(len(packed[k]) for k in packed)==320
        assert all(p['weights'].sum()>0 for pairs in packed.values() for p in pairs)
    files=[]
    for path in sorted(here.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'),filename=path.name);files.append(path.name)
    save(here/'STATIC_CHECKS.json',dict(status='STATIC_AND_METADATA_PASS',source_files=files,
        actual_frozen_manifest_pairs=320,predicted_target_bindings=1280,images=612,vocabulary_size=10000,
        all_lexical_spans_valid=True,local_neural_execution=False,
        not_checked=['Actual HDF5 payload or neural values','Torch/autograd/numerical/DDP checks: user-run Kaggle only']))
    print('Static/real cohort metadata PASS; no Torch execution; neural checks remain user-run.')


if __name__=='__main__':main()
