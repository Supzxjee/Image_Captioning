"""Validate embedded artifacts and annotation policy only; never run a notebook/model."""
import ast
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
from make_notebook import HERE,ROOT,sources,COMMIT
sys.path.insert(0,str(ROOT/'experiments/semantic_evidence_pilot'))
from pilot_data import cohorts


def main():
    notebook=HERE/'VERIFIED_SAME_IMAGE_F_2GPU.ipynb'
    nb=json.loads(notebook.read_text(encoding='utf-8'))
    assert notebook.stat().st_size<900000
    assigned={};codes=[]
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        assert cell['execution_count'] is None and cell['outputs']==[]
        code=''.join(cell['source']);codes.append(code)
        for node in ast.parse(code).body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ('BUNDLE_B64','BUNDLE_SHA','SOURCE_HASHES','VERIFIED_MANIFEST_SHA','PROJECT_COMMIT','ARM'):
                        assigned[target.id]=ast.literal_eval(node.value)
    assert sum(len(''.join(c['source']).encode()) for c in nb['cells'])<900000
    bundle=zlib.decompress(base64.b64decode(assigned['BUNDLE_B64'],validate=True))
    assert hashlib.sha256(bundle).hexdigest()==assigned['BUNDLE_SHA']
    payload=json.loads(bundle);assigned.update(SOURCES=payload['sources'],VERIFIED_LABELS=payload['labels'])
    assert assigned['SOURCES']==sources() and assigned['PROJECT_COMMIT']==COMMIT and assigned['ARM']=='F'
    for name,source in assigned['SOURCES'].items():
        ast.parse(source);assert hashlib.sha256(source.encode()).hexdigest()==assigned['SOURCE_HASHES'][name]
    labels=assigned['VERIFIED_LABELS'];assert labels==(HERE/'verified_manifest.json').read_text(encoding='utf-8')
    assert hashlib.sha256(labels.encode()).hexdigest()==assigned['VERIFIED_MANIFEST_SHA']
    # Run precisely the embedded data tests in an isolated directory, without imports of model code.
    with tempfile.TemporaryDirectory(prefix='verified_F_data_',dir=HERE) as directory:
        temp=Path(directory)
        # Exercise just our generated unpack/install block, not setup, GPU code, or a notebook cell.
        setup=next(code for code in codes if 'BUNDLE_B64 = ' in code)
        bootstrap='import base64, hashlib, zlib\n'+setup.split('import base64, hashlib, zlib\n',1)[1]
        exec(compile(bootstrap,'embedded_bundle_install','exec'),{'REPO':temp,'json':json})
        for name,source in assigned['SOURCES'].items():assert (temp/name).read_text(encoding='utf-8')==source
        assert (temp/'verified_manifest.json').read_text(encoding='utf-8')==labels
        for test in ('test_pilot_data.py','test_verified_data.py'):
            subprocess.run([sys.executable,str(temp/test)],cwd=temp,check=True)
    m=json.loads(labels)
    dataset=json.loads(Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json').read_text(encoding='utf-8'))
    groups=cohorts(dataset);train={r['coco_id'] for r in groups['train']};val={r['coco_id'] for r in groups['val']}
    split_ids={k:{r['coco_id'] for r in rows} for k,rows in m['splits'].items()}
    assert not train&split_ids['probe'] and not val&set.union(*split_ids.values())
    fit=m['splits']['fit'];directions=Counter()
    for row in fit:
        assert row['source_positive']['label']==1 and row['source_negative']['label']==0
        assert row['source_positive']['object_id']==row['source_negative']['object_id']
        if row['task']=='wrong_predicate':
            positive,negative=row['source_positive']['action'],row['source_negative']['action']
            directions[row['object_label'],tuple(sorted((positive,negative))),positive]+=1
            assert row['candidates'][0]['bounds']==row['candidates'][1]['bounds']
        else:assert row['candidates'][0]['text']==row['candidates'][1]['text']
    for (obj,pair,action),count in directions.items():
        assert count==directions[obj,pair,next(a for a in pair if a!=action)]
    result=dict(status='LOCAL_DATA_AND_NOTEBOOK_VALIDATION_PASS',notebook_executed=False,model_tests_executed=False,
        two_gpu_tests_executed=False,reason='No local PyTorch; mandatory model/DDP preflight embedded for manual Kaggle run',
        data_tests_passed=7,code_cells_parsed=len(codes),embedded_python_files=len(sources()),
        notebook_bytes=notebook.stat().st_size,kaggle_source_limit_bytes=1000000,compressed_bundle_roundtrip_pass=True,
        verified_manifest_canonical_sha256=assigned['VERIFIED_MANIFEST_SHA'],counts=m['counts'],
        auxiliary_main_caption_train_overlap=len(train&split_ids['fit']),
        probe_main_caption_train_overlap=len(train&split_ids['probe']),caption_val_auxiliary_overlap=0,
        test_used=False,status_of_experiment='READY_NOT_RUN')
    (HERE/'local_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
