"""Data/packaging checks only. Does not run notebook, GPU inference or model tests."""
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
from make_notebook import HERE,ROOT,sources


def main():
    path=HERE/'RELATION_CONDITIONING_DIAGNOSTIC_20_2GPU.ipynb';nb=json.loads(path.read_text(encoding='utf-8'))
    assigned={};code_count=0
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        code_count+=1;assert cell['outputs']==[] and cell['execution_count'] is None
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('PAYLOAD_B64','PAYLOAD_SHA','SOURCE_HASHES'):
                assigned[node.targets[0].id]=ast.literal_eval(node.value)
    assert path.stat().st_size<900000
    raw=zlib.decompress(base64.b64decode(assigned['PAYLOAD_B64'],validate=True))
    assert hashlib.sha256(raw).hexdigest()==assigned['PAYLOAD_SHA'];payload=json.loads(raw)
    assert payload['sources']==sources() and payload['manifest']==(HERE/'cases_manifest.json').read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='oracle_data_check_',dir=HERE) as directory:
        temp=Path(directory)
        for name,source in payload['sources'].items():
            ast.parse(source);assert hashlib.sha256(source.encode()).hexdigest()==assigned['SOURCE_HASHES'][name]
            (temp/name).write_text(source,encoding='utf-8')
        (temp/'cases_manifest.json').write_text(payload['manifest'],encoding='utf-8')
        subprocess.run([sys.executable,temp/'test_oracle_data.py'],cwd=temp,check=True)
    fit=json.loads((ROOT/'experiments/verified_same_image/verified_manifest.json').read_text(encoding='utf-8'))['splits']['fit']
    action_pairs=Counter('|'.join(sorted((r['source_positive']['action'],r['source_negative']['action']))) for r in fit if r['task']=='wrong_predicate')
    assert action_pairs['carry|hold']==30 and sum(action_pairs.values())==80
    result=dict(status='LOCAL_DATA_AND_PACKAGING_PASS',notebook_bytes=path.stat().st_size,code_cells_parsed=code_count,
        data_tests_passed=3,model_tests_run_locally=False,gpu_inference_run=False,notebook_executed=False,
        payload_sha256=assigned['PAYLOAD_SHA'],manifest_sha256=hashlib.sha256(payload['manifest'].encode()).hexdigest(),
        fit_predicate_action_pair_counts=dict(action_pairs),note='Action overlap is a semantic validity concern, not proof all such annotation labels are erroneous',
        status_of_experiment='READY_NOT_RUN')
    (HERE/'local_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
