"""Parse notebook code and roundtrip payload; no notebook/model execution."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
from make_notebooks import HERE,sources


def main():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    parsed=[];payloads=[]
    for arm in ('CE','CROSS'):
        path=HERE/f'PHRASE_DECODER_{arm}_2GPU.ipynb';nb=json.loads(path.read_text(encoding='utf-8'));assign={};count=0
        for c in nb['cells']:
            if c['cell_type']!='code':continue
            count+=1;assert c['execution_count'] is None and c['outputs']==[]
            tree=ast.parse(''.join(c['source']))
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('PAYLOAD_B64','PAYLOAD_SHA','SOURCE_HASHES'):
                    assign[node.targets[0].id]=ast.literal_eval(node.value)
        raw=zlib.decompress(base64.b64decode(assign['PAYLOAD_B64'],validate=True));assert hashlib.sha256(raw).hexdigest()==assign['PAYLOAD_SHA']
        p=json.loads(raw);assert p['sources']==sources();assert p['manifest']==(HERE/'locked_manifest.json').read_text(encoding='utf-8')
        for name,source in p['sources'].items():ast.parse(source);assert hashlib.sha256(source.encode()).hexdigest()==assign['SOURCE_HASHES'][name]
        assert path.stat().st_size<900000;parsed.append(dict(arm=arm,bytes=path.stat().st_size,code_cells=count));payloads.append(p)
    assert payloads[0]==payloads[1],'Arms must embed exactly the same code and data'
    subprocess.run([sys.executable,HERE/'test_phrase_data.py'],check=True)
    subprocess.run([sys.executable,HERE/'test_comparison.py'],check=True)
    # Run the actual embedded data checks from an isolated folder, like notebook setup.
    with tempfile.TemporaryDirectory(prefix='phrase_data_check_',dir=HERE) as folder:
        root=Path(folder)
        for n,source in payloads[0]['sources'].items():(root/n).write_text(source,encoding='utf-8')
        (root/'locked_manifest.json').write_text(payloads[0]['manifest'],encoding='utf-8')
        subprocess.run([sys.executable,root/'test_phrase_data.py'],cwd=root,check=True)
    result=dict(status='LOCAL_DATA_AND_PACKAGING_PASS',notebooks=parsed,identical_embedded_code_and_manifest=True,
        local_data_tests_passed=3,local_synthetic_artifact_audit_tests_passed=3,embedded_data_checks='2passed;1parent-reconstruction test skipped when parent bundle absent',
        torch_model_tests_run_locally=False,DDP_or_GPU_tests_run_locally=False,notebooks_executed=False,
        required_kaggle_preflight='Synthetic CPU model3tests + two-T4 DDP smoke before real training',
        experiment_status='READY_NOT_RUN')
    (HERE/'LOCAL_VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
