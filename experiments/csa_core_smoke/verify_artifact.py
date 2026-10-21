"""No torch/model execution: verify portable notebook/source/hash contracts locally."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent


def main():
    notebook = json.loads((HERE/'CSA_CORE_SMOKE_2GPU.ipynb').read_text(encoding='utf-8'))
    assert (HERE/'CSA_CORE_SMOKE_2GPU.ipynb').stat().st_size < 1_000_000
    code_cells = [c for c in notebook['cells'] if c['cell_type']=='code']
    for cell in code_cells:
        assert cell['execution_count'] is None and cell['outputs'] == []
        ast.parse(''.join(cell['source']))
    tree = ast.parse(''.join(notebook['cells'][2]['source']))
    packed = next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    sources = json.loads(zlib.decompress(base64.b64decode(packed)))
    hashes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='SOURCE_HASHES' for t in n.targets))
    for name,source in sources.items():
        ast.parse(source,filename=name)
        assert hashlib.sha256(source.encode()).hexdigest() == hashes[name]
        if (HERE/name).is_file():
            assert (HERE/name).read_text(encoding='utf-8') == source
    source_files = list(HERE.glob('*.py'))
    for p in source_files:
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    # Independently compare declared asset hashes/shapes against the completed source audit.
    report = json.loads((HERE.parent/'csa_inheritance_audit/CSA_INHERITANCE_FEASIBILITY.json').read_text(encoding='utf-8'))
    manifest = {Path(r['path']).name:r['sha256'] for r in report['reference']['files']}
    core = ast.parse((HERE/'csa_core.py').read_text(encoding='utf-8'))
    literals = {n.targets[0].id:ast.literal_eval(n.value) for n in core.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('REFERENCE_SHA','BANKS')}
    assert literals['REFERENCE_SHA'] == manifest['csa_net.py']
    for name,(expected,shape) in literals['BANKS'].items():
        assert expected == manifest[name] and shape[1] == 1024
    # Reject accidental promotion/evaluation or training checkpoint writes in the worker.
    worker = next(n for n in ast.parse((HERE/'run_csa_smoke.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    calls = [n for n in ast.walk(worker) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('save','evaluate','predict') for n in calls)
    assert any(isinstance(n.func,ast.Name) and n.func.id=='range' and [ast.literal_eval(a) for a in n.args]==[1,5] for n in calls)
    print(json.dumps(dict(status='STATIC_PORTABLE_ARTIFACT_PASS',embedded_sources=len(sources),
         code_cells=len(code_cells),python_sources_parsed=len(source_files),asset_manifest_matches=True,
         torch_parity_run=False,gpu_smoke_run=False),indent=2))


if __name__=='__main__':
    main()
