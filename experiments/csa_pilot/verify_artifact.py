"""Local static/notebook contracts and deterministic build; no torch or model runs."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'CSA_TWO_PHASE_PILOT_2GPU.ipynb'
    raw=path.read_bytes()
    assert len(raw)<1_000_000
    nb=json.loads(raw)
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            ast.parse(''.join(cell['source']))
    tree=ast.parse(''.join(nb['cells'][2]['source']))
    packed=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    sources=json.loads(zlib.decompress(base64.b64decode(packed)))
    hashes=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='SOURCE_HASHES' for t in n.targets))
    assert set(hashes)==set(sources)
    for name,source in sources.items():
        ast.parse(source,filename=name)
        assert hashlib.sha256(source.encode()).hexdigest()==hashes[name]
        if (HERE/name).is_file():
            assert (HERE/name).read_text(encoding='utf-8')==source
    for name in ('csa_core.py','csa_model.py'):
        assert sources[name]==(HERE.parent/'csa_core_smoke'/name).read_text(encoding='utf-8')
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    # Pin the shared baseline/reward/runtime code to the successful smoke artifact.
    old=json.loads((HERE.parent/'csa_core_smoke/CSA_CORE_SMOKE_2GPU.ipynb').read_text(encoding='utf-8'))
    oldtree=ast.parse(''.join(old['cells'][2]['source']))
    oldhashes=next(ast.literal_eval(n.value) for n in oldtree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='SOURCE_HASHES' for t in n.targets))
    for name in set(oldhashes)&set(hashes):
        assert oldhashes[name]==hashes[name],name
    worker_source=sources['run_csa_pilot.py']
    worker_tree=ast.parse(worker_source)
    evaluate=next(n for n in worker_tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    assert ast.literal_eval(next(n.iter for n in ast.walk(evaluate) if isinstance(n,ast.For)
                                and isinstance(n.target,ast.Name) and n.target.id=='condition'))==('origin','on','off')
    assert "cfg['phase1_checkpoint']" in worker_source and 'phase1_final_equals_phase2_initial=True' in worker_source
    assert 'weights_only=False' in worker_source and '_use_new_zipfile_serialization=False' in worker_source
    assert 'checkpoint.pth' not in worker_source
    # Existing outputs must be protected, and a finally archive must be present.
    last=''.join(nb['cells'][4]['source'])
    assert any(isinstance(n,ast.Try) and n.finalbody for n in ast.walk(ast.parse(last)))
    assert "phases=(2,) if PHASE1_CHECKPOINT is not None else (1,2)" in last
    from test_policy_static import main as recipe_tests
    recipe_tests()
    print(json.dumps(dict(status='STATIC_PORTABLE_PILOT_PASS',bytes=len(raw),embedded_sources=len(sources),
                         inherited_smoke_sources_match=True,torch_tests_run=False,gpu_run=False),indent=2))


if __name__=='__main__':
    main()
