"""Validate the user-run technical smoke ZIP without loading model tensors."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
import zlib

HERE = Path(__file__).resolve().parent


def audit(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        payload = {n:z.read(n) for n in z.namelist()}
    parsed = {n:json.loads(v) for n,v in payload.items() if n.endswith('.json')}
    cfg = parsed['protocol.json']
    nb = json.loads((HERE/'CSA_CORE_SMOKE_2GPU.ipynb').read_text(encoding='utf-8'))
    tree = ast.parse(''.join(nb['cells'][2]['source']))
    packed = next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    expected = json.loads(zlib.decompress(base64.b64decode(packed)))
    assert set(cfg['implementation_sha256']) == set(expected)
    for name,text in expected.items():
        raw = payload['source/'+name]
        assert hashlib.sha256(raw).hexdigest() == cfg['implementation_sha256'][name],name
        assert raw.decode('utf-8') == text,name
    assert hashlib.sha256(payload['reference/csa_net.py']).hexdigest() == 'd6bc6d8072c03cb7d42bc8ffcc48e48cefb03b1b128e081b608902d845b6ded1'
    assert cfg['project_commit'] == 'dfa1d4e0c52bea8672c6b770127034c2ec377b58'
    assert cfg['csa_commit'] == '0cc570ba7c50ef08c46a21439fe73ef0a5df97ab'
    assert (cfg['train_images'],cfg['global_batch'],cfg['steps_CE'],cfg['steps_SCST']) == (8,8,2,2)
    assert cfg['probe_only'] and not cfg['dev_used'] and not cfg['test_used']
    parity = parsed['reference/parity.json']
    assert parity['status'] == 'CPU_PARITY_AND_INTERFACE_PASS'
    assert parity['tolerance'] == dict(rtol=1e-4,atol=1e-6)
    assert [r['tokens'] for r in parity['cases']] == [32,150]
    assert all(r['output_max_abs'] == [0.,0.] and r['parameter_gradients_compared'] == 93 for r in parity['cases'])
    stage = parsed['stage_receipt.json']
    assert stage['train_only'] and stage['no_dev_or_test'] and stage['images'] == 8
    assert stage['checkpoint_sha256'] == '5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75'
    rows = parsed['train8.json']
    assert len(rows) == len({r['coco_id'] for r in rows}) == 8
    # Validate real Karpathy split and deterministic sampling, not filename train/val prefix.
    data = json.loads(Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json').read_bytes()).hexdigest() == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    selected = sorted((i for i,r in enumerate(data['images']) if r['split'] in ('train','restval')),
                      key=lambda i:hashlib.sha256(('upt-alignment-pilot-v1:train:'+data['images'][i]['filename']).encode()).hexdigest())[:8]
    assert sorted(selected) == [r['image_id'] for r in rows]
    for r in rows:
        original = data['images'][r['image_id']]
        assert original['filename'] == r['filename'] and original['split'] in ('train','restval')
        assert int(Path(r['filename']).stem.rsplit('_',1)[1]) == r['coco_id']
    assert [r['image_id'] for r in parsed['object_targets.json']['data']] == [r['image_id'] for r in rows]
    histories = [parsed[f'history_rank{r}.json'] for r in range(2)]
    complete = [parsed[f'complete_rank{r}.json'] for r in range(2)]
    for r in range(2):
        c,h = complete[r],histories[r]
        assert c['status'] == 'TECHNICAL_SMOKE_PASS' and c['completed_updates'] == 4
        assert c['bank_buffers_unchanged'] and c['real_baseline_bypass_pass']
        assert c['core_parameters'] == 23161644 and not c['dev_used'] and not c['test_used']
        assert [s['step'] for s in h] == [1,2,3,4]
        assert [s['mode'] for s in h] == ['CE','CE','CE_SCST_OBJECT','CE_SCST_OBJECT']
        for s in h:
            assert all(math.isfinite(v) for v in s.values() if isinstance(v,(int,float)))
            assert s['seconds'] > 0 and s['core_grad_norm'] > 0
            w = 0 if s['step'] <= 2 else .05
            assert abs(s['loss']-(s['ce']+w*s['scst'])) < 1e-6
    assert complete[0]['ddp_core_hash'] == complete[1]['ddp_core_hash']
    assert parsed['runtime_rank0.json'] == parsed['runtime_rank1.json']
    assert parsed['runtime_rank0.json']['gpu']['name'] == 'Tesla T4'
    log = payload['notebook.log'].decode('utf-8')
    assert 'Traceback (most recent call last)' not in log
    steps = [s for h in histories for s in h]
    result = dict(status='TECHNICAL_SMOKE_REVIEW_AUDIT_PASS',date='2026-10-22',
        archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),archive_bytes=Path(path).stat().st_size,
        source_files_verified=len(expected),reference_sha256_verified=True,parity=parity,
        gpu='Tesla T4 x2',core_parameters=23161644,images=8,updates=4,global_batch=8,
        peak_allocated_gib=max(s['peak_allocated_bytes'] for s in steps)/2**30,
        peak_reserved_gib=max(s['peak_reserved_bytes'] for s in steps)/2**30,
        ce_warm_step_seconds=max(h[1]['seconds'] for h in histories),
        scst_step_seconds=[max(histories[r][i]['seconds'] for r in range(2)) for i in (2,3)],
        synchronized_core_sha256=complete[0]['ddp_core_hash'],baseline_bypass_pass=True,
        gradients_finite_nonzero=True,clip_active_rank_steps=sum(s['global_grad_norm']>1 for s in steps),
        dev_used=False,test_used=False,caption_gain_evaluated=False,
        limits=['Reference and adapter share unchanged author forward; not independent implementation parity.',
                'Only8train images repeated4steps; loss drop is not generalization or grounding evidence.',
                'Peak/time excludes setup, fullTRAIN IDF, staging and evaluation; not a worst-case memory bound.',
                'Bank/source/checkpoint/stage validation enforced by matched worker code; bank/checkpoint/stage bytes absent from reviewZIP.',
                'Global clipping active8/8ranksteps; reported core norm is preclip.',
                'Log has unrelated gradio/diffusers dependency warnings and DDP unused-parameter overhead warning; smoke completed.'])
    return result


if __name__=='__main__':
    result = audit(sys.argv[1])
    (HERE/'SMOKE_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
