"""Verify a received smoke ZIP using hashes and scalar receipts, never run a model."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARCHIVE=Path('C:/Users/leduc/Downloads/coco_routing_interface_smoke_review.zip')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        names=z.namelist();assert len(names)==len(set(names))
        expected={'protocol.json','stage_receipt.json','notebook.log',
            'complete_rank0.json','complete_rank1.json','numeric_rank0.json','numeric_rank1.json',
            'runtime_rank0.json','runtime_rank1.json','source/route_probe.py','source/run_route_smoke.py','source/train4_masks.json'}
        assert set(names)==expected,'Missing, extra or failure artifacts; inspect before accepting.'
        data={n:json.loads(z.read(n)) for n in names if n.endswith('.json')}
        cfg=data['protocol.json']; sources={n:z.read('source/'+n) for n in cfg['source_sha256']}
        log=z.read('notebook.log').decode('utf-8')
    # Compare both the protocol hash and the source actually delivered inside the notebook.
    notebook=json.loads((HERE/'COCO_ROUTING_INTERFACE_SMOKE_2GPU.ipynb').read_text(encoding='utf-8'))
    bundles=[]
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
                bundles.append(json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(node.args[0])))))
    assert len(bundles)==1
    for name,raw in sources.items():
        assert sha(raw)==cfg['source_sha256'][name],name
        assert raw.decode('utf-8')==bundles[0][name],name
    model_blob=subprocess.check_output(['git','show',cfg['project_commit']+':captioning/models.py'],cwd=ROOT)
    assert sha(model_blob)==cfg['models_source_sha256']
    assert cfg['project_commit']=='6acd3c9f0553d4d12d46684bbe7dcb8c38033d6b'
    assert (cfg['rtol'],cfg['atol'],cfg['route_epsilon'],cfg['optimizer_updates'])==(1e-5,1e-6,1e-8,0)
    assert not cfg['test_used'] and not cfg['dev_used'] and not cfg['caption_quality_evaluated']
    assert sha((HERE/'DATA_AUDIT.json').read_bytes())==cfg['data_audit_sha256']
    samples=json.loads(sources['train4_masks.json']);assert samples==cfg['samples']
    assert [r['coco_id'] for r in samples]==[64627,529017,468966,580507]
    receipt=data['stage_receipt.json']
    assert receipt['images']==4 and receipt['train_only'] and receipt['optimizer_updates']==0
    expected_inputs=dict(checkpoint='5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75',
        legacy_prompt='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d',
        dataset_json='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d')
    assert receipt['input_sha256']==expected_inputs
    states=set(); records=[]; resources=[]
    for rank in range(2):
        c=data[f'complete_rank{rank}.json']; runtime=data[f'runtime_rank{rank}.json']
        assert c['status']=='INTERFACE_SMOKE_PASS' and c['rank']==rank
        assert c['optimizer_updates']==c['parameters_added']==0
        assert c['before_state_sha256']==c['after_state_sha256'];states.add(c['before_state_sha256'])
        assert c['logits_max_abs_difference']==0 and c['future_noun_causal_check']
        assert not c['caption_quality_evaluated'] and not c['CLIP_backbone_loaded'] and not c['cached_CLIP_input_gradient']
        assert (c['rtol'],c['atol'])==(cfg['rtol'],cfg['atol'])
        assert len(c['gradients'])==4 and all(math.isfinite(v) and v>0 for v in c['gradients'].values())
        assert runtime['gpu']['name']=='Tesla T4' and runtime['packages']['transformers']=='4.57.1'
        assert runtime['deterministic_algorithms'] and not runtime['warn_only'] and runtime['sdp_math']
        assert not runtime['mha_fastpath'] and not runtime['sdp_flash'] and not runtime['sdp_mem_efficient']
        numerical=data[f'numeric_rank{rank}.json']
        assert [n['device'] for n in numerical]==['cpu',f'cuda:{rank}']
        assert all(n['status']=='SYNTHETIC_PASS' and n['public_mha_weights_checked'] and n['bool_and_float_padding_checked'] for n in numerical)
        subset=samples[rank::2];assert len(c['records'])==len(subset)==2
        supports=[]
        for r,s in zip(c['records'],subset):
            for key in ['coco_id','caption','raw_span','decoder_logit_positions','target_token_positions']:
                assert r[key]==s[key],key
            for q,p,cls,b in zip(r['decoder_q_mass'],r['patch_path_mass'],r['cls_path_mass'],r['bbox_mass']):
                assert all(math.isfinite(v) for v in [q,p,cls,b])
                assert 0<=b<=p+cfg['atol']<=q+2*cfg['atol']<=1+3*cfg['atol']
                assert math.isclose(q,p+cls,rel_tol=cfg['rtol'],abs_tol=cfg['atol'])
                supports.append(b)
            records.append(r)
        independently_computed=-sum(math.log(v+cfg['route_epsilon']) for v in supports)/len(supports)
        assert math.isclose(independently_computed,c['route_loss'],rel_tol=cfg['rtol'],abs_tol=cfg['atol'])
        resources.append({k:c[k] for k in ['rank','seconds','peak_allocated_GiB','peak_reserved_GiB']})
    assert len(states)==1
    assert log.count('INTERFACE SMOKE PASS; 0 updates; no caption-gain claim.')==2
    assert len({r['coco_id'] for r in records})==4
    result=dict(status='RECEIVED_TECHNICAL_SMOKE_PASS_SOURCE_AUDIT_PASS',archive_sha256=sha(ARCHIVE.read_bytes()),
        archive_bytes=ARCHIVE.stat().st_size,files=len(names),project_commit=cfg['project_commit'],
        source_sha256=cfg['source_sha256'],baseline_state_sha256=next(iter(states)),
        images=4,gpus=2,optimizer_updates=0,parameters_added=0,logits_max_abs_difference=0.,
        cpu_cuda_public_mha_checks_passed=True,future_noun_causal_checks_passed=True,
        gradients={str(r):data[f'complete_rank{r}.json']['gradients'] for r in range(2)},
        mass_records=records,resources=resources,runtime=data['runtime_rank0.json'],
        caption_quality_evaluated=False,root_model_run=False,
        limits=['No independent tensor replay: model/cache/stage tensors are absent from the small ZIP.',
            'Real greedy equality, memory/padding and gradient checks verified by matched worker source and receipts, not by local model execution.',
            'Four TRAIN examples / no optimizer updates; neither causal grounding nor caption gain is established.',
            'Observer differentiates a shared-parameter probability recomputation; baseline caption forward is unchanged.',
            'Worker seconds exclude cache preparation, hashing, model setup and synthetic checks; not full notebook duration.'])
    (HERE/'SMOKE_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cache=HERE/'review';cache.mkdir(exist_ok=True)
    (cache/'loaded_smoke.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    (cache/'notebook.log').write_text(log,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','archive_sha256','images','gpus','optimizer_updates','logits_max_abs_difference']}))
    print(log[-3500:])


if __name__=='__main__':main()
