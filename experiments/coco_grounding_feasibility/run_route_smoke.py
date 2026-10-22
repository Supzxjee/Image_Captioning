"""Two independent GPU workers, four TRAIN images, zero optimizer updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

INPUTS=dict(checkpoint='5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75',
    legacy_prompt='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d',
    dataset_json='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d')


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def corpus(cfg):
    from captioning.tokenizer import CaptionTokenizer
    data=json.loads(Path(cfg['dataset_json']).read_text())
    tok=CaptionTokenizer([s['raw'] for r in data['images'] if r['split'] in ('train','restval') for s in r['sentences'][:5]])
    assert (tok.pad_idx,tok.bos_idx,tok.eos_idx)==(0,1,2)
    return data,tok


def prepare(cfg):
    import numpy as np
    import torch
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.visual_cache import VisualCache
    out=Path(cfg['output'])
    assert (cfg['rtol'],cfg['atol'],cfg['route_epsilon'],cfg['optimizer_updates'])==(1e-5,1e-6,1e-8,0)
    assert digest(Path(__file__).parent/'captioning/models.py')==cfg['models_source_sha256']
    for key,expected in INPUTS.items():
        print('Hashing pinned input:',key,flush=True)
        assert digest(cfg[key])==expected,'Pinned input differs: '+key
    for name,expected in cfg['source_sha256'].items():
        assert digest(Path(__file__).parent/name)==expected,name
    data,tok=corpus(cfg); samples=cfg['samples'];assert len(samples)==4
    assert len({r['coco_id'] for r in samples})==4
    for r in samples:
        original=data['images'][r['karpathy_imgid']]
        assert original['split'] in ('train','restval') and original['cocoid']==r['coco_id']
        assert original['filename']==r['filename']
        assert original['sentences'][r['caption_index']]['raw']==r['caption']
    names=[str(Path('/images')/data['images'][r['karpathy_imgid']]['filepath']/r['filename']) for r in samples]
    print('Loading legacy cache once; only four TRAIN rows will be staged.',flush=True)
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names);assert len(legacy)==4
    del bundle
    cache=VisualCache([cfg['visual_cache']],id_key='coco_id')
    rows=[]
    try:
        cache.require_ids([r['coco_id'] for r in samples],'routing smoke TRAIN4')
        for r,name in zip(samples,names):
            ids,padding=tok.encode(r['caption']); positions=r['decoder_logit_positions']
            assert positions and max(positions)<len(ids)-1
            assert r['target_token_positions']==[p+1 for p in positions]
            assert [int(ids[p+1]) for p in positions]==r['token_ids']
            assert [tok.word2idx.get(w,3) for w in r['raw_span']]==r['token_ids']
            assert all(int(ids[p+1]) not in (0,1,2,3) for p in positions)
            old=legacy[name];t=old['tokens'].float();m=old['mask'].long()
            assert t.ndim==2 and t.shape[-1]==512 and m.shape==t.shape[:1]
            assert bool(torch.isfinite(t).all()) and bool(((m==0)|(m==1)).all())
            region=torch.tensor(r['patch_area_fraction'],dtype=torch.float32).flatten()
            assert region.shape==(196,) and bool(((region>=0)&(region<=1)).all()) and float(region.sum())>0
            visual=cache.read(r['coco_id']); assert visual.shape==(197,768) and np.isfinite(visual).all()
            rows.append(dict(visual=torch.from_numpy(visual),legacy=t,legacy_mask=m,caption=ids,bbox_mask=region))
            print('Staged TRAIN COCO',r['coco_id'],'noun',r['raw_span'],'prediction positions',positions,flush=True)
    finally: cache.close()
    torch.save({k:torch.stack([r[k] for r in rows]) for k in rows[0]},out/'train4.pt')
    save(out/'stage_receipt.json',dict(stage_sha256=digest(out/'train4.pt'),input_sha256=INPUTS,
        vocab_sha256=hashlib.sha256(json.dumps(tok.word2idx,sort_keys=True).encode()).hexdigest(),
        train_only=True,images=4,optimizer_updates=0,model_inference_during_prepare=False,
        visual_scope='Selected cache rows only; source HDF5 bytes/provenance not independently pinned.'))


def build(cfg,tok):
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.checkpoints import load_checkpoint
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',
        num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(tok.word2idx)))
    load_checkpoint(base,cfg['checkpoint'],tok)
    assert base.encoder.feature_extractor is None
    return base.cuda().eval()


def teacher(model,batch):
    memory,padding=model.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
    ids=batch['caption'][:,:-1]
    logits=model.decoder(ids,memory,tgt_key_padding_mask=ids.eq(0),memory_key_padding_mask=padding)
    return logits,memory,padding


def greedy(model,batch):
    import torch
    memory,padding=model.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
    ids=torch.ones((len(memory),1),device=memory.device,dtype=torch.long)
    done=torch.zeros(len(memory),device=memory.device,dtype=torch.bool)
    for _ in range(29):
        logits=model.decoder(ids,memory,tgt_key_padding_mask=ids.eq(0),memory_key_padding_mask=padding)
        nxt=logits[:,-1].argmax(-1);nxt=torch.where(done,0,nxt)
        ids=torch.cat((ids,nxt[:,None]),1);done|=nxt.eq(2)
        if bool(done.all()): break
    return ids


def worker(cfg):
    import torch
    from captioning.scst_runtime import configure_strict_runtime,runtime_manifest,model_digest
    from route_probe import RouteProbe,route,numeric_checks,RTOL,ATOL
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output'])
    assert int(os.environ['WORLD_SIZE'])==2
    configure_strict_runtime(True);torch.cuda.set_device(rank);torch.set_num_threads(2)
    assert torch.cuda.get_device_name(rank)=='Tesla T4','Use T4x2.'
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    save(out/f'runtime_rank{rank}.json',runtime_manifest())
    checks=[numeric_checks(torch.device('cpu')),numeric_checks(torch.device('cuda',rank))]
    save(out/f'numeric_rank{rank}.json',checks)
    data,tok=corpus(cfg);del data
    assert digest(cfg['checkpoint'])==INPUTS['checkpoint']
    receipt=json.loads((out/'stage_receipt.json').read_text())
    assert digest(out/'train4.pt')==receipt['stage_sha256']
    assert hashlib.sha256(json.dumps(tok.word2idx,sort_keys=True).encode()).hexdigest()==receipt['vocab_sha256']
    staged=torch.load(out/'train4.pt',map_location='cpu',weights_only=True)
    batch={k:v[rank::2].cuda() for k,v in staged.items()}; del staged
    samples=cfg['samples'][rank::2]; assert len(samples)==2
    model=build(cfg,tok);before=model_digest(model)
    params=set(model.state_dict());parameter_ids={n:id(p) for n,p in model.named_parameters()}
    assert all(not m.training for m in model.modules())
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    with torch.no_grad():
        expected,expected_memory,expected_pad=teacher(model,batch)
        expected_greedy=greedy(model,batch)
    print(f'[rank {rank}] Baseline logits/greedy captured; enabling observation only.',flush=True)
    with RouteProbe(model) as probe:
        with torch.no_grad():
            actual,memory,pad=teacher(model,batch)
            torch.testing.assert_close(memory,expected_memory,rtol=RTOL,atol=ATOL)
            assert torch.equal(pad,expected_pad)
            torch.testing.assert_close(actual,expected,rtol=RTOL,atol=ATOL)
            max_delta=float((actual-expected).abs().max())
            observed_greedy=greedy(model,batch)
            assert torch.equal(observed_greedy,expected_greedy),'Greedy parity failed; do not relax tolerance.'
        probe.clear();model.zero_grad(set_to_none=True)
        actual,memory,pad=teacher(model,batch)
        torch.testing.assert_close(actual,expected,rtol=RTOL,atol=ATOL)
        dw,cw=probe.weights['decoder'],probe.weights['qformer']
        assert dw.requires_grad and cw.requires_grad
        dw.retain_grad();cw.retain_grad()
        legacy=batch['legacy'].shape[1]
        assert memory.shape[1]==legacy+32 and cw.shape[-2:]==(32,197)
        for weights in (dw,cw):
            torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights.sum(-1)),rtol=RTOL,atol=ATOL)
        assert bool((dw.masked_select(pad[:,None,None,:].expand_as(dw))==0).all())
        parts=route(dw,cw,legacy,batch['bbox_mask'])
        full=route(dw,cw,legacy,torch.ones_like(batch['bbox_mask']))
        empty=route(dw,cw,legacy,torch.zeros_like(batch['bbox_mask']))
        torch.testing.assert_close(full['bbox_mass'],parts['patch_path_mass'],rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(parts['decoder_q_mass'],parts['patch_path_mass']+parts['cls_path_mass'],rtol=RTOL,atol=ATOL)
        assert bool((empty['bbox_mass']==0).all())
        assert bool((parts['bbox_mass']<=parts['patch_path_mass']+ATOL).all())
        assert bool((parts['patch_path_mass']<=parts['decoder_q_mass']+ATOL).all())
        selected=[];records=[]
        for i,r in enumerate(samples):
            pos=r['decoder_logit_positions']; selected.append(parts['bbox_mass'][i,pos])
            assert [int(batch['caption'][i,p+1]) for p in pos]==r['token_ids']
            records.append(dict(coco_id=r['coco_id'],caption=r['caption'],raw_span=r['raw_span'],
                decoder_logit_positions=pos,target_token_positions=r['target_token_positions'],
                decoder_q_mass=parts['decoder_q_mass'][i,pos].detach().cpu().tolist(),
                patch_path_mass=parts['patch_path_mass'][i,pos].detach().cpu().tolist(),
                cls_path_mass=parts['cls_path_mass'][i,pos].detach().cpu().tolist(),
                bbox_mass=parts['bbox_mass'][i,pos].detach().cpu().tolist()))
        loss=-(torch.cat(selected)+1e-8).log().mean()
        assert torch.isfinite(loss);loss.backward()
        gradients={}
        for name in ('decoder.transformer_decoder.layers.2.multihead_attn.in_proj_weight',
                     'encoder.qformer.layers.1.cross_attn.in_proj_weight','encoder.vis_projection.weight',
                     'encoder.qformer.query_tokens'):
            p=dict(model.named_parameters())[name]
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0,name
            gradients[name]=float(p.grad.norm())
        assert dw.grad is not None and cw.grad is not None
        assert bool(torch.isfinite(dw.grad).all()) and bool(torch.isfinite(cw.grad).all())
        assert model.decoder.fc_out.weight.grad is None,'No caption CE is used in this smoke.'
        assert batch['visual'].grad is None and not batch['visual'].requires_grad
        # Changing the noun/future input must not affect its own prediction query.
        with torch.no_grad():
            changed=dict(batch);changed['caption']=batch['caption'].clone()
            for i,r in enumerate(samples):
                p=min(r['decoder_logit_positions'])
                changed['caption'][i,p+1:]=torch.where(changed['caption'][i,p+1:].eq(0),0,3)
            later,_,_=teacher(model,changed)
            later_weights=probe.weights['decoder']
            for i,r in enumerate(samples):
                p=min(r['decoder_logit_positions'])
                torch.testing.assert_close(later[i,p],actual[i,p],rtol=RTOL,atol=ATOL)
                torch.testing.assert_close(later_weights[i,:,p],dw[i,:,p],rtol=RTOL,atol=ATOL)
    assert params==set(model.state_dict())
    assert parameter_ids=={n:id(p) for n,p in model.named_parameters()}
    after=model_digest(model);assert after==before,'Weights changed without an optimizer.'
    save(out/f'complete_rank{rank}.json',dict(status='INTERFACE_SMOKE_PASS',rank=rank,
        optimizer_updates=0,parameters_added=0,before_state_sha256=before,after_state_sha256=after,
        logits_max_abs_difference=max_delta,rtol=RTOL,atol=ATOL,greedy_token_ids=expected_greedy.cpu().tolist(),
        caption_quality_evaluated=False,future_noun_causal_check=True,gradients=gradients,records=records,
        route_loss=float(loss.detach()),CLIP_backbone_loaded=False,cached_CLIP_input_gradient=False,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,seconds=time.monotonic()-begin,
        scope='Observation recomputes probabilities with shared Q/K weights; original MHA calls/output unchanged. Routing is not causal attribution.'))
    print(f'[rank {rank}] INTERFACE SMOKE PASS; 0 updates; no caption-gain claim.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker']);parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    try:
        (prepare if args.mode=='prepare' else worker)(cfg)
    except Exception:
        tag=os.environ.get('LOCAL_RANK','prepare')
        save(Path(cfg['output'])/f'failure_{tag}.json',dict(status='FAIL',traceback=traceback.format_exc(),optimizer_updates=0))
        raise
