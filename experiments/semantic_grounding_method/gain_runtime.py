"""Kaggle-only preparation/training. Importing this module starts no work."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
from gain_data import *


def prepare(cfg):
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets
    from gain_model import GainHead
    start=time.monotonic();out=Path(cfg['prepared_output'])
    out.mkdir(parents=True,exist_ok=False)
    meta=load_metadata(cfg['metadata']);groups=meta['groups']
    for key,h in [('checkpoint',A_SHA),('legacy_prompt',LEGACY_SHA),('dataset_json',DATA_SHA),('detections',DETECTION_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if sha(cfg[key])!=h:raise ValueError('Wrong pinned input: '+key)
    if sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Wrong CHAIR annotations')
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    if state['caption_word2idx']!=meta['word2idx']:raise ValueError('A vocabulary differs')
    del state
    targets=make_targets(groups,data,Path(cfg['chair'])/'chair.pkl',json.loads(Path(cfg['detections']).read_text()),'aliases_v2','unique_count_v2')
    save(out/'object_targets.json',targets)
    rows=groups['train']+groups['val'];names=[r['filename'] for r in rows]
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names);del bundle
    if set(legacy)!=set(names):raise ValueError('Original prompts incomplete')
    cache=VisualCache([cfg['visual_cache']]);cache.require_ids([r['coco_id'] for r in rows],'grounded gain')
    stage=out/'inputs.h5'
    if shutil.disk_usage(out).free<3*1024**3:raise RuntimeError('Need3GiB free for reusable exact inputs')
    checked=set()
    try:
        with h5py.File(stage,'w') as f:
            for name,shape,dtype in [('visual',(197,768),'f4'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:
                f.create_dataset(name,(3000,*shape),dtype=dtype)
            f.create_dataset('coco_ids',data=[r['coco_id'] for r in rows])
            for count,i in enumerate(sorted(range(3000),key=lambda j:cache.index[rows[j]['coco_id']]),1):
                r=rows[i];fi,_=cache.index[r['coco_id']]
                if fi not in checked:
                    with h5py.File(cache.paths[fi],'r') as source:
                        info=json.loads(source.attrs['metadata_json'])
                        expected=dict(dataset_json_sha256=DATA_SHA,image_size=[224,224],preprocessing='bilinear',
                            model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',
                            mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711],antialias=True,compute_dtype='float32',autocast=False)
                        if any(info.get(k)!=v for k,v in expected.items()):raise ValueError('Original visual metadata differs')
                    checked.add(fi)
                entry=legacy[r['filename']];p=entry['tokens'].float().numpy();m=entry['mask'].numpy()
                if p.shape!=(20,512) or m.shape!=(20,) or not np.isfinite(p).all() or not np.isin(m,[0,1]).all():raise ValueError('Invalid original prompt')
                f['visual'][i]=cache.read(r['coco_id']);f['prompt'][i]=p;f['mask'][i]=m
                if count%100==0:print(f'Copy exact cached inputs: {count}/3000, {time.monotonic()-start:.1f}s; no feature extraction.',flush=True)
                if time.monotonic()-start>POLICY['prepare_seconds']:raise RuntimeError('Preparation cap exceeded')
            f.attrs['complete']=True
    finally:cache.close()
    torch.manual_seed(42);head=GainHead();torch.save(head.state_dict(),out/'initial_head.bin')
    shutil.copyfile(cfg['metadata'],out/'gain_metadata.json.gz')
    receipt=dict(status='PREPARATION_COMPLETE',metadata_sha256=sha(cfg['metadata']),policy=POLICY,
        inputs=dict(checkpoint=A_SHA,legacy_prompt=LEGACY_SHA,dataset_json=DATA_SHA,detections=DETECTION_SHA),
        payload_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},seconds=time.monotonic()-start,
        source_sha256=cfg['sources'],feature_extraction=False,caption_training=False)
    if receipt['seconds']>POLICY['prepare_seconds']:raise RuntimeError('Preparation cap exceeded during payload hashing')
    save(out/'receipt.json',receipt)
    print('KEEP ENTIRE PREPARATION FOLDER AS DATASET:',out,flush=True)


def build(cfg,device):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.scst_runtime import model_digest
    from gain_model import GainHead,Captioner
    meta=load_metadata(cfg['metadata']);state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    if sha(cfg['checkpoint'])!=A_SHA or state['caption_word2idx']!=meta['word2idx']:raise ValueError('Wrong checkpoint/vocabulary')
    weights=state['model_state_dict']
    if not weights or any(not n.startswith('base.') for n in weights):raise ValueError('Wrong A state wrapper')
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(10000))
    base.load_state_dict({n[5:]:v for n,v in weights.items()},strict=True)
    if model_digest(base)!=BASE_SHA:raise ValueError('A tensor identity differs')
    head=GainHead();head.load_state_dict(torch.load(Path(cfg['prepared'])/'initial_head.bin',map_location='cpu',weights_only=True),strict=True)
    model=Captioner(base,head).to(device).eval()
    if sum(p.numel() for p in head.parameters())!=262144:raise AssertionError('Wrong branch parameter count')
    return model,meta


def tokenizer(meta):
    from captioning.tokenizer import CaptionTokenizer
    tok=CaptionTokenizer([]);tok.word2idx=meta['word2idx'];tok.idx2word={i:w for w,i in tok.word2idx.items()};return tok


def batch(f,indices,device):
    import torch
    if isinstance(device,int):device=f'cuda:{device}'
    return {k:torch.tensor(np.stack([f[k][int(i)] for i in indices]),device=device) for k in ('visual','prompt','mask')}


def sync_gradients(head,world):
    import torch
    import torch.distributed as dist
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in head.parameters()])
    dist.all_reduce(flat);flat/=world;offset=0
    for p in head.parameters():
        p.grad=flat[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()


def diagnostics(result):
    return {k:float(result[k].detach()) for k in ('gain','cancel','direction','rank','delta','cg','cu','residual','rank_defined')}


def predict(model,f,meta,tok,rank,baseline):
    from captioning.scst import decode
    rows=meta['groups']['val'];pred=[];device=next(model.head.parameters()).device
    for first in range(0,500,16):
        indices=list(range(rank,1000,2))[first:first+16]
        b=batch(f,[2000+i for i in indices],device);mem,pad,z=model.memory(b)
        seq,_=model.generate(mem,pad,z,tok,baseline=baseline)
        for i,text in zip(indices,decode(seq,tok)):pred.append(dict(image_id=rows[i]['image_id'],coco_id=rows[i]['coco_id'],caption=text))
        if first%128==0:print(f'Rank{rank} {"A" if baseline else "end"} greedy: {min(first+16,500)}/500',flush=True)
    return pred


def preflight(model,f,meta,tok,rank,world):
    import torch
    from gain_model import check_completeness
    from captioning.scst_runtime import model_digest
    b=batch(f,list(range(rank*4,rank*4+4)),rank);mem,pad,z=model.memory(b)
    ids=torch.full((4,1),tok.bos_idx,dtype=torch.long,device=mem.device)
    h,base=model.states(ids,mem,pad)
    direct=model.base.decoder(ids,mem,memory_key_padding_mask=pad)
    torch.testing.assert_close(base,direct,atol=1e-5,rtol=1e-4)
    new,_=model.logits(ids,mem,pad,z)
    torch.testing.assert_close(new,base,atol=0,rtol=0)
    a,_=model.generate(mem,pad,z,tok,baseline=True);s,_=model.generate(mem,pad,z,tok)
    if not torch.equal(a,s):raise AssertionError('Zero-init greedy mismatch')
    from test_gain_model import check_distributed_head
    check_distributed_head(rank,world)
    initial={n:p.detach().clone() for n,p in model.head.state_dict().items()};rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(rank)
    old=model_digest(model.head)
    try:
        torch.manual_seed(734)
        with torch.no_grad():model.head.o.weight.normal_(0,.02)
        with torch.no_grad():
            seq,active,trace=model.generate(mem,pad,z,tok,sample=True,return_trace=True)
            logits,_=model.logits(seq[:,:-1],mem,pad,z)
            rescored=logits.log_softmax(-1).gather(2,seq[:,1:,None]).squeeze(-1)
            torch.testing.assert_close(trace[active],rescored[active],atol=1e-5,rtol=1e-4)
        found=False
        for index,row in enumerate(meta['groups']['train']):
            groups=meta['auxiliary'].get(str(row['coco_id']),[])
            if not groups:continue
            bb=batch(f,[index],rank);m,p,zz=model.memory(bb);g=groups[0]
            r64=model.auxiliary(g,m,p,zz,64);check_completeness(r64)
            grads64=torch.autograd.grad(r64['cg'],tuple(model.head.parameters()),allow_unused=False)
            r128=model.auxiliary(g,m,p,zz,128);check_completeness(r128)
            grads128=torch.autograd.grad(r128['cg'],tuple(model.head.parameters()),allow_unused=False)
            for key in ('components','cg'):
                torch.testing.assert_close(r64[key],r128[key],atol=1e-5,rtol=1e-4)
            v64=torch.cat([t.flatten() for t in grads64]);v128=torch.cat([t.flatten() for t in grads128])
            if not torch.isfinite(v64).all() or not torch.isfinite(v128).all() or not torch.linalg.vector_norm(v128)>0:raise AssertionError('Nonzero finite gradient fixture required')
            if torch.linalg.vector_norm(v64-v128)>1e-5+1e-4*torch.linalg.vector_norm(v128):raise AssertionError('64/128 gradient comparison failed')
            found=True;break
        if not found:raise AssertionError('No actual auxiliary fixture')
    finally:
        model.head.load_state_dict(initial);model.head.zero_grad(set_to_none=True)
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng,rank)
    if model_digest(model.head)!=old:raise AssertionError('Preflight changed initialization')
    return dict(status='PASS',zero_init_parity=True,nonzero_sample_rescore_parity=True,quadrature_gradient_fixture=True,distributed_head_parity=True)


def worker(cfg,checking=False):
    import h5py
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from captioning.scst import TrainingCider,policy_loss,decode
    from captioning.object_reward import TrainingObjectReward,combine_rewards
    from gain_model import check_completeness
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('Requires T4x2')
    torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True)
    dist.init_process_group('nccl');out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    try:
        if sha(cfg['dataset_json'])!=DATA_SHA or sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:
            raise ValueError('Wrong reward/evaluation inputs; stop before training')
        # One rank hashes the shared mount; broadcast even a failure to avoid a hung peer.
        verification=[None]
        if rank==0:
            try:verification[0]=dict(receipt=checked_cache(cfg['prepared'],sha(cfg['metadata'])),error=None)
            except Exception as error:verification[0]=dict(receipt=None,error=repr(error))
        dist.broadcast_object_list(verification,src=0)
        if verification[0]['error']:raise ValueError(verification[0]['error'])
        receipt=verification[0]['receipt']
        if receipt['source_sha256']!=cfg['sources']:raise ValueError('Preparation/runtime sources differ')
        model,meta=build(cfg,f'cuda:{rank}');tok=tokenizer(meta)
        with h5py.File(Path(cfg['prepared'])/'inputs.h5','r') as f:
            expected=[r['coco_id'] for split in ('train','val') for r in meta['groups'][split]]
            if not f.attrs.get('complete') or f['coco_ids'][:].tolist()!=expected:raise ValueError('Stage identity/completion differs')
            if checking:
                result=preflight(model,f,meta,tok,rank,world);save(out/f'preflight_rank{rank}.json',result);return
            if not all((out/f'preflight_rank{r}.json').is_file() for r in (0,1)):raise ValueError('Run successful two-rank preflight first')
            initial_base=model_digest(model.base);initial_head=model_digest(model.head)
            save(out/f'runtime_rank{rank}.json',runtime_manifest())
            baseline=predict(model,f,meta,tok,rank,True)
            expected_a={r['image_id']:r['caption'] for r in meta['baseline']}
            if any(expected_a[r['image_id']]!=r['caption'] for r in baseline):raise AssertionError('A captions drifted from locked baseline; stop before training')
            save(out/f'A_rank{rank}.json',baseline)
            data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
            references={i:[s['raw'] for s in r['sentences'][:5]] for i,r in enumerate(data['images']) if r['split'] in ('train','restval')}
            print(f'Rank{rank}: fixed full TRAIN/train+restval IDF: {len(references)} images',flush=True)
            reward=TrainingCider(references,reference_cache_size=256)
            target=json.loads((Path(cfg['prepared'])/'object_targets.json').read_text())
            obj=TrainingObjectReward(target['data'],'aliases_v2','unique_count_v2')
            opt=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
            torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
            history=out/f'history_rank{rank}.jsonl';step=0;start=time.monotonic()
            for epoch in range(2):
                for all_indices in schedule(meta,epoch):
                    step+=1;indices=all_indices[rank*4:(rank+1)*4].tolist();b=batch(f,indices,rank);memory,padding,z=model.memory(b)
                    sample,active=model.generate(memory,padding,z,tok,sample=True);greedy,_=model.generate(memory,padding,z,tok)
                    image_ids=[meta['groups']['train'][i]['image_id'] for i in indices]
                    st,gt=decode(sample,tok),decode(greedy,tok)
                    sr=combine_rewards(reward.score(image_ids,st),obj.score(image_ids,st)[0],.5)
                    gr=combine_rewards(reward.score(image_ids,gt),obj.score(image_ids,gt)[0],.5)
                    advantage=torch.tensor(sr-gr,device=memory.device)
                    logits,base=model.logits(sample[:,:-1],memory,padding,z)
                    rl=policy_loss(logits,sample[:,1:],active,advantage)
                    divergence=(base.log_softmax(-1).exp()*(base.log_softmax(-1)-logits.log_softmax(-1))).sum(-1)
                    valid=active.sum().to(torch.float32);total_valid=valid.detach().clone();dist.all_reduce(total_valid)
                    kl=(divergence*active).sum()*world/total_valid.clamp_min(1)
                    aux=logits.sum()*0;records=[]
                    groups=[]
                    for bi,i in enumerate(indices):
                        available=meta['auxiliary'].get(str(meta['groups']['train'][i]['coco_id']),[])
                        if available:groups.append((bi,available[epoch%len(available)]))
                    count=torch.tensor(float(len(groups)),device=memory.device);dist.all_reduce(count)
                    for bi,g in groups:
                        r=model.auxiliary(g,memory[bi:bi+1],padding[bi:bi+1],z[bi:bi+1])
                        if step<=10 or step%25==0:check_completeness(r)
                        value=r['gain']+r['cancel']
                        if cfg['arm']=='H':value=value+r['rank']
                        if cfg['arm']=='T':value=value+r['direction']
                        aux=aux+value*world/count.clamp_min(1);records.append(diagnostics(r))
                    loss=rl+.1*kl+.1*aux
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    opt.zero_grad(set_to_none=True);loss.backward();sync_gradients(model.head,world)
                    if any(not torch.isfinite(p.grad).all() for p in model.head.parameters()):raise FloatingPointError('Nonfinite averaged gradient')
                    if step==1 and not model.head.o.weight.grad.abs().sum()>0:raise AssertionError('No initial Wo gradient')
                    if step==2 and any(not p.weight.grad.abs().sum()>0 for p in (model.head.q,model.head.k,model.head.v)):
                        raise AssertionError('No subsequent q/k/v gradient; stop before expensive training')
                    norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);opt.step()
                    row=dict(step=step,epoch=epoch,loss=float(loss.detach()),scst=float(rl.detach()),kl=float(kl.detach()),aux=float(aux.detach()),
                        sample_reward=float(sr.mean()),greedy_reward=float(gr.mean()),grad_norm=float(norm),auxiliary=records,seconds=time.monotonic()-start)
                    with history.open('a',encoding='utf-8') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                    if step<=10 or step%25==0:
                        torch.cuda.synchronize();print(f'[{cfg["arm"]} rank{rank}] '+json.dumps(row),flush=True)
                    if step==10:
                        # Parent enforces hard cap; this is an early conservative feasibility stop.
                        forecast=(time.time()-cfg['arm_started_unix'])+(time.monotonic()-start)/10*490+300
                        if forecast>POLICY['arm_seconds']:raise RuntimeError('Throughput forecast exceeds fixed45minute cap; INCOMPLETE_COMPUTATIONAL_NO_GO')
                    if time.monotonic()-start>POLICY['arm_seconds']:raise RuntimeError('Training time cap exceeded')
            if step!=500:raise AssertionError('Wrong update count')
            if model_digest(model.base)!=initial_base or initial_base!=BASE_SHA:raise AssertionError('Frozen A changed')
            final=model_digest(model.head)
            if rank==0:torch.save(dict(head_state_dict=model.head.state_dict(),optimizer=opt.state_dict(),base_sha256=BASE_SHA,initial_head_sha256=initial_head,final_head_sha256=final,updates=500),out/'checkpoint.bin')
            end=predict(model,f,meta,tok,rank,False);save(out/f'end_rank{rank}.json',end)
            save(out/f'worker_rank{rank}.json',dict(status='COMPLETE',updates=step,initial_head_sha256=initial_head,final_head_sha256=final,
                initial_base_sha256=initial_base,final_base_sha256=model_digest(model.base),elapsed=time.monotonic()-start,
                grounding_evaluation='Unavailable: no held-out box-label bank packaged; no factual/grounding claim from training attribution.'))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    if sha(cfg['dataset_json'])!=DATA_SHA or sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Evaluation input identities differ')
    out=Path(cfg['output']);meta=load_metadata(cfg['metadata']);data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
    done=[json.loads((out/f'worker_rank{r}.json').read_text()) for r in (0,1)]
    if any(r['status']!='COMPLETE' or r['updates']!=500 for r in done) or done[0]['final_head_sha256']!=done[1]['final_head_sha256']:raise ValueError('Incomplete or divergent workers')
    expected={r['image_id']:r for r in meta['groups']['val']};token=PTBTokenizer()
    refs=token.tokenize({i:[dict(caption=s['raw']) for s in data['images'][i]['sentences'][:5]] for i in sorted(expected)})
    metrics={}
    for condition in ('A','end'):
        rows=sorted(sum([json.loads((out/f'{condition}_rank{r}.json').read_text()) for r in (0,1)],[]),key=lambda r:r['image_id'])
        if len(rows)!=1000 or {r['image_id'] for r in rows}!=set(expected) or any(r['coco_id']!=expected[r['image_id']]['coco_id'] for r in rows):raise ValueError('Evaluation IDs differ')
        hyp=token.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in rows});score,per=Cider().compute_score(refs,hyp)
        save(out/f'{condition}_predictions.json',rows);save(out/f'{condition}_per_image_cider.json',[dict(image_id=r['image_id'],coco_id=r['coco_id'],CIDEr=float(v)) for r,v in zip(rows,per)])
        if condition=='A' and abs(score-1.205512599)>1e-7:raise AssertionError('Pinned A evaluator mismatch')
        inp=out/f'{condition}_chair_input.json';res=out/f'{condition}_chair_output.json'
        save(inp,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(inp),'--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(res)],cwd=cfg['chair'],check=True)
        chair=json.loads(res.read_text());sentences=chair['sentences']
        if len(sentences)!=1000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in rows}:raise ValueError('CHAIR IDs differ')
        counts=dict(hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),
            object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),recalled_unique=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),reference_unique=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        values=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=counts['hallucinated_mentions']/max(1,counts['object_mentions']),Recall=counts['recalled_unique']/max(1,counts['reference_unique']))
        if any(abs(v-chair['overall_metrics'][k])>1e-12 for k,v in values.items()):raise ValueError('CHAIR aggregation differs')
        metrics[condition]=dict(CIDEr=float(score),chair=values,counts=counts,mean_words=float(np.mean([len(r['caption'].split()) for r in rows])))
    save(out/'summary.json',dict(status='COMPLETE_WAIT_MATCHED_CHT',arm=cfg['arm'],metrics=metrics,updates=500,test_used=False,
        checkpoint_sha256=sha(out/'checkpoint.bin'),preparation_receipt_sha256=sha(Path(cfg['prepared'])/'receipt.json'),sources=cfg['sources'],policy=POLICY,
        limits=['One seed, reused dev; no novelty/causal/factual certificate.','Held-out phrase/box grounding unavailable; no generated-phrase grounding claim.']))


def archive(cfg):
    out=Path(cfg.get('output',cfg.get('prepared_output')))
    if not out.exists():return
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.py','.md'):z.write(p,p.relative_to(out))
    print('SEND REVIEW:',str(out)+'_review.zip','KEEP CHECKPOINT:',out/'checkpoint.bin',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','worker','evaluate','archive']);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    for n,h in cfg['sources'].items():
        if sha(Path(n))!=h:raise ValueError('Runtime source differs: '+n)
    if a.mode in ('preflight','worker'):worker(cfg,a.mode=='preflight')
    else:globals()[a.mode](cfg)


if __name__=='__main__':main()
