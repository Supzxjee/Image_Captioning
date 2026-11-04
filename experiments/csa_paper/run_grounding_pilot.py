"""USER-RUN ONLY: matched CSA10 continuation on two GPUs, then VALID5000."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from real_bank_data import read,save,sha
from grounding_pilot_data import prepare,exact_checkpoint,batch,CHECKPOINT_SHA
from full_cache_data import collate


def atomic(path,value):
    tmp=path.with_suffix('.tmp');save(tmp,value);os.replace(tmp,path)


def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model import Config,CSACaptioner,beam_search
    from grounding_model import install,grounding_sum
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    start=time.monotonic();deadline=start+cfg['worker_seconds'];out=Path(cfg['output']);stage=Path(cfg['stage'])
    try:
        receipt=read(out/'pilot_stage.json');staged=read(stage/'stage_receipt.json')
        for n,h in {'grounding_labels.json':receipt['labels_sha256'],'batches.npy':receipt['batches_sha256']}.items():
            if sha(stage/n)!=h:raise ValueError('Pilot stage changed')
        if rank==0:
            for n,h in staged['files_sha256'].items():
                if sha(stage/n)!=h:raise ValueError('ROI stage changed')
        dist.barrier(device_ids=[rank])
        data={k:np.load(stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in staged['shapes']}
        labels=read(stage/'grounding_labels.json');batches=np.load(stage/'batches.npy',allow_pickle=False)
        ckpath=exact_checkpoint(cfg['checkpoint_root']);ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        if ck['state']['updates']!=177110 or ck['state']['epoch']!=11:raise ValueError('Not exact CSA10 state')
        if set(ck['model'])==set() or ck['identity']['epochs']!=10:raise ValueError('Wrong checkpoint identity')
        if sha(Path(__file__).with_name('model.py'))!='139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09':raise ValueError('Original CSA model source changed')
        torch.manual_seed(42)
        base=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
        base.load_state_dict(ck['model'],strict=True);del ck;base.to(device).eval()
        # Same-shape original-vs-instrumented forward, not cross-batch beam-ID equality.
        x,v,seq,_=batch(data,batches[0,:4],labels)
        x,v,seq=[torch.from_numpy(a).to(device) for a in (x,v,seq)]
        with torch.no_grad():expected=base(x,v,seq[:,:-1])
        att=install(base);att.collect=True
        with torch.no_grad():actual=base(x,v,seq[:,:-1])
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        att.collect=False;att.log_attention=None
        if rank==0:save(out/'instrumentation_parity.json',dict(passed=True,rtol=1e-4,atol=1e-5,images=4,max_abs=float((actual-expected).abs().max())))
        del actual,expected,x,v,seq

        class Reader(nn.Module):
            def __init__(self):super().__init__();self.base=base
            def forward(self,x,v,prefix,positive):
                att.collect=True;logits=self.base(x,v,prefix)
                loss,_=grounding_sum(att.log_attention,positive)
                return logits,loss

        model=DDP(Reader(),device_ids=[rank],output_device=rank,broadcast_buffers=False,find_unused_parameters=False)
        optimizer=torch.optim.Adam(base.parameters(),lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
        identity=dict(schema='csa_grounding_pilot_v1',arm=cfg['arm'],weight=0. if cfg['arm']=='XE' else .1,
            source_checkpoint=CHECKPOINT_SHA,updates=2000,global_batch=32,seed=42,lr=1e-5,
            source=read(out/'source_receipt.json'),stage={k:v for k,v in receipt.items() if k!='checkpoint'},
            labels_human_verified=False)
        state=dict(updates=0,caption_events=0,nll_sum=0.,tokens=0,ground_sum=0.,ground_mentions=0)
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42);torch.cuda.manual_seed(4242+rank)
        if cfg.get('resume_root'):
            root=Path(cfg['resume_root']);resume_receipt=read(root/'checkpoint_receipt.json')
            if sha(root/'checkpoint.bin')!=resume_receipt['sha256']:raise ValueError('Changed resume checkpoint')
            restored=torch.load(root/'checkpoint.bin',map_location='cpu',weights_only=False)
            if restored['identity']!=identity:raise ValueError('Resume arm/protocol/source/data differs')
            state=restored['state'];base.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
            rng=restored['rng_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
            del restored
        if not 0<=state['updates']<=2000 or state['caption_events']!=state['updates']*32:raise ValueError('Bad pilot cursor')
        if rank==0:save(out/'identity.json',identity)

        def checkpoint():
            for name in ('nouns','verbs'):
                if getattr(base,name).requires_grad:raise ValueError('Bank must be frozen')
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
            rngs=[None,None];dist.all_gather_object(rngs,rng)
            if rank==0:
                temp=out/'checkpoint.tmp';torch.save(dict(model=base.state_dict(),optimizer=optimizer.state_dict(),state=state,identity=identity,rng_by_rank=rngs),temp)
                os.replace(temp,out/'checkpoint.bin');atomic(out/'checkpoint_receipt.json',dict(sha256=sha(out/'checkpoint.bin'),updates=state['updates'],caption_events=state['caption_events'],arm=cfg['arm']))
                atomic(out/'progress.json',state)
            dist.barrier(device_ids=[rank])

        base.train()
        for step in range(state['updates'],2000):
            if step%25==0:
                stop=torch.tensor(int(time.monotonic()>deadline-900),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if int(stop):checkpoint();raise TimeoutError('Saved pilot checkpoint before wall cap; preserve whole output for resume')
            events=batches[step,rank*16:(rank+1)*16]
            x,v,seq,p=[torch.from_numpy(a).to(device) for a in batch(data,events,labels)]
            local_tokens=int((seq[:,1:]!=0).sum());local_mentions=int(p.any(-1).sum())
            denominators=torch.tensor([local_tokens,local_mentions],device=device,dtype=torch.long);dist.all_reduce(denominators)
            optimizer.zero_grad(set_to_none=True);logits,ground=model(x,v,seq[:,:-1],p)
            nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            loss=nll*(2/int(denominators[0]))+identity['weight']*ground*(2/max(1,int(denominators[1])))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(base.parameters(),1,error_if_nonfinite=True);optimizer.step()
            stats=torch.stack((nll.detach().double(),ground.detach().double()));dist.all_reduce(stats)
            state['updates']=step+1;state['caption_events']+=32;state['nll_sum']+=float(stats[0]);state['tokens']+=int(denominators[0]);state['ground_sum']+=float(stats[1]);state['ground_mentions']+=int(denominators[1])
            att.log_attention=None
            if (step+1)%50==0 and rank==0:print('TRAIN',dict(arm=cfg['arm'],step=step+1,total=2000,nll=float(stats[0])/int(denominators[0]),ground=float(stats[1])/max(1,int(denominators[1])),grad_norm=float(norm),seconds=time.monotonic()-start),flush=True)
            if (step+1)%500==0:checkpoint()
        if state['updates']!=2000:raise ValueError('Incomplete training')
        checkpoint();att.collect=False;att.log_attention=None;base.eval()
        refs=read(out/'validation_references.json');vocab=read(out/'vocabulary.json');indices=np.flatnonzero(data['group']==1)[rank::2]
        if len(refs)!=5000 or len(indices)!=2500:raise ValueError('Wrong VALID population')
        rows=[];total=0.;tokens=0;began=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(indices),8):
                if time.monotonic()>deadline:raise TimeoutError('Validation cap: endpoint checkpoint preserved')
                images=indices[offset:offset+8];events=np.searchsorted(data['caption_image'],images)
                x,v,_=[torch.from_numpy(a).to(device) for a in collate(data,events)]
                vo,vr=base.encode(x,v)
                for i,(seq,scores) in zip(images,beam_search(base,vo,vr,v,beams=5,maximum=20)):
                    ids=seq[0].cpu().tolist()[1:];content=ids[:ids.index(2)] if 2 in ids else ids
                    rows.append(dict(coco_id=int(data['ids'][i]),caption=' '.join(vocab[t] for t in content if t!=0),ids=ids,eos=2 in ids,score=float(scores[0])))
                if offset%400==0:print('VALID beam',cfg['arm'],'rank',rank,len(rows),'/2500',flush=True)
            events=np.flatnonzero(np.isin(data['caption_image'],indices))
            for offset in range(0,len(events),16):
                if time.monotonic()>deadline:raise TimeoutError('Validation NLL cap: endpoint preserved')
                x,v,seq=[torch.from_numpy(a).to(device) for a in collate(data,events[offset:offset+16])]
                nll=F.cross_entropy(base(x,v,seq[:,:-1]).flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                if not torch.isfinite(nll):raise ValueError('Nonfinite VALID NLL')
                total+=float(nll);tokens+=int((seq[:,1:]!=0).sum())
        save(out/f'val_rank{rank}.json',rows)
        save(out/f'worker{rank}.json',dict(complete=True,arm=cfg['arm'],updates=2000,images=len(rows),nll_sum=total,nll_tokens=tokens,seconds=time.monotonic()-start,validation_seconds=time.monotonic()-began))
        dist.barrier(device_ids=[rank])
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def score(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import check_rows,chair_counts
    out=Path(cfg['output']);chair=Path(cfg['chair']);refs=read(out/'validation_references.json');rows=read(out/'val_rank0.json')+read(out/'val_rank1.json')
    workers=[read(out/f'worker{r}.json') for r in (0,1)]
    if not all(w['complete'] and w['updates']==2000 and w['images']==2500 for w in workers):raise ValueError('Incomplete pilot workers')
    check_rows(rows,{r['coco_id'] for r in refs})
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Wrong CHAIR cache')
    ptb=PTBTokenizer();ref=ptb.tokenize({r['coco_id']:[dict(caption=c) for c in r['references']] for r in refs});hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
    cider,per_image=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
    ci=out/'chair_input.json';co=out/'chair_output.json';save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    subprocess.run([sys.executable,'-u',str(chair/'chair.py'),'--cap_file',str(ci),'--cache',str(chair/'chair.pkl'),'--save_path',str(co)],cwd=chair,check=True,timeout=600)
    metrics,counts=chair_counts(read(co),rows);state=read(out/'progress.json')
    if state['updates']!=2000 or state['caption_events']!=64000:raise ValueError('Wrong endpoint')
    point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metrics,counts=counts,
        NLL=sum(w['nll_sum'] for w in workers)/sum(w['nll_tokens'] for w in workers),
        train_NLL=state['nll_sum']/state['tokens'],train_ground=state['ground_sum']/max(1,state['ground_mentions']),
        mean_words=sum(len(r['caption'].split()) for r in rows)/5000,short_le2=sum(len(r['caption'].split())<=2 for r in rows))
    if not all(np.isfinite(v) for k,v in point.items() if k!='counts'):raise ValueError('Nonfinite pilot metric')
    save(out/'cider_per_image.json',[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(ref,per_image)])
    save(out/'result.json',dict(status='CSA_GROUNDING_PILOT_COMPLETE',arm=cfg['arm'],updates=2000,caption_events=64000,validation_images=5000,test_used=False,labels_human_verified=False,metrics=point,selection='fixed update2000 endpoint, no checkpoint selection'))
    print('PILOT COMPLETE',cfg['arm'],point,flush=True)


def metric_preflight(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from saved_quality_data import chair_counts
    out=Path(cfg['output']);chair=Path(cfg['chair']);refs=read(out/'validation_references.json')[:2]
    rows=[dict(coco_id=r['coco_id'],caption=r['references'][0]) for r in refs]
    tokenized=PTBTokenizer().tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
    value,_=Cider().compute_score(tokenized,tokenized)
    if not np.isfinite(value) or value<=0:raise ValueError('PTB/Java/CIDEr preflight failed')
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Wrong CHAIR cache')
    ci=out/'metric_preflight_input.json';co=out/'metric_preflight_chair.json'
    save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    subprocess.run([sys.executable,'-u',str(chair/'chair.py'),'--cap_file',str(ci),'--cache',str(chair/'chair.pkl'),'--save_path',str(co)],cwd=chair,check=True,timeout=180)
    chair_counts(read(co),rows);save(out/'metric_preflight.json',dict(passed=True,CIDEr=float(value),images=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','metric_preflight','worker','score'));p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
