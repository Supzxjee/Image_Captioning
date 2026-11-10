"""USER-RUN fixed199 inference and CPU scoring. No training mode."""
import argparse
from datetime import timedelta
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from real_bank_data import read,save,sha
from test199_data import prepare,load,lock,validate_lock,checkpoint,runtime_policy,CHECKPOINT_SHA,TEST_SIZE
from saved_quality_data import check_rows,chair_counts

def contracts(cfg):
    if cfg.get('semantic'):
        import semantic_test_data as module
    else:
        import test199_data as module
    return module


def checked_weights(cfg):
    import torch
    d=contracts(cfg);ck=torch.load(d.checkpoint(cfg['checkpoint_root']),map_location='cpu',weights_only=False)
    if cfg.get('semantic'):
        d.validate_checkpoint_state(ck)
        return ck,d.runtime_policy(ck['identity']['software']['torch'],str(torch.__version__))
    if ck['state']['updates']!=177110 or ck['state']['best_epoch']!=10 or ck['identity']['source']['commit']!=lock()['training_commit']:raise ValueError('Unexpected fixed199 training state')
    return ck,runtime_policy(ck['identity']['software']['torch'],str(torch.__version__))

def checkpoint_preflight(cfg):
    ck,policy=checked_weights(cfg);del ck
    save(Path(cfg['output'])/'checkpoint_runtime_preflight.json',dict(checkpoint_sha256=contracts(cfg).CHECKPOINT_SHA,runtime_policy=policy,models_executed=False))
    print('Checkpoint/runtime metadata PASS:',policy,flush=True)

def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from patch_model import PatchCaptioner
    from patch_beam import beam_search,check_paths,text_from_ids
    from patch_data import batch as caption_batch
    from cuda_runtime_guard import check_runtime
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);check_runtime(out,rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Use T4x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=1200));started=time.monotonic();deadline=started+cfg['worker_seconds']
    try:
        d=contracts(cfg);data,words,receipt,locked=d.load(cfg,verify=rank==0);dist.barrier(device_ids=[rank])
        ck,policy=checked_weights(cfg);save(out/f'runtime_policy_rank{rank}.json',policy)
        context=None
        if cfg.get('semantic'):
            from semantic_patch_model import SemanticPatchCaptioner
            from semantic_data import SemanticContext
            model=SemanticPatchCaptioner(len(words),ck['model']['noun_bank'],ck['model']['verb_bank']);context=SemanticContext(cfg['stage'])
        else:model=PatchCaptioner(len(words))
        model=model.to(device);model.load_state_dict(ck['model'],strict=True);del ck;model.eval();rows=[];images=np.arange(TEST_SIZE)[rank::2]
        with torch.inference_mode():
            for offset in range(0,len(images),8):
                if time.monotonic()>deadline-30:raise TimeoutError('TEST decode phase deadline')
                index=images[offset:offset+8];x=torch.from_numpy(np.asarray(data['patches'][index],np.float32)).to(device)
                memory=model.encode(x,context.images(index,device)) if context else model.encode(x);seq,scores=beam_search(model,memory,width=locked['beam_width'],maximum=locked['max_new_tokens'])
                if offset==0:check_paths(model,memory,seq,scores)
                for i,s,v in zip(index.tolist(),seq[:,0].cpu().tolist(),scores[:,0].cpu().tolist()):rows.append(dict(coco_id=int(data['ids'][i]),caption=text_from_ids(s,words),ids=s,score=float(v)))
                if offset%200==0:print('TEST decode rank',rank,len(rows),'/',len(images),flush=True)
            save(out/f'captions_rank{rank}.json',rows)
            nll_sum=0.;tokens=0;ev=np.flatnonzero(np.isin(data['caption_image'],images))
            for offset in range(0,len(ev),32):
                if time.monotonic()>deadline-30:raise TimeoutError('TEST teacherNLL phase deadline; captions preserved')
                x,seq=caption_batch(data,ev[offset:offset+32]);x=torch.from_numpy(x).to(device);seq=torch.from_numpy(seq).to(device)
                logits=model(x,seq[:,:-1],context.events(data,ev[offset:offset+32],device)) if context else model(x,seq[:,:-1]);nll_sum+=float(F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum'));tokens+=int((seq[:,1:]!=0).sum())
        if len(rows)!=2500:raise ValueError('Incomplete TEST rank')
        save(out/f'worker{rank}.json',dict(complete=True,rank=rank,images=len(rows),NLL_sum=nll_sum,NLL_tokens=tokens,seconds=time.monotonic()-started,training=False,checkpoint_sha256=d.CHECKPOINT_SHA,caption_sha256=sha(out/f'captions_rank{rank}.json'),runtime_policy=policy,first_GPU_beam_path_check_passed=True))
        dist.barrier(device_ids=[rank])
    finally:
        if dist.is_initialized():dist.destroy_process_group()

def inputs(cfg):
    d=contracts(cfg);out=Path(cfg['output']);locked=d.lock();saved=read(out/'test_lock.json');d.validate_lock(saved);receipt=read(out/'stage_receipt.json')
    if saved['stage_receipt_sha256']!=sha(out/'stage_receipt.json') or sha(out/'test_references.json')!=receipt['output_files_sha256']['test_references.json']:raise ValueError('Changed TEST references/receipt')
    refs=read(out/'test_references.json');workers=[read(out/f'worker{r}.json') for r in (0,1)]
    if not all(w['complete'] and w['images']==2500 and w['checkpoint_sha256']==d.CHECKPOINT_SHA for w in workers):raise ValueError('Incomplete TEST model output')
    if workers[0]['runtime_policy']!=workers[1]['runtime_policy'] or not all(w['first_GPU_beam_path_check_passed'] for w in workers):raise ValueError('Runtime/beam guards differ between ranks')
    rows=[]
    for rank,w in enumerate(workers):
        if sha(out/f'captions_rank{rank}.json')!=w['caption_sha256']:raise ValueError('Changed TEST rank captions')
        rows+=read(out/f'captions_rank{rank}.json')
    check_rows(rows,locked['expected_test_ids'])
    if len(refs)!=5000 or {r['coco_id'] for r in refs}!=set(locked['expected_test_ids']):raise ValueError('Wrong TEST references')
    return out,refs,rows,workers

def tokenize(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    out,refs,rows,workers=inputs(cfg);rows=sorted(rows,key=lambda r:r['coco_id']);save(out/'captions.json',rows)
    tok=PTBTokenizer();gts=tok.tokenize({r['coco_id']:[dict(caption=t) for t in r['references']] for r in refs});hyp=tok.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
    save(out/'reference_ptb.json',gts);save(out/'caption_ptb.json',hyp)
    save(out/'scoring_lock.json',dict(checkpoint_sha256=contracts(cfg).CHECKPOINT_SHA,files_sha256={n:sha(out/n) for n in ('captions.json','test_references.json','test_lock.json','reference_ptb.json','caption_ptb.json')},training=False,reranking=False))

def score(cfg,mode):
    out=Path(cfg['output']);sealed=read(out/'scoring_lock.json')
    for n,h in sealed['files_sha256'].items():
        if sha(out/n)!=h:raise ValueError('Changed scoring input '+n)
    gts=read(out/'reference_ptb.json');hyp=read(out/'caption_ptb.json')
    if len(gts)!=5000 or set(gts)!=set(hyp):raise ValueError('Incomplete TEST scorer cohort')
    result={}
    if mode=='basic':
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        b,_=Bleu(4).compute_score(gts,hyp);c,per=Cider().compute_score(gts,hyp);r,_=Rouge().compute_score(gts,hyp)
        result=dict(BLEU_1=float(b[0]),BLEU_4=float(b[3]),CIDEr=float(c),ROUGE_L=float(r));save(out/'cider_per_image.json',[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(gts,per)])
    elif mode in ('meteor','spice'):
        if mode=='meteor':
            from pycocoevalcap.meteor.meteor import Meteor
            scorer=Meteor()
        else:
            from pycocoevalcap.spice.spice import Spice
            scorer=Spice()
        try:point,per=scorer.compute_score(gts,hyp)
        finally:del scorer
        if len(per)!=5000:raise ValueError('Incomplete '+mode)
        result={mode.upper():float(point)}
        values=[float(x) for x in per] if mode=='meteor' else [float(x['All']['f']) for x in per]
        if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite per-image '+mode)
        save(out/(mode+'_per_image.json'),[dict(coco_id=int(i),score=v) for i,v in zip(gts,values)])
    elif mode=='chair':
        rows=read(out/'captions.json');save(out/'chair_input.json',[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows]);chair=Path(cfg['chair'])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',out/'chair_input.json','--cache',chair/'chair.pkl','--save_path',out/'chair_output.json'],cwd=chair,check=True,timeout=600)
        result,counts=chair_counts(read(out/'chair_output.json'),rows);save(out/'chair_counts.json',counts)
    else:raise ValueError(mode)
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metric')
    save(out/(mode+'.json'),dict(images=5000,metrics=result,input_sha256=sealed['files_sha256']))

def combine(cfg):
    out,refs,rows,workers=inputs(cfg);sealed=read(out/'scoring_lock.json');metrics={}
    for n,h in sealed['files_sha256'].items():
        if sha(out/n)!=h:raise ValueError('Changed final scoring input '+n)
    for mode in ('basic','meteor','spice','chair'):
        p=read(out/(mode+'.json'))
        if p['images']!=5000 or p['input_sha256']!=sealed['files_sha256']:raise ValueError('Incomplete/changed metric phase')
        metrics.update(p['metrics'])
    required={'BLEU_1','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi','Recall'}
    if set(metrics)!=required or not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Need all eight metrics and recall')
    metrics['NLL']=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    save(out/'result.json',dict(status='CLIP_PATCH_BANK_PROMPT_TEST5000_COMPLETE' if cfg.get('semantic') else 'CLIP_PATCH_199_TEST5000_COMPLETE',images=5000,checkpoint_sha256=contracts(cfg).CHECKPOINT_SHA,source_version=contracts(cfg).lock()['source_version'],epoch=contracts(cfg).lock()['epoch'],metrics=metrics,display_x100={k:100*v for k,v in metrics.items() if k!='NLL'},beam_width=5,max_new_tokens=20,training=False,reranking=False,test_tuning=False,prior_project_test_exposure=True,chair_counts=read(out/'chair_counts.json'),environment=read(out/'environment.json'),runtime_policy=workers[0]['runtime_policy']))
    print('TEST5000 COMPLETE',metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('checkpoint_preflight','prepare','worker','environment','smoke','tokenize','basic','meteor','spice','chair','combine'));p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    if a.mode in ('environment','smoke','basic','meteor','spice','chair'):
        import score_prompt_test_extra as cpu
        if a.mode=='environment':cpu.environment(Path(cfg['output']))
        else:
            cpu.java_path(Path(cfg['output']))
            if a.mode=='smoke':cpu.smoke(Path(cfg['output']))
            else:score(cfg,a.mode)
    elif a.mode=='tokenize':
        import score_prompt_test_extra as cpu
        cpu.java_path(Path(cfg['output']));tokenize(cfg)
    elif a.mode=='prepare':contracts(cfg).prepare(cfg)
    else:globals()[a.mode](cfg)
