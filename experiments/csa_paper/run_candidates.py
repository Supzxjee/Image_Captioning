"""USER RUN ONLY. Decode all existing beams; no training or new selector."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from candidate_data import baseline, check_record, image_batch, choose_oracle, prepare, CHECKPOINT_SHA, MODEL_SHA
from real_bank_data import read,save,sha


def worker(cfg):
    import torch
    from model import Config,CSACaptioner,beam_search
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(cfg['output']);stage=Path(cfg['stage']);started=time.monotonic()
    if sha(cfg['checkpoint'])!=CHECKPOINT_SHA or sha(Path(__file__).with_name('model.py'))!=MODEL_SHA:raise ValueError('Changed checkpoint/model')
    receipt=read(out/'prepared.json')
    for n,h in receipt['files_sha256'].items():
        if sha(stage/n)!=h:raise ValueError('Changed staged VALID arrays')
    ck=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    if ck['identity']['arm']!='SCST_CE' or ck['state']['updates']!=14161 or ck['state']['image_events']!=113287:
        raise ValueError('Not full1 SCST+CE END161')
    torch.manual_seed(42)
    model=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
    model.load_state_dict(ck['model'],strict=True);del ck;model.to(device).eval()
    data={k:np.load(stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('features','offsets','ids')}
    vocab=read(out/'vocabulary.json');expected={r['coco_id']:r for r in baseline()['predictions']}
    images=np.arange(5000)[rank::2];path=out/f'candidates_rank{rank}.jsonl';max_error=0.;count=0
    with path.open('x',encoding='utf-8') as log,torch.inference_mode():
        for offset in range(0,len(images),8):
            if time.monotonic()-started>cfg['worker_seconds']:raise TimeoutError('Candidate decode cap; preserve partial rows')
            indices=images[offset:offset+8];x,valid=[torch.from_numpy(a).to(device) for a in image_batch(data,indices)]
            vo,vr=model.encode(x,valid)
            for i,(sequences,scores) in zip(indices,beam_search(model,vo,vr,valid,beams=5,maximum=20)):
                candidates=[]
                for ids,score in zip(sequences.cpu().tolist(),scores.cpu().tolist()):
                    from candidate_data import content
                    candidates.append(dict(caption=' '.join(vocab[t] for t in content(ids)),ids=ids,logprob=float(score)))
                row=dict(coco_id=int(data['ids'][i]),candidates=candidates)
                max_error=max(max_error,check_record(row,expected[row['coco_id']],vocab))
                log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');count+=1
            log.flush()
            if offset%200==0:print('CANDIDATES GPU',rank,count,'/2500',flush=True)
            if offset==24:
                projected=(time.monotonic()-started)/count*(2500-count)
                if projected>cfg['worker_seconds']-(time.monotonic()-started):raise TimeoutError('Measured candidate projection exceeds cap')
    save(out/f'worker{rank}.json',dict(complete=True,images=count,rank0_replayed=count,max_score_error=max_error,
        checkpoint_sha256=CHECKPOINT_SHA,candidates_sha256=sha(path),seconds=time.monotonic()-started,torch_version=torch.__version__))


def score(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import chair_counts
    out=Path(cfg['output']);saved=baseline();vocab=read(out/'vocabulary.json');expected={r['coco_id']:r for r in saved['predictions']}
    rows=[]
    if cfg.get('metric_preflight'):
        rows=[dict(coco_id=r['coco_id'],candidates=[dict(caption=r['caption'],ids=r['ids'],logprob=r['score']-k) for k in range(5)]) for r in saved['predictions']]
    else:
        for rank in (0,1):
            path=out/f'candidates_rank{rank}.jsonl';w=read(out/f'worker{rank}.json')
            if not w['complete'] or w['images']!=2500 or sha(path)!=w['candidates_sha256'] or w['checkpoint_sha256']!=CHECKPOINT_SHA:
                raise ValueError('Incomplete candidate worker')
            part=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if [r['coco_id'] for r in part]!=saved['order'][rank::2]:raise ValueError('Worker coverage/order differs')
            rows.extend(part)
    for row in rows:check_record(row,expected[row['coco_id']],vocab)
    if len(rows)!=5000 or len({r['coco_id'] for r in rows})!=5000:raise ValueError('Missing/duplicate VALID images')
    # Preserve original rank0-then-rank1 evaluation order for metric replay.
    if [r['coco_id'] for r in rows]!=[r['coco_id'] for r in saved['predictions']]:raise ValueError('Historical evaluation order differs')
    refs=read(out/'validation_references.json')
    if refs!=saved['references']:raise ValueError('Reference lists changed')
    ptb=PTBTokenizer();ref=ptb.tokenize({r['coco_id']:[dict(caption=c) for c in r['references']] for r in refs})
    ids=list(ref);order={r['coco_id']:i for i,r in enumerate(rows)}
    metrics={};per_rank=[];chair=Path(cfg['chair'])
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')

    def evaluate(name,selected):
        hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in selected})
        c,values=Cider().compute_score(ref,hyp);b,_=Bleu(4).compute_score(ref,hyp);r,_=Rouge().compute_score(ref,hyp)
        aligned=[None]*5000
        for key,value in zip(ids,values):aligned[order[key]]=float(value)
        ci=out/(name+'_chair_input.json');co=out/(name+'_chair_output.json');save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in selected])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=300)
        m,counts=chair_counts(read(co),selected)
        point=dict(CIDEr=float(c),Bleu_4=float(b[3]),ROUGE_L=float(r),**m,counts=counts,
            short_le2=sum(len(x['caption'].split())<=2 for x in selected))
        save(out/(name+'_captions.json'),selected);metrics[name]=point
        return aligned

    for k in range(5):
        print('Scoring beam rank',k,'with unchanged5000reference corpus',flush=True)
        selected=[dict(coco_id=r['coco_id'],caption=r['candidates'][k]['caption']) for r in rows]
        per_rank.append(evaluate('rank'+str(k),selected))
        if k==0:
            old=saved['metrics'];now=metrics['rank0']
            for key in ('CIDEr','Bleu_4','ROUGE_L'):
                if abs(now[key]-old[key])>1e-7:raise ValueError('Historical metric replay differs: '+key)
            if now['counts']!=old['counts'] or now['short_le2']!=old['short_le2']:raise ValueError('Historical CHAIR/length replay differs')
            if cfg.get('metric_preflight'):
                save(out/'metric_preflight.json',dict(passed=True,images=5000,metrics=now,saved_text_only=True))
                print('Historical5000 metric replay PASS before GPU decode.',flush=True)
                return
    selected,choices,upper=choose_oracle(rows,per_rank)
    evaluate('reference_oracle_diagnostic_only',selected)
    if abs(metrics['reference_oracle_diagnostic_only']['CIDEr']-upper)>1e-7:raise ValueError('Oracle IDF/alignment differs')
    save(out/'candidate_cider.json',[dict(coco_id=r['coco_id'],CIDEr=[per_rank[k][i] for k in range(5)],oracle_rank=choices[i]) for i,r in enumerate(rows)])
    summary=dict(status='CSA_CANDIDATE_DIAGNOSTIC_COMPLETE',images=5000,test_used=False,training=False,actual_reranking=False,
        checkpoint_sha256=CHECKPOINT_SHA,oracle_uses_references=True,metrics=metrics,
        CIDEr_oracle_headroom=upper-metrics['rank0']['CIDEr'],oracle_rank_counts={str(k):choices.count(k) for k in range(5)},
        unique_candidate_histogram={str(k):sum(len({c['caption'] for c in r['candidates']})==k for r in rows) for k in range(1,6)},
        all_five_short=sum(all(len(c['caption'].split())<=2 for c in r['candidates']) for r in rows))
    save(out/'result.json',summary)
    print('COMPLETE: oracle is reference-dependent diagnostic, NOT achieved reranker performance.',flush=True)
    print(json.dumps(summary,indent=2),flush=True)


def metric_preflight(cfg):
    score(dict(cfg,metric_preflight=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','score','metric_preflight']);p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
