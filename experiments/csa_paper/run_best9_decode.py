"""USER RUN ONLY: frozen best9 greedy diagnostic; no training or test images."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from real_bank_data import read, save, sha, validate_arrays, HERE
from saved_quality_data import check_rows, common_refs, chair_counts, align_cider_scores

BEST_SHA='f09e11c374477be93cc142a238dbd0e206060f33fa4fab5988e767ea7d259fe6'


def unique_file(root,name,digest):
    root=Path(root)
    matches=[p for p in root.rglob(name) if p.is_file() and sha(p)==digest]
    if len(matches)!=1:raise ValueError(f'Expected one exact {name} under {root}; found {len(matches)}')
    return matches[0]


def tokens(ids):
    ids=list(ids)
    if 2 in ids:
        end=ids.index(2)
        if any(ids[end+1:]):raise ValueError('Non-PAD after EOS')
        ids=ids[:end+1]
    if any(t<2 for t in ids):raise ValueError('PAD/BOS before EOS')
    return ids


def summary(rows):
    lengths=[len(r['caption'].split()) for r in rows]
    return dict(images=len(rows),mean_words=float(np.mean(lengths)),short_le2=sum(n<=2 for n in lengths),
                empty=sum(n==0 for n in lengths),eos=sum(r['eos'] for r in rows))


def saved_text(root=HERE):
    root=Path(root);package=read(root/'BEST9_SAVED_TEXT_PACKAGE.json')
    bundle=root/'best9_saved_text.json.gz'
    if sha(bundle)!=package['bundle_sha256']:raise ValueError('Changed saved text bundle')
    expected=read(root/'SAVE_XE139_PACKAGE.json')['files']['csa_full_xe_10_v2_review.zip']
    if package['source_review_sha256']!=expected:raise ValueError('Wrong source review identity')
    entries=json.loads(gzip.decompress(bundle.read_bytes()))
    if set(entries)!=set(package['files_sha256']):raise ValueError('Saved text members differ')
    raw={n:v.encode('utf-8') for n,v in entries.items()}
    for n,h in package['files_sha256'].items():
        if hashlib.sha256(raw[n]).hexdigest()!=h:raise ValueError('Changed saved text payload: '+n)
    return raw,package


def prepare(args):
    checkpoint=unique_file(args.checkpoint_root,'best.bin',BEST_SHA)
    raw,package=saved_text()
    for name in ('validation_references.json','vocabulary.json','input_identity.json'):
        (args.output/name).write_bytes(raw[name])
    beam=json.loads(raw['val_epoch09_rank0.json'])+json.loads(raw['val_epoch09_rank1.json'])
    save(args.output/'saved_text_receipt.json',package)
    refs=read(args.output/'validation_references.json');vocab=read(args.output/'vocabulary.json')
    check_rows(beam,{r['coco_id'] for r in refs})
    if len(refs)!=5000 or len(vocab)!=10000:raise ValueError('Wrong full validation/vocabulary')
    save(args.output/'beam5.json',beam)
    pin=read(HERE/'SAVE_FULL_ROI_V134_PACKAGE.json')['files']
    root=unique_file(args.full_root,'receipt.json',pin['receipt.json']).parent
    for name in ('manifest.json','vocabulary.json'):
        if sha(root/name)!=pin[name]:raise ValueError('Changed ROI '+name)
    if read(root/'vocabulary.json')!=vocab:raise ValueError('ROI/checkpoint vocabulary differs')
    features=[];offsets=[0];ids=[];observed_refs=[]
    for entry in read(root/'manifest.json')['shards']:
        if entry['group']!='val':continue
        for name in entry['files']:
            if sha(root/name)!=pin[name]:raise ValueError('Changed validation shard '+name)
        images=read(root/next(n for n in entry['files'] if n.endswith('.json')))['images']
        with np.load(root/next(n for n in entry['files'] if n.endswith('.npz')),allow_pickle=False) as f:
            raw={k:f[k] for k in f.files}
        validate_arrays(raw,images,vocab)
        features.append(raw['features']);offsets.extend((raw['offsets'][1:]+offsets[-1]).tolist());ids.extend(raw['ids'].tolist())
        observed_refs.extend(dict(coco_id=r['coco_id'],references=r['captions']) for r in images)
        print('Staged validation only:',len(ids),'/5000',flush=True)
    if observed_refs!=refs or len(set(ids))!=5000:raise ValueError('Validation order/references differ')
    for name,value in dict(features=np.concatenate(features),offsets=np.asarray(offsets,np.int64),ids=np.asarray(ids,np.int64)).items():
        np.save(args.output/(name+'.npy'),value,allow_pickle=False)
    save(args.output/'prepared.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=BEST_SHA,
        staged_sha256={n:sha(args.output/n) for n in ('features.npy','offsets.npy','ids.npy')},
        validation_images=5000,training_images=0,test_used=False))


def worker(args):
    import torch
    from model import CSACaptioner,Config,beam_search
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4 x2')
    torch.cuda.set_device(rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prepared=read(args.output/'prepared.json');checkpoint=Path(prepared['checkpoint'])
    if sha(checkpoint)!=BEST_SHA:raise ValueError('Changed best9 checkpoint')
    for n,h in prepared['staged_sha256'].items():
        if sha(args.output/n)!=h:raise ValueError('Changed validation stage')
    # Exact SHA above is the user's own optimizer/RNG checkpoint; never accept arbitrary pickle.
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['identity']!=read(args.output/'input_identity.json'):raise ValueError('Checkpoint input/source identity differs')
    if ck['state']['updates']!=159399:raise ValueError('Not epoch9 endpoint')
    model=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
    model.load_state_dict(ck['model'],strict=True);del ck
    model.to(rank).eval().requires_grad_(False)
    data={n:np.load(args.output/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('features','offsets','ids')}
    vocab=read(args.output/'vocabulary.json');beam={r['coco_id']:r for r in read(args.output/'beam5.json')}
    indices=list(range(rank,5000,2));rows=[];start=time.monotonic();replay=[]
    with torch.inference_mode():
        for offset in range(0,len(indices),8):
            if time.monotonic()-start>args.wall_seconds:raise TimeoutError('Decode cap exceeded; preserve partial output')
            group=indices[offset:offset+8];counts=[int(data['offsets'][i+1]-data['offsets'][i]) for i in group]
            x=np.zeros((len(group),max(counts),768),np.float32);valid=np.zeros(x.shape[:2],bool)
            for j,i in enumerate(group):
                lo,hi=data['offsets'][i:i+2];x[j,:counts[j]]=data['features'][lo:hi];valid[j,:counts[j]]=True
            x=torch.from_numpy(x).to(rank);valid=torch.from_numpy(valid).to(rank);vo,vr=model.encode(x,valid)
            if offset==0:
                for i,(seq,scores) in zip(group,beam_search(model,vo,vr,valid,beams=5,maximum=20)):
                    old=beam[int(data['ids'][i])];new_ids=seq[0].cpu().tolist()[1:]
                    record=dict(coco_id=old['coco_id'],actual_ids=new_ids,expected_ids=old['ids'],actual_score=float(scores[0]),expected_score=old['score'])
                    replay.append(record);save(args.output/f'replay_rank{rank}.json',replay)
                    if tokens(new_ids)!=tokens(old['ids']) or not math.isclose(float(scores[0]),old['score'],rel_tol=1e-5,abs_tol=1e-4):
                        raise ValueError('Sample beam5 replay differs; inspect replay JSON, do not loosen guard')
            for i,(seq,scores) in zip(group,beam_search(model,vo,vr,valid,beams=1,maximum=20)):
                ids=seq[0].cpu().tolist()[1:];clean=tokens(ids);content=clean[:-1] if clean and clean[-1]==2 else clean
                rows.append(dict(coco_id=int(data['ids'][i]),caption=' '.join(vocab[t] for t in content),ids=ids,eos=2 in ids,score=float(scores[0])))
            if offset%200==0:
                save(args.output/f'greedy_rank{rank}_partial.json',rows)
                elapsed=time.monotonic()-start
                print('GREEDY rank',rank,len(rows),'/2500 seconds',round(elapsed,1),'projected_remaining',round(elapsed/max(1,len(rows))*(2500-len(rows)),1),flush=True)
    check_rows(rows,{int(data['ids'][i]) for i in indices})
    save(args.output/f'greedy_rank{rank}.json',rows)
    save(args.output/f'worker_rank{rank}.json',dict(complete=True,summary=summary(rows),seconds=time.monotonic()-start,optimizer_steps=0,device=torch.cuda.get_device_name(rank)))


def score(args):
    import gzip
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    if sha(args.chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Wrong CHAIR cache')
    refs=read(args.output/'validation_references.json');all_ids={r['coco_id'] for r in refs}
    beam=read(args.output/'beam5.json');greedy=read(args.output/'greedy_rank0.json')+read(args.output/'greedy_rank1.json')
    for rows in (beam,greedy):check_rows(rows,all_ids)
    bundle=json.loads(gzip.decompress((HERE/'comparison_baseline4000.json.gz').read_bytes()))
    expected=read(HERE/'saved_quality_review_text/comparison.json')['metrics'];results={};ptb=PTBTokenizer()
    for cohort,references in (('full5000',refs),('common4000',common_refs(refs,bundle))):
        ids={r['coco_id'] for r in references};metric_id=(lambda i:bundle['legacy_image_ids'][str(i)]) if cohort=='common4000' else (lambda i:i)
        ref=ptb.tokenize({metric_id(r['coco_id']):[dict(caption=s) for s in r['references']] for r in references});results[cohort]={}
        for name,all_rows in (('beam5',beam),('greedy',greedy)):
            rows=[r for r in all_rows if r['coco_id'] in ids]
            if cohort=='common4000':rows.sort(key=lambda r:metric_id(r['coco_id']))
            hyp=ptb.tokenize({metric_id(r['coco_id']):[dict(caption=r['caption'])] for r in rows})
            cider,per=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
            label=cohort+'_'+name;ci=args.output/(label+'_chair_input.json');co=args.output/(label+'_chair_output.json')
            save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
            subprocess.run([sys.executable,'-u',str(args.chair/'chair.py'),'--cap_file',str(ci),'--cache',str(args.chair/'chair.pkl'),'--save_path',str(co)],cwd=args.chair,check=True,timeout=600)
            metrics,counts=chair_counts(read(co),rows)
            point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metrics,counts=counts,**summary(rows))
            if any(not math.isfinite(point[k]) for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall')):raise ValueError('Nonfinite metric')
            results[cohort][name]=point;save(args.output/'metrics_partial.json',results)
            save(args.output/(label+'_per_image.json'),[dict(coco_id=r['coco_id'],CIDEr=v) for r,v in zip(rows,align_cider_scores(references,per,rows))])
            if name=='beam5':
                old=expected[cohort]['CSA9']
                for key in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall','mean_words'):
                    if abs(point[key]-old[key])>1e-7:raise ValueError('Beam metric replay differs: '+cohort+'/'+key)
                if counts!=old['counts']:raise ValueError('Beam raw counts replay differs')
            print('QUALITY',label,point,flush=True)
    b={r['coco_id']:r['caption'] for r in beam}
    save(args.output/'result.json',dict(status='BEST9_DECODE_COMPLETE',metrics=results,changed_captions=sum(b[r['coco_id']]!=r['caption'] for r in greedy),
        checkpoint_sha256=BEST_SHA,epoch=9,optimizer_steps=0,max_new_tokens=20,beam_reference='saved Output139 epoch9',
        beam_sample_replay_images=16,training=False,test_used=False,development_reused=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker','score'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-root',type=Path);p.add_argument('--checkpoint-root',type=Path);p.add_argument('--chair',type=Path);p.add_argument('--wall-seconds',type=float,default=5400)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);globals()[a.mode](a)
