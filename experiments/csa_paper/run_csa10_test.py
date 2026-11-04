"""USER-RUN locked epoch10 KarpathyTEST inference/scoring, no training/selection."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from real_bank_data import HERE,read,save,sha,validate_arrays
from run_best9_decode import saved_text,unique_file,tokens
from saved_quality_data import check_rows,chair_counts

CHECKPOINT_SHA='272f843a21b3a63c61323f748ed916fa3ae0cf2777549061439c07b410006950'


def load_test(root):
    receipt=read(root/'test_receipt.json')
    if receipt['status']!='KARPATHY_TEST_ROI_COMPLETE' or receipt['images']!=5000 or receipt['group']!='test':raise ValueError('Not full KarpathyTEST cache')
    for n,h in receipt['files'].items():
        if sha(root/n)!=h:raise ValueError('Changed TEST input '+n)
    metadata=read(root/'test_metadata.json');vocab=read(root/'vocabulary.json')
    with np.load(root/'test_regions.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    validate_arrays(arrays,metadata,vocab)
    if len(metadata)!=5000 or any(r['source_split']!='test' for r in metadata):raise ValueError('Foreign split rows')
    return arrays,metadata,vocab


def batch(arrays,indices):
    counts=[int(arrays['offsets'][i+1]-arrays['offsets'][i]) for i in indices]
    x=np.zeros((len(indices),max(counts),768),np.float32);valid=np.zeros(x.shape[:2],bool)
    for j,i in enumerate(indices):
        lo,hi=arrays['offsets'][i:i+2];x[j,:counts[j]]=arrays['features'][lo:hi];valid[j,:counts[j]]=True
    return x,valid


def worker(a):
    import torch
    from torch.nn import functional as F
    from model import Config,CSACaptioner,beam_search
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    arrays,metadata,vocab=load_test(a.output)
    arm=getattr(a,'arm','csa')
    if arm=='plain':
        from plain_test_data import PLAIN_SHA,exact_file
        if read(a.output/'test_receipt.json')!=read(HERE/'csa10_test_review_text/test_receipt.json'):raise ValueError('PlainTEST cache differs from CSA TEST')
        checkpoint_sha=PLAIN_SHA;path=exact_file(a.checkpoint_root,'checkpoint.bin',checkpoint_sha)
    else:checkpoint_sha=CHECKPOINT_SHA;path=unique_file(a.checkpoint_root,'checkpoint.bin',checkpoint_sha)
    # Exact original user checkpoint SHA is verified before loading its optimizer/RNG pickle.
    ck=torch.load(path,map_location='cpu',weights_only=False);original,_=saved_text()
    expected_identity=read(HERE/'plain10_review_text/input_identity.json') if arm=='plain' else json.loads(original['input_identity.json'])
    if arm=='plain' and any(sha(HERE/n)!=expected_identity['source'][n] for n in ('model.py','plain_model.py')):
        raise ValueError('Plain inference architecture source differs from training')
    if ck['identity']!=expected_identity or ck['state']['updates']!=177110 or ck['state']['epoch']!=11:raise ValueError('Not fixed epoch10 checkpoint/identity')
    if arm=='plain':
        from plain_model import PlainCaptioner
        base=PlainCaptioner(Config(vocab=10000))
    else:base=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
    base.load_state_dict(ck['model'],strict=True);del ck
    base.to(device).eval().requires_grad_(False);indices=list(range(rank,5000,2));rows=[];start=time.monotonic();nll_sum=0.;nll_tokens=0
    with torch.inference_mode():
        for offset in range(0,len(indices),8):
            if time.monotonic()-start>2400:raise TimeoutError('TEST inference40mincap')
            group=indices[offset:offset+8];x,v=[torch.from_numpy(t).to(device) for t in batch(arrays,group)];vo,vr=base.encode(x,v)
            for i,(seq,scores) in zip(group,beam_search(base,vo,vr,v,beams=5,maximum=20)):
                ids=seq[0].cpu().tolist()[1:];content=[t for t in tokens(ids) if t!=2]
                rows.append(dict(coco_id=metadata[i]['coco_id'],caption=' '.join(vocab[t] for t in content),ids=ids,eos=2 in ids,score=float(scores[0])))
            if offset%200==0:save(a.output/f'test_rank{rank}_partial.json',rows);print('TEST beam rank',rank,len(rows),'/2500',flush=True)
        # Separate teacher-forced evaluation only: never supplies references to beam inference.
        events=[(i,s) for i in indices for s in metadata[i]['sequences']]
        for offset in range(0,len(events),16):
            if time.monotonic()-start>2400:raise TimeoutError('TEST NLL40mincap')
            group=events[offset:offset+16];x,v=[torch.from_numpy(t).to(device) for t in batch(arrays,[i for i,s in group])]
            seq=np.asarray([s for i,s in group],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())];seq=torch.from_numpy(seq).to(device)
            logits=base(x,v,seq[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            if not torch.isfinite(loss):raise ValueError('Nonfinite TEST NLL')
            nll_sum+=float(loss);nll_tokens+=int((seq[:,1:]!=0).sum())
    check_rows(rows,{metadata[i]['coco_id'] for i in indices})
    save(a.output/f'test_rank{rank}.json',rows)
    save(a.output/f'test_worker{rank}.json',dict(complete=True,images=len(rows),nll_sum=nll_sum,nll_tokens=nll_tokens,seconds=time.monotonic()-start,
        checkpoint_sha256=checkpoint_sha,epoch=10,optimizer_steps=0,beam=5,max_new_tokens=20,test_used=True))


def score(a):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    if sha(a.chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Wrong CHAIR cache')
    metadata=read(a.output/'test_metadata.json');rows=read(a.output/'test_rank0.json')+read(a.output/'test_rank1.json')
    check_rows(rows,{m['coco_id'] for m in metadata});ptb=PTBTokenizer()
    ref=ptb.tokenize({m['coco_id']:[dict(caption=c) for c in m['captions']] for m in metadata})
    hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
    cider,_=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
    ci=a.output/'test_chair_input.json';co=a.output/'test_chair_output.json'
    save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    subprocess.run([sys.executable,'-u',str(a.chair/'chair.py'),'--cap_file',str(ci),'--cache',str(a.chair/'chair.pkl'),'--save_path',str(co)],cwd=a.chair,check=True,timeout=600)
    ch,counts=chair_counts(read(co),rows);workers=[read(a.output/f'test_worker{i}.json') for i in (0,1)]
    if not all(w['complete'] and w['images']==2500 for w in workers):raise ValueError('Incomplete TEST workers')
    point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**ch,counts=counts,
        NLL=sum(w['nll_sum'] for w in workers)/sum(w['nll_tokens'] for w in workers),
        mean_words=sum(len(r['caption'].split()) for r in rows)/5000,short_le2=sum(len(r['caption'].split())<=2 for r in rows))
    if not all(math.isfinite(v) for k,v in point.items() if k!='counts'):raise ValueError('Nonfinite TEST score')
    arm=getattr(a,'arm','csa')
    checkpoint_sha=CHECKPOINT_SHA
    if arm=='plain':
        from plain_test_data import PLAIN_SHA
        checkpoint_sha=PLAIN_SHA
    save(a.output/'result.json',dict(status='PLAIN10_KARPATHY_TEST_COMPLETE' if arm=='plain' else 'CSA10_KARPATHY_TEST_COMPLETE',metrics=point,images=5000,checkpoint_sha256=checkpoint_sha,
        epoch=10,beam=5,max_new_tokens=20,training=False,checkpoint_selection='locked exact epoch10 beforeTEST',test_used=True))
    print('KARPATHY TEST5000',point,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('worker','score'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=('csa','plain'),default='csa')
    p.add_argument('--checkpoint-root',type=Path);p.add_argument('--chair',type=Path);a=p.parse_args();globals()[a.mode](a)
