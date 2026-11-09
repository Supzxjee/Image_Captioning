"""USER-RUN CPU scoring of saved best/end VALID captions; no model inference."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import numpy as np
from real_bank_data import read,save,sha
from full_data import validate_state,STEPS,TRAIN_EVENTS

def probe(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    if importlib.metadata.version('pycocoevalcap')!='1.2':raise ValueError('Wrong evaluator version')
    p=PTBTokenizer();r=p.tokenize({1:[dict(caption='a dog running in a field')],2:[dict(caption='a cat sleeping on a chair')]})
    value,_=Cider().compute_score(r,r)
    if not np.isfinite(value):raise ValueError('Nonfinite metric probe')
    if sha(Path(cfg['chair'])/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR asset')
    save(Path(cfg['output'])/'metric_preflight.json',dict(passed=True,synthetic_only=True,CIDEr=float(value)))

def score(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import chair_counts,check_rows
    out=Path(cfg['output']);state=read(out/'progress.json');validate_state(state)
    if state['updates']!=10*STEPS or state['caption_events']!=10*TRAIN_EVENTS or state['epoch']!=11 or len(state['history'])!=10:raise ValueError('Not full10XE')
    refs=read(Path(cfg['stage'])/'validation_references.json');ids={r['coco_id'] for r in refs};results={};chair=Path(cfg['chair'])
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    for epoch in sorted({state['best_epoch'],10}):
        folder=out/'evaluations'/f'epoch_{epoch:02d}';rows=read(folder/'val_rank0.json')+read(folder/'val_rank1.json');check_rows(rows,ids)
        for name,h in state['history'][epoch-1]['files_sha256'].items():
            if sha(folder/name)!=h:raise ValueError('Changed evaluation file '+name)
        dest=out/'quality'/f'epoch_{epoch:02d}';dest.mkdir(parents=True,exist_ok=True)
        p=PTBTokenizer();ref=p.tokenize({r['coco_id']:[dict(caption=t) for t in r['references']] for r in refs});hyp=p.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
        cider,per=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
        if abs(cider-state['history'][epoch-1]['CIDEr'])>1e-7:raise ValueError('Historical epoch CIDEr replay differs')
        cap=dest/'chair_input.json';save(cap,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',cap,'--cache',chair/'chair.pkl','--save_path',dest/'chair_output.json'],cwd=chair,check=True,timeout=600)
        metric,counts=chair_counts(read(dest/'chair_output.json'),rows)
        result=dict(epoch=epoch,CIDEr=float(cider),BLEU_1=float(bleu[0]),BLEU_4=float(bleu[3]),ROUGE_L=float(rouge),**metric,counts=counts,NLL=state['history'][epoch-1]['NLL'],
            mean_words=float(np.mean([len(r['caption'].split()) for r in rows])),short_le2=sum(len(r['caption'].split())<=2 for r in rows))
        save(dest/'result.json',result);save(dest/'cider_per_image.json',[dict(coco_id=i,CIDEr=float(v)) for i,v in zip(ref,per)]);results[str(epoch)]=result
    receipts={n:read(out/n) for n in ('best_receipt.json','checkpoint_receipt.json')}
    for name,receipt in receipts.items():
        binary='best.bin' if name.startswith('best') else 'checkpoint.bin'
        if sha(out/binary)!=receipt['sha256']:raise ValueError('Changed '+binary)
    save(out/'result.json',dict(status='CLIP_PATCH_FULL_XE10_COMPLETE',epochs=10,updates=10*STEPS,caption_events=10*TRAIN_EVENTS,train_images=113287,validation_images=5000,
        method=read(out/'identity.json').get('method','plain_patch_v1'),best_epoch=state['best_epoch'],metrics_by_epoch=results,receipts=receipts,test_used=False,SCST=False,reranking=False,
        metrics_not_measured=['METEOR','SPICE','TEST metrics'],selection='maxVALID CIDEr among fixed10epochs; earlier ties; all metrics use same saved checkpoint captions',
        limits=(['Combined bank+prompt addition; parameter count and compute increase.','Geometric patch support is not ground-truth localization.'] if cfg.get('semantic') else [])+['One seed and reused VALID.','New architecture/patch IO/LR schedule differ from old plain: not an isolated bank-effect comparison.','No guarantee of110CIDEr or full convergence.']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('probe','score'));p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
