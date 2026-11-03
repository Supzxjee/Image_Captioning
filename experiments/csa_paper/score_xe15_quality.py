"""USER-RUN saved-caption CPU scoring, never loads a neural model."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from real_bank_data import read,save,sha,HERE
from saved_quality_data import check_rows,chair_counts


def run(output,chair):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    result=read(output/'result.json')
    if result['status']!='FULL_XE_15_COMPLETE' or result['updates']!=265665:raise ValueError('Training not complete15epochs')
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Wrong CHAIR cache')
    refs=read(output/'validation_references.json');ids={r['coco_id'] for r in refs}
    ref={int(k):v for k,v in read(output/'validation_ptb.json').items()}
    if set(ref)!=ids or len(ids)!=5000:raise ValueError('Wrong reference coverage')
    expected=read(HERE/'saved_quality_review_text/comparison.json')['metrics']['full5000']['CSA9'];metrics={}
    for epoch in (9,11,12,13,14,15):
        rows=read(output/f'val_epoch{epoch:02d}_rank0.json')+read(output/f'val_epoch{epoch:02d}_rank1.json');check_rows(rows,ids)
        prefix=output/f'quality_epoch{epoch:02d}';ci=prefix.with_suffix('.chair_input.json');co=prefix.with_suffix('.chair_output.json')
        hyp=PTBTokenizer().tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
        cider,_=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
        target=result['history'][epoch-1]['cider']
        if abs(cider-target)>1e-7:raise ValueError('Stored validation CIDEr replay differs: '+str(epoch))
        save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(chair/'chair.py'),'--cap_file',str(ci),'--cache',str(chair/'chair.pkl'),'--save_path',str(co)],cwd=chair,check=True,timeout=600)
        ch,counts=chair_counts(read(co),rows);lengths=[len(r['caption'].split()) for r in rows]
        point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**ch,counts=counts,
            mean_words=sum(lengths)/len(lengths),short_le2=sum(x<=2 for x in lengths),empty=sum(x==0 for x in lengths))
        if not all(math.isfinite(point[k]) for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall')):raise ValueError('Nonfinite score')
        metrics[str(epoch)]=point;save(output/'quality_partial.json',metrics)
        if epoch==9:
            if counts!=expected['counts'] or any(abs(point[k]-expected[k])>1e-7 for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall','mean_words')):
                raise ValueError('Old best9 quality replay differs')
        print('SAVED QUALITY',epoch,point,flush=True)
    save(output/'quality_result.json',dict(status='XE15_SAVED_QUALITY_COMPLETE',metrics=metrics,best_epoch=result['best_epoch'],
        selection='fixed validation CIDEr, not CHAIR',new_model_generation=False,test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--chair',type=Path,required=True)
    a=p.parse_args();run(a.output,a.chair)
