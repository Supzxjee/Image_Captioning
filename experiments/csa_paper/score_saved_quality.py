"""USER-RUN CPU metrics on saved captions; no neural imports or checkpoint reads."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile
from saved_quality_data import check_rows,common_refs,chair_counts

CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(args):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    import numpy as np
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if sha(args.review)!=manifest['new_review_zip_sha256'] or sha(args.baseline)!=manifest['payload_sha256']:raise ValueError('Changed review/baseline package')
    if sha(args.chair/'chair.pkl')!=CHAIR_SHA:raise ValueError('Wrong CHAIR cache')
    baseline=json.loads(gzip.decompress(args.baseline.read_bytes()))
    with zipfile.ZipFile(args.review) as z:
        def read(n):return json.loads(z.read(n))
        result=read('result.json');refs=read('validation_references.json')
        if result['status']!='FULL_XE_10_COMPLETE' or result['best_epoch']!=9:raise ValueError('Wrong completed CSA run')
        new={f'CSA{epoch}':read(f'val_epoch{epoch:02d}_rank0.json')+read(f'val_epoch{epoch:02d}_rank1.json') for epoch in (9,10)}
    old_refs=common_refs(refs,baseline);all_ids={r['coco_id'] for r in refs};old_ids=set(baseline['cohort_ids'])
    for rows in new.values():check_rows(rows,all_ids)
    for rows in baseline['models'].values():check_rows(rows,old_ids)
    args.output.mkdir(parents=True,exist_ok=True)
    ptb=PTBTokenizer();summaries={};per={}
    for cohort,references,conditions in [
        ('common4000',old_refs,{**baseline['models'],**{k:[r for r in v if r['coco_id'] in old_ids] for k,v in new.items()}}),
        ('full5000',refs,new)]:
        refs_ptb=ptb.tokenize({r['coco_id']:[dict(caption=s) for s in r['references']] for r in references})
        summaries[cohort]={};per[cohort]={}
        for name,rows in conditions.items():
            rows=sorted(rows,key=lambda r:r['coco_id']);check_rows(rows,{r['coco_id'] for r in references})
            label=cohort+'_'+name;save(args.output/(label+'_captions.json'),rows)
            hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
            cider,values=Cider().compute_score(refs_ptb,hyp);bleu,_=Bleu(4).compute_score(refs_ptb,hyp);rouge,_=Rouge().compute_score(refs_ptb,hyp)
            ci=args.output/(label+'_chair_input.json');co=args.output/(label+'_chair_output.json')
            save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
            subprocess.run([sys.executable,'-u',str(args.chair/'chair.py'),'--cap_file',str(ci),'--cache',str(args.chair/'chair.pkl'),
                '--save_path',str(co)],cwd=args.chair,check=True,timeout=600)
            ch=json.loads(co.read_text(encoding='utf-8'));metric,counts=chair_counts(ch,rows)
            point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metric,counts=counts,
                mean_words=float(np.mean([len(r['caption'].split()) for r in rows])))
            if any(not math.isfinite(v) for k,v in point.items() if k!='counts'):raise ValueError('Nonfinite metric:'+label)
            if cohort=='common4000' and name in baseline['models']:
                expected=baseline['expected_metrics'][name]
                for key in ('CIDEr','CHAIRs','CHAIRi','Recall','Bleu_4','ROUGE_L','mean_words'):
                    if abs(point[key]-expected[key])>1e-7:raise ValueError('Historical evaluator replay differs:'+name+'/'+key)
                if any(counts[k]!=v for k,v in expected['counts'].items()):raise ValueError('Historical CHAIR raw counts differ:'+name)
            if cohort=='full5000':
                expected=result['history'][int(name[3:])-1]['cider']
                if abs(float(cider)-expected)>1e-7:raise ValueError('CSA stored CIDEr replay differs:'+name)
            sentences={r['image_id']:r for r in ch['sentences']}
            vectors=[dict(coco_id=r['coco_id'],CIDEr=float(v),hallucination=bool(sentences[r['coco_id']]['mscoco_hallucinated_words'])) for r,v in zip(rows,values)]
            save(args.output/(label+'_per_image.json'),vectors);per[cohort][name]=vectors
            summaries[cohort][name]=point;print('METRICS',label,point,flush=True)
    # Exploratory paired differences on one reused development cohort, not a new test.
    paired={};rng=np.random.default_rng(20261103)
    for before,after in [('A','CSA9'),('C','CSA9'),('CSA9','CSA10')]:
        a=per['common4000'][before];b=per['common4000'][after]
        if [x['coco_id'] for x in a]!=[x['coco_id'] for x in b]:raise ValueError('Paired ID alignment')
        d=np.array([y['CIDEr']-x['CIDEr'] for x,y in zip(a,b)]);means=[]
        for _ in range(100):means.extend(d[rng.integers(0,len(d),size=(100,len(d)))].mean(1).tolist())
        paired[after+'_minus_'+before]=dict(CIDEr_delta=float(d.mean()),paired95_CIDEr=np.quantile(means,[.025,.975]).tolist(),
            repaired=sum(x['hallucination'] and not y['hallucination'] for x,y in zip(a,b)),
            introduced=sum(not x['hallucination'] and y['hallucination'] for x,y in zip(a,b)),resamples=10000)
    save(args.output/'comparison.json',dict(status='SAVED_CAPTION_COMPARISON_COMPLETE',metrics=summaries,paired=paired,
        csa_loss_history=result['history'],old_comparable_val_nll=None,shared_word_types=manifest['shared_word_types'],
        loss_note='Different vocabularies/objectives; old comparable validation NLL unavailable. Do not rank raw training losses.',
        model_executed=False,checkpoints_loaded=False,new_generation=False,test_used=False,
        limits=['Historically reused development, not independent test.','Old greedy vs new beam5: compares configured pipelines, not isolated architecture.',
                'CHAIR measures objects, not general relations/attributes.','Best9 selected by previous CIDEr; not reselected using CHAIR.']))
    print('SAVED CAPTION COMPARISON COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('review','baseline','manifest','chair','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
