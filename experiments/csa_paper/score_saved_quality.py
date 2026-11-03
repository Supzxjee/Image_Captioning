"""USER-RUN CPU metrics on saved captions; no neural imports or checkpoint reads."""
import argparse
import gzip
import hashlib
import json
import math
import inspect
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import zipfile
from saved_quality_data import check_rows,common_refs,chair_counts,align_cider_scores

CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(args):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    import numpy as np
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if sha(args.review)!=manifest['new_review_zip_sha256'] or sha(args.baseline)!=manifest['payload_sha256']:raise ValueError('Changed review/baseline package')
    if sha(args.chair/'chair.pkl')!=CHAIR_SHA:raise ValueError('Wrong CHAIR cache')
    baseline=json.loads(gzip.decompress(args.baseline.read_bytes()))
    with zipfile.ZipFile(args.review) as z:
        def read(n):return json.loads(z.read(n))
        result=read('result.json');refs=read('validation_references.json');stored_refs_ptb=read('validation_ptb.json')
        if result['status']!='FULL_XE_10_COMPLETE' or result['best_epoch']!=9:raise ValueError('Wrong completed CSA run')
        new={f'CSA{epoch}':read(f'val_epoch{epoch:02d}_rank0.json')+read(f'val_epoch{epoch:02d}_rank1.json') for epoch in (9,10)}
    old_refs=common_refs(refs,baseline);all_ids={r['coco_id'] for r in refs};old_ids=set(baseline['cohort_ids'])
    for rows in new.values():check_rows(rows,all_ids)
    for rows in baseline['models'].values():check_rows(rows,old_ids)
    args.output.mkdir(parents=True,exist_ok=True)
    ptb=PTBTokenizer();summaries={};per={};mismatches=[]
    save(args.output/'evaluator_provenance.json',dict(packages={k:importlib.metadata.version(k) for k in ('pycocoevalcap','numpy','nltk')},
        tokenizer_sha256=sha(Path(inspect.getfile(PTBTokenizer))),cider_sha256=sha(Path(inspect.getfile(Cider))),cider_scorer_sha256=sha(Path(inspect.getfile(CiderScorer))),
        original_legacy_keys_order_restored=True,full5000_original_rank_concat_order=True))
    for cohort,references,conditions in [
        ('common4000',old_refs,{**baseline['models'],**{k:[r for r in v if r['coco_id'] in old_ids] for k,v in new.items()}}),
        ('full5000',refs,new)]:
        metric_id=(lambda i:baseline['legacy_image_ids'][str(i)]) if cohort=='common4000' else (lambda i:i)
        refs_ptb=ptb.tokenize({metric_id(r['coco_id']):[dict(caption=s) for s in r['references']] for r in references})
        save(args.output/(cohort+'_references_ptb.json'),refs_ptb)
        drift=[r['coco_id'] for r in references if refs_ptb[metric_id(r['coco_id'])]!=stored_refs_ptb[str(r['coco_id'])][:len(r['references'])]]
        save(args.output/(cohort+'_reference_tokenization_check.json'),dict(images_differing_from_stored_CSA_tokens=len(drift),first20_ids=drift[:20],
            note='Diagnostics only: old4000 PTB output was not archived. No replay tolerance changed.'))
        summaries[cohort]={};per[cohort]={}
        for name,rows in conditions.items():
            if cohort=='common4000':rows=sorted(rows,key=lambda r:metric_id(r['coco_id']))
            check_rows(rows,{r['coco_id'] for r in references})
            label=cohort+'_'+name;save(args.output/(label+'_captions.json'),rows)
            hyp=ptb.tokenize({metric_id(r['coco_id']):[dict(caption=r['caption'])] for r in rows})
            save(args.output/(label+'_hypotheses_ptb.json'),hyp)
            cider,values=Cider().compute_score(refs_ptb,hyp);bleu,_=Bleu(4).compute_score(refs_ptb,hyp);rouge,_=Rouge().compute_score(refs_ptb,hyp)
            ci=args.output/(label+'_chair_input.json');co=args.output/(label+'_chair_output.json')
            save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
            subprocess.run([sys.executable,'-u',str(args.chair/'chair.py'),'--cap_file',str(ci),'--cache',str(args.chair/'chair.pkl'),
                '--save_path',str(co)],cwd=args.chair,check=True,timeout=600)
            ch=json.loads(co.read_text(encoding='utf-8'));metric,counts=chair_counts(ch,rows)
            point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metric,counts=counts,
                mean_words=float(np.mean([len(r['caption'].split()) for r in rows])))
            if any(not math.isfinite(v) for k,v in point.items() if k!='counts'):raise ValueError('Nonfinite metric:'+label)
            expected=None;failures=[]
            if cohort=='common4000' and name in baseline['models']:
                expected=baseline['expected_metrics'][name]
                for key in ('CIDEr','CHAIRs','CHAIRi','Recall','Bleu_4','ROUGE_L','mean_words'):
                    if abs(point[key]-expected[key])>1e-7:failures.append(dict(condition=label,metric=key,expected=expected[key],actual=point[key],delta=point[key]-expected[key]))
                for key,value in expected['counts'].items():
                    if counts[key]!=value:failures.append(dict(condition=label,metric='counts.'+key,expected=value,actual=counts[key],delta=counts[key]-value))
            if cohort=='full5000':
                stored=result['history'][int(name[3:])-1]['cider'];expected=dict(CIDEr=stored)
                if abs(float(cider)-stored)>1e-7:failures.append(dict(condition=label,metric='CIDEr',expected=stored,actual=float(cider),delta=float(cider)-stored))
            mismatches.extend(failures)
            save(args.output/(label+'_metrics.json'),dict(point=point,historical_expected=expected,replay_mismatches=failures))
            save(args.output/'replay_mismatches.json',mismatches)
            sentences={r['image_id']:r for r in ch['sentences']}
            # Cider returns values in reference-dictionary order, which may differ from hypothesis insertion order.
            aligned=align_cider_scores(references,values,rows)
            vectors=[dict(coco_id=r['coco_id'],CIDEr=v,hallucination=bool(sentences[r['coco_id']]['mscoco_hallucinated_words'])) for r,v in zip(rows,aligned)]
            save(args.output/(label+'_per_image.json'),vectors);per[cohort][name]=vectors
            summaries[cohort][name]=point;print('METRICS',label,point,'REPLAY',failures or 'PASS',flush=True)
    # Exploratory paired differences on one reused development cohort, not a new test.
    paired={};rng=np.random.default_rng(20261103)
    cider_valid=not any(x['metric']=='CIDEr' for x in mismatches)
    chair_valid=not any(x['metric'] in ('CHAIRs','CHAIRi','Recall') or x['metric'].startswith('counts.') for x in mismatches)
    for before,after in ([('A','CSA9'),('C','CSA9'),('CSA9','CSA10')] if cider_valid else []):
        a=per['common4000'][before];b=per['common4000'][after]
        if [x['coco_id'] for x in a]!=[x['coco_id'] for x in b]:raise ValueError('Paired ID alignment')
        d=np.array([y['CIDEr']-x['CIDEr'] for x,y in zip(a,b)]);means=[]
        for _ in range(100):means.extend(d[rng.integers(0,len(d),size=(100,len(d)))].mean(1).tolist())
        paired[after+'_minus_'+before]=dict(CIDEr_delta=float(d.mean()),paired95_CIDEr=np.quantile(means,[.025,.975]).tolist(),
            repaired=sum(x['hallucination'] and not y['hallucination'] for x,y in zip(a,b)),
            introduced=sum(not x['hallucination'] and y['hallucination'] for x,y in zip(a,b)),resamples=10000)
    save(args.output/'comparison.json',dict(status='SAVED_CAPTION_COMPARISON_COMPLETE' if not mismatches else 'METRIC_REPLAY_FAILED',metrics=summaries,paired=paired,
        replay_mismatches=mismatches,cider_replay_valid=cider_valid,chair_replay_valid=chair_valid,
        csa_loss_history=result['history'],old_comparable_val_nll=None,shared_word_types=manifest['shared_word_types'],
        loss_note='Different vocabularies/objectives; old comparable validation NLL unavailable. Do not rank raw training losses.',
        model_executed=False,checkpoints_loaded=False,new_generation=False,test_used=False,
        limits=['Historically reused development, not independent test.','Old greedy vs new beam5: compares configured pipelines, not isolated architecture.',
                'CHAIR measures objects, not general relations/attributes.','Best9 selected by previous CIDEr; not reselected using CHAIR.']))
    if mismatches:raise ValueError('Metric replay mismatch; all raw outputs preserved, no tolerance relaxed: '+json.dumps(mismatches))
    print('SAVED CAPTION COMPARISON COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('review','baseline','manifest','chair','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
