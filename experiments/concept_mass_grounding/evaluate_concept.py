"""Official CPU caption metrics plus exploratory paired intervals; no neural execution."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from concept_data import *
def paired_ci(values):
    values=np.asarray(values,dtype='float64');rng=np.random.default_rng(20261029);means=[]
    for _ in range(10):means.extend(values[rng.integers(0,len(values),(1000,len(values)))].mean(1).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    out=Path(cfg['output']);meta=json.loads((out/'metadata.json').read_bytes());prep=json.loads((out/'preparation.json').read_bytes())
    workers=[json.loads((out/f'worker_rank{r}.json').read_bytes()) for r in (0,1)]
    if any(w['status']!='ENDPOINT_500_COMPLETE' or w['updates']!=500 or w['global_negative_events']<=0 for w in workers):raise ValueError('Incomplete or zero-exposure method')
    for k in ('initial_digest','final_digest','initial_frozen_digest','final_frozen_digest'):
        if workers[0][k]!=workers[1][k]:raise ValueError('Rank-state mismatch')
    if workers[0]['initial_frozen_digest']!=workers[0]['final_frozen_digest']:raise ValueError('Frozen state drift')
    expected={meta['rows'][i]['image_id']:meta['rows'][i] for i in meta['dev_indices']};ptb=PTBTokenizer()
    refs=ptb.tokenize({i:[dict(caption=s) for s in row['references']] for i,row in sorted(expected.items())})
    metrics={};per_cider={};chair_flags={};meteor=Meteor()
    for condition in ('A','end'):
        rows=sorted(sum([json.loads((out/f'{condition}_rank{r}.json').read_bytes()) for r in (0,1)],[]),key=lambda r:r['image_id'])
        if len(rows)!=1000 or {r['image_id']:r['coco_id'] for r in rows}!={i:r['coco_id'] for i,r in expected.items()}:raise ValueError('Changed prediction IDs')
        save(out/f'{condition}_predictions.json',rows);hyp=ptb.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in rows})
        cider,per=Cider().compute_score(refs,hyp);bleu,_=Bleu(4).compute_score(refs,hyp);met,_=meteor.compute_score(refs,hyp);rouge,_=Rouge().compute_score(refs,hyp)
        per_cider[condition]=np.array(per);save(out/f'{condition}_per_image_cider.json',[dict(image_id=r['image_id'],coco_id=r['coco_id'],CIDEr=float(v)) for r,v in zip(rows,per)])
        save(out/f'{condition}_chair_input.json',[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(out/f'{condition}_chair_input.json'),
            '--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(out/f'{condition}_chair_output.json')],cwd=cfg['chair'],check=True)
        chair=json.loads((out/f'{condition}_chair_output.json').read_bytes());sentences=chair['sentences']
        if len(sentences)!=1000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in rows}:raise ValueError('CHAIR IDs differ')
        count=dict(hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
            hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
            recalled_unique=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),reference_unique=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        ch=dict(CHAIRs=count['hallucinated_captions']/1000,CHAIRi=count['hallucinated_mentions']/max(1,count['object_mentions']),Recall=count['recalled_unique']/max(1,count['reference_unique']))
        if any(abs(v-chair['overall_metrics'][k])>1e-12 for k,v in ch.items()):raise ValueError('Metric aggregation mismatch')
        flag_by={s['image_id']:int(bool(s['mscoco_hallucinated_words'])) for s in sentences};chair_flags[condition]=np.array([flag_by[r['coco_id']] for r in rows])
        metrics[condition]=dict(CIDEr=float(cider),Bleu_1=bleu[0],Bleu_2=bleu[1],Bleu_3=bleu[2],Bleu_4=bleu[3],METEOR=met,ROUGE_L=rouge,chair=ch,counts=count,mean_words=float(np.mean([len(r['caption'].split()) for r in rows])))
        if condition=='A':
            historical={r['image_id']:r['caption'] for r in meta['baseline']}
            if any(r['caption']!=historical[r['image_id']] for r in rows) or abs(cider-1.2055125991020814)>1e-7:raise ValueError('A baseline/evaluator drift')
    diagnostics=sum([json.loads((out/f'diagnostic_rank{r}.json').read_bytes())['records'] for r in (0,1)],[])
    per_class={}
    for c in sorted({r['category'] for r in diagnostics}):
        result={}
        for state in ('supported','weak_negative'):
            pool=[r for r in diagnostics if r['category']==c and r['state']==state]
            result[state]=dict(n=len(pool),A_mass=float(np.mean([r['A_mass'] for r in pool])),end_mass=float(np.mean([r['end_mass'] for r in pool])))
        result['A_separation']=result['supported']['A_mass']-result['weak_negative']['A_mass']
        result['end_separation']=result['supported']['end_mass']-result['weak_negative']['end_mass'];per_class[c]=result
    save(out/'class_conditioning.json',dict(per_class=per_class,macro_A_separation=float(np.mean([r['A_separation'] for r in per_class.values()])) if per_class else None,
        macro_end_separation=float(np.mean([r['end_separation'] for r in per_class.values()])) if per_class else None,limits='Weak-label diagnostic, fixed neutral prefix; not physical causal grounding'))
    summary=dict(status='ENDPOINT_500_EVALUATED_WAIT_MATCHED_C_G',arm=cfg['arm'],updates=500,policy=POLICY,metrics=metrics,
        end_minus_A_CIDEr=float((per_cider['end']-per_cider['A']).mean()),end_minus_A_CIDEr_paired95_ci=paired_ci(per_cider['end']-per_cider['A']),
        end_minus_A_CHAIRs=float((chair_flags['end']-chair_flags['A']).mean()),end_minus_A_CHAIRs_paired95_ci=paired_ci(chair_flags['end']-chair_flags['A']),
        initial_digest=workers[0]['initial_digest'],final_digest=workers[0]['final_digest'],frozen_digest=workers[0]['final_frozen_digest'],
        checkpoint_sha256=sha(out/'checkpoint.bin'),source_commit=cfg['source_commit'],source_assets=cfg['source_assets'],
        tensor_content_sha256=prep['tensor_content_sha256'],preparation_payload_sha256=prep['payload_sha256'],global_negative_events=workers[0]['global_negative_events'],global_positive_events=workers[0]['global_positive_events'],
        historical_A_caption_mismatches=0,test_used=False,gpu_phase_device_hours=sum(w['elapsed_seconds'] for w in workers)/3600,
        limits=['One seed, reused dev1000. No automated checkpoint promotion or acceptance/novelty claim.','Absence is weak annotation supervision, not proven physical absence; metric shortcut and missed objects remain risks.'])
    save(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
