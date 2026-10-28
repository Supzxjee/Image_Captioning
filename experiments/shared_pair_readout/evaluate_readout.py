"""CPU caption evaluation after both ranks finish, without loading a model."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from readout_data import *


def paired_ci(values):
    values=np.asarray(values,dtype='float64');rng=np.random.default_rng(20261028);means=[]
    for _ in range(10):means.extend(values[rng.integers(0,len(values),(1000,len(values)))].mean(1).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    out=Path(cfg['output']);meta=json.loads((out/'metadata.json').read_bytes())
    prep=json.loads((out/'preparation.json').read_bytes())
    done=[json.loads((out/f'worker_rank{r}.json').read_bytes()) for r in (0,1)]
    if any(d['status']!='ENDPOINT_500_COMPLETE' or d['updates']!=500 for d in done):raise ValueError('Both ranks must finish endpoint500')
    for key in ('initial_digest','final_digest','initial_frozen_digest','final_frozen_digest'):
        if done[0][key]!=done[1][key]:raise ValueError('Rank state differs: '+key)
    if done[0]['initial_frozen_digest']!=done[0]['final_frozen_digest']:raise ValueError('Frozen state drift')
    if sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Changed CHAIR cache')
    expected={meta['rows'][i]['image_id']:meta['rows'][i] for i in meta['dev_indices']}
    ptb=PTBTokenizer();refs=ptb.tokenize({i:[dict(caption=s) for s in r['references']] for i,r in sorted(expected.items())})
    metrics={};per_cider={};predictions={};all_chair={}
    meteor=Meteor()
    for condition in ('A','end','off'):
        rows=sorted(sum([json.loads((out/f'{condition}_rank{r}.json').read_bytes()) for r in (0,1)],[]),key=lambda r:r['image_id'])
        if len(rows)!=1000 or len({r['image_id'] for r in rows})!=1000 or {r['image_id'] for r in rows}!=set(expected):raise ValueError('Dev IDs differ')
        if any(r['coco_id']!=expected[r['image_id']]['coco_id'] for r in rows):raise ValueError('Dev COCO IDs differ')
        predictions[condition]=rows;save(out/f'{condition}_predictions.json',rows)
        hyp=ptb.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in rows})
        cider,per=Cider().compute_score(refs,hyp);bleu,_=Bleu(4).compute_score(refs,hyp)
        met,_=meteor.compute_score(refs,hyp);rouge,_=Rouge().compute_score(refs,hyp)
        per_cider[condition]=np.asarray(per,dtype='float64')
        save(out/f'{condition}_per_image_cider.json',[dict(image_id=r['image_id'],coco_id=r['coco_id'],CIDEr=float(v)) for r,v in zip(rows,per)])
        save(out/f'{condition}_chair_input.json',[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(out/f'{condition}_chair_input.json'),
            '--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(out/f'{condition}_chair_output.json')],cwd=cfg['chair'],check=True)
        chair=json.loads((out/f'{condition}_chair_output.json').read_bytes());sentences=chair['sentences']
        if len(sentences)!=1000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in rows}:raise ValueError('CHAIR IDs differ')
        counts=dict(hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
            hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),
            object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
            recalled_unique=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),
            reference_unique=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        values=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=counts['hallucinated_mentions']/max(1,counts['object_mentions']),Recall=counts['recalled_unique']/max(1,counts['reference_unique']))
        if any(abs(v-chair['overall_metrics'][k])>1e-12 for k,v in values.items()):raise ValueError('CHAIR aggregation differs')
        metrics[condition]=dict(CIDEr=float(cider),Bleu_1=float(bleu[0]),Bleu_2=float(bleu[1]),Bleu_3=float(bleu[2]),Bleu_4=float(bleu[3]),METEOR=float(met),ROUGE_L=float(rouge),
            chair=values,counts=counts,mean_words=float(np.mean([len(r['caption'].split()) for r in rows])))
        all_chair[condition]=sentences
    historical={r['image_id']:r['caption'] for r in meta['baseline']}
    drift=sum(r['caption']!=historical[r['image_id']] for r in predictions['A'])
    if not drift and abs(metrics['A']['CIDEr']-1.205512599)>1e-7:raise ValueError('Same A captions but evaluator drifted')
    held_expected={meta['rows'][i]['coco_id']:meta['rows'][i] for i in meta['held_indices']}
    held={}
    for condition in ('A','end'):
        rows=sorted(sum([json.loads((out/f'held_{condition}_rank{r}.json').read_bytes()) for r in (0,1)],[]),key=lambda r:r['coco_id'])
        if len(rows)!=128 or {r['coco_id'] for r in rows}!=set(held_expected):raise ValueError('Heldout caption IDs differ')
        held[condition]={r['coco_id']:r['caption'] for r in rows};save(out/f'held_{condition}_predictions.json',rows)
    scores=sum([json.loads((out/f'held_likelihood_rank{r}.json').read_bytes()) for r in (0,1)],[])
    if len(scores)!=64 or len({r['pair_id'] for r in scores})!=64 or {r['pair_id'] for r in scores}!={p['pair_id'] for p in meta['pairs']['heldout']}:raise ValueError('Heldout likelihood IDs differ')
    save(out/'held_likelihood.json',scores)
    panel=[];selected=set(meta['semantic_panel_pair_ids'])
    if len(selected)!=32 or selected!=set(panel_ids(meta['pairs']['heldout'])):raise ValueError('Preselected panel changed')
    for p in meta['pairs']['heldout']:
        if p['pair_id'] not in selected:continue
        panel.append(dict(pair_id=p['pair_id'],stratum=p['stratum'],statement=p['statement'],
            positive=dict(coco_id=p['positive'],filename=held_expected[p['positive']]['filename'],captions={c:held[c][p['positive']] for c in held}),
            negative=dict(coco_id=p['negative'],filename=held_expected[p['negative']]['filename'],captions={c:held[c][p['negative']] for c in held})))
    save(out/'semantic_panel.json',dict(preselected=True,pairs=panel,human_scores_available=False))
    summary=dict(status='ENDPOINT_500_EVALUATED_WAIT_MATCHED_I_J',arm=cfg['arm'],updates=500,policy=POLICY,
        metrics=metrics,end_minus_A_CIDEr=metrics['end']['CIDEr']-metrics['A']['CIDEr'],
        end_minus_A_paired95_ci=paired_ci(per_cider['end']-per_cider['A']),
        initial_digest=done[0]['initial_digest'],final_digest=done[0]['final_digest'],
        frozen_digest=done[0]['final_frozen_digest'],checkpoint_sha256=sha(out/'checkpoint.bin'),
        source_commit=cfg['source_commit'],source_assets=cfg['source_assets'],
        tensor_content_sha256=prep['tensor_content_sha256'],metadata_sha256=prep['payload_sha256']['metadata.json'],
        object_targets_sha256=prep['payload_sha256']['object_targets.json'],historical_A_caption_mismatches=drift,
        held_tail_ranking=float(np.mean([r['tail_gap']>0 for r in scores])),
        held_predicate_ranking=float(np.mean([r['predicate_gap']>0 for r in scores])),
        sensitivity_changed_captions=sum(a['caption']!=b['caption'] for a,b in zip(predictions['end'],predictions['off'])),
        worker_timings=[d['timings'] for d in done],gpu_phase_device_hours=sum(d['elapsed_seconds'] for d in done)/3600,
        test_used=False,semantic_panel_scores='PENDING_ASSISTED_BLIND_REVIEW',
        limits=['One seed, reused development/adaptation-heldout data; exploratory, no independent confirmation.',
            'Attention and zero-reader sensitivity do not prove semantic correctness or physical causal grounding.',
            'Compare I/J only with identical same-runtime A captions and input content identities. Historical A never substituted.',
            'Application-specific readout change, not new attention/OT primitive; no automatic novelty/acceptance claim.'])
    save(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
