"""Matched endpoint metrics and prelocked decisions; no test data."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from detail_data import *


def paired(a,b):
    delta=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(20261023);draws=[]
    for _ in range(200):draws.extend(delta[rng.integers(0,len(delta),(100,len(delta)))].mean(1).tolist())
    return dict(mean=float(delta.mean()),paired95=np.quantile(draws,[.025,.975]).tolist(),resamples=20000,seed=20261023)


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    out=Path(cfg['output']);done=[json.loads((out/f'worker_complete_rank{r}.json').read_text()) for r in (0,1)]
    if any(d['status']!='COMPLETE' or d['updates']!=5000 or d['final_base_sha256']!=BASE_SHA for d in done) or done[0]['final_head_sha256']!=done[1]['final_head_sha256']:raise ValueError('Incomplete/different endpoint.')
    import torch
    from detail_model import DetailAdapter
    from captioning.scst_runtime import model_digest
    state=torch.load(out/'head.bin',map_location='cpu',weights_only=False);head=DetailAdapter();head.load_state_dict(state['head'],strict=True)
    if state['completed_updates']!=5000 or state['source_A_sha256']!=A_SHA or model_digest(head)!=done[0]['final_head_sha256']:raise ValueError('Saved head is not the audited endpoint.')
    histories=[json.loads((out/f'history_rank{r}.json').read_text()) for r in (0,1)];expected_train=cfg['groups']['train']
    by_id={r['image_id']:i for i,r in enumerate(expected_train)};counts=Counter()
    for h in histories:
        if len(h)!=5000 or [r['step'] for r in h]!=list(range(1,5001)):raise ValueError('Wrong update sequence.')
        for row in h:
            for iid,c in zip(row['image_ids'],row['conditions']):
                if c!=cfg['schedule'][row['epoch']][by_id[iid]]:raise ValueError('Condition schedule changed.')
                counts[(iid,c)]+=1
    if set(i for i,_ in counts)!=set(by_id) or any(counts[(i,c)]!=2 for i in by_id for c in (0,1)):raise ValueError('Each train image must see Htwice/Ltwice.')
    for epoch in range(4):
        ids=[i for h in histories for r in h if r['epoch']==epoch for i in r['image_ids']]
        if len(ids)!=10000 or len(set(ids))!=10000 or set(ids)!=set(by_id):raise ValueError('Epoch image coverage mismatch.')
    save(out/'exposure_check.json',dict(status='PASS',epochs=4,updates=5000,image_exposures=40000,H=20000,L=20000,H_and_L_per_image=2))
    data=json.loads(Path(cfg['dataset_json']).read_text());rows=cfg['groups']['val'];expected={r['image_id']:r for r in rows};ids=sorted(expected)
    tokenizer=PTBTokenizer();refs=tokenizer.tokenize({i:[dict(caption=s['raw']) for s in data['images'][i]['sentences'][:5]] for i in ids})
    scores={};ciders={}
    for condition in ('A','L','H'):
        predictions=sum([json.loads((out/f'{condition}_rank{r}.json').read_text()) for r in (0,1)],[]);predictions.sort(key=lambda r:r['image_id'])
        if len(predictions)!=5000 or [r['image_id'] for r in predictions]!=ids:raise ValueError('Wrong/duplicated val predictions.')
        save(out/f'{condition}_predictions.json',predictions)
        hyp=tokenizer.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in predictions});score,values=Cider().compute_score(refs,hyp);ciders[condition]=values
        save(out/f'{condition}_per_image_cider.json',[dict(image_id=i,CIDEr=float(v)) for i,v in zip(ids,values)])
        chair_in=out/f'{condition}_chair_input.json';chair_out=out/f'{condition}_chair_output.json'
        save(chair_in,[dict(image_id=expected[r['image_id']]['coco_id'],caption=r['caption']) for r in predictions])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(chair_in),'--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(chair_out)],cwd=cfg['chair'],check=True)
        chair=json.loads(chair_out.read_text());sentences=chair['sentences']
        if len(sentences)!=5000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in rows}:raise ValueError('CHAIR identity differs.')
        halluc=sum(len(s['mscoco_hallucinated_words']) for s in sentences);mentions=sum(len(s['mscoco_generated_words']) for s in sentences)
        correct=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences);total=sum(len(set(s['mscoco_gt_words'])) for s in sentences)
        reagg=dict(CHAIRs=sum(bool(s['mscoco_hallucinated_words']) for s in sentences)/5000,CHAIRi=halluc/mentions if mentions else 0.,Recall=correct/total if total else 0.)
        if any(abs(reagg[k]-chair['overall_metrics'][k])>1e-12 for k in reagg):raise ValueError('CHAIR reaggregation mismatch.')
        scores[condition]=dict(CIDEr=float(score),**reagg,mean_words=float(np.mean([len(r['caption'].split()) for r in predictions])),mean_object_mentions=mentions/5000)
        save(out/f'{condition}_metrics.json',scores[condition])
    comparisons=dict(H_minus_A=paired(ciders['H'],ciders['A']),H_minus_L=paired(ciders['H'],ciders['L']))
    h,a=scores['H'],scores['A'];gate=comparisons['H_minus_A']['mean']>=.01 and comparisons['H_minus_A']['paired95'][0]>0 and comparisons['H_minus_L']['mean']>=.005 and comparisons['H_minus_L']['paired95'][0]>0 and h['CHAIRs']<=a['CHAIRs'] and h['CHAIRi']<=a['CHAIRi'] and h['Recall']>=a['Recall']-.005
    save(out/'run_summary.json',dict(status='PILOT_COMPLETE',metrics=scores,paired_CIDEr=comparisons,investment_gate=bool(gate),
        selection='Fixed5000-update endpoint;raw greedy;no rerank or val checkpoint selection',test_used=False,
        head_file_sha256=digest(out/'head.bin'),protocol_sha256=digest(out/'protocol.json'),final_head_sha256=done[0]['final_head_sha256'],source_A_sha256=A_SHA,
        interpretation='Gain supports this shared-adapter native-detail processing path only; not verified lexical/relation grounding or novelty.',
        limits=['One seed,historically reusedval5000;bootstrapCI is image sampling,not training-seed uncertainty.',
                'Fixedtiles/pooling and mixedH/Ltraining constrain negative conclusions;no automatic sweep/scale if gate fails.']))
