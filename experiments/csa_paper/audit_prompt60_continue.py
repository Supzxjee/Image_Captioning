"""Cached-result audit and conditional paired bootstrap; no model/scorer run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from real_bank_data import read
from saved_quality_data import check_rows,chair_counts
from prompt_scst_continue_data import SOURCE,SOURCES,PHASES,validate_state,select
from prompt_scst_rerank import fixed_rank
from weight_calibration_data import chair_counts as sentence_counts

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def audit(cache,version):
    stored=read(cache);f=stored['files'];checks=[];vectors={}
    def load(n):return json.loads(f[n])
    def check(name,ok):
        if not ok:raise ValueError(name)
        checks.append(name)
    def digest(n):return hashlib.sha256(f[n].encode()).hexdigest()
    result=load('result.json');state=load('progress.json');validate_state(state)
    check('complete',load('run_status.json')['complete'] and result['status']=='PROMPT_SCST_CONTINUE_RERANK_COMPLETE' and state['updates']==14161 and state['image_events']==113287 and state['validated']==list(PHASES))
    check('source',result['source']==SOURCES['PROMPT'] and not result['test_used'] and not result['oracle_used'])
    src=load('source_receipt.json')
    for n,h in src['sha256'].items():check('pin/'+n,hashlib.sha256(subprocess.check_output(['git','show',src['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
    for n,h in state['validation_files'].items():check('seal/'+n,digest(n)==h)
    for rank in (0,1):
        check('cuda/'+str(rank),load(f'cuda_runtime_rank{rank}.json')['cuda_available'] and load(f'cuda_runtime_rank{rank}.json')['device_count']==2)
        guard=load(f'preflight_rank{rank}.json');check('preflight/'+str(rank),guard['passed'] and guard['source_sha256']==SOURCES['PROMPT']['sha256'] and guard['updates_discarded'])
    identity=load('identity.json');prepared=load('prepared.json');previous=SOURCE['identity.json']['prepared']
    check('recipe',identity['lr']==5e-6 and identity['scst_weight']==.05 and identity['object_weight']==0 and identity['global_batch']==8 and not identity['dropout'])
    check('same_inputs',prepared['stage_sha256']==previous['stage_sha256'] and prepared['prompt_files']==previous['prompt_files'] and prepared['epoch_order_seed']==43 and prepared['CE_reference_rotation']==1)
    for n in ('validation_references.json','vocabulary.json'):
        check('input/'+n,digest(n)==prepared['files_sha256'][n]==previous['files_sha256'][n])
    check('different_epoch_schedule',all(prepared['files_sha256'][n]!=previous['files_sha256'][n] for n in ('events.npy','coco_ids.npy')))
    log=[json.loads(s) for s in f['training.jsonl'].splitlines()]
    check('training_cursor',len(log)==567 and [r['updates'] for r in log]==list(range(25,14161,25))+[14161] and log[-1]['image_events']==113287)
    check('sealed_training',digest('training.jsonl')==load('checkpoint_receipt.json')['training_log_sha256'])
    refs=load('validation_references.json');ids={r['coco_id'] for r in refs};ordered=sorted(ids)
    check('VALID5000',len(ids)==5000 and sum(len(r['references']) for r in refs)==25010)
    inputs=load('rerank_inputs.json');imap={r['coco_id']:r for r in inputs};scored={}
    rp=load('rerank_prepared.json');check('rerank_input',digest('rerank_inputs.json')==rp['input_sha256'] and digest('validation_references.json')==rp['references_sha256'])
    for rank in (0,1):
        name=f'rerank_scores_rank{rank}.jsonl';worker=load(f'rerank_worker{rank}.json');rows=[json.loads(s) for s in f[name].splitlines()]
        check('CLIP_worker/'+str(rank),worker['complete'] and worker['images']==2500 and worker['candidate_pairs']==37500 and digest(name)==worker['scores_sha256'] and [r['coco_id'] for r in rows]==[r['coco_id'] for r in inputs[rank::2]])
        scored.update({r['coco_id']:r for r in rows})
    for phase,step in PHASES.items():
        beams=load(f'evaluations/{phase}/candidates_rank0.json')+load(f'evaluations/{phase}/candidates_rank1.json');lookup={r['coco_id']:r for r in beams}
        if phase=='original':
            for n in ('val_rank0.json','val_rank1.json','candidates_rank0.json','candidates_rank1.json'):
                check('exact_original/'+n,digest('evaluations/original/'+n)==SOURCE['checkpoint_receipt.json']['validation_files']['evaluations/end/'+n])
        for policy,folder in [('decoder_rank0','evaluations'),('fixed_rerank','reranked')]:
            d=f'{folder}/{phase}/';rows=load(d+'val_rank0.json')+load(d+'val_rank1.json');check_rows(rows,ids)
            point=result['metrics'][phase][policy];ch=load(d+'chair_output.json');m,c=chair_counts(ch,rows)
            check(d+'CHAIR',c==point['counts'] and all(abs(v-point[k])<1e-12 for k,v in m.items()))
            check(d+'input',load(d+'chair_input.json')==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
            per={r['coco_id']:r['CIDEr'] for r in load(d+'cider_per_image.json')};check(d+'CIDEr',set(per)==ids and abs(np.mean(list(per.values()))-point['CIDEr'])<1e-12)
            sentences={r['image_id']:r for r in ch['sentences']}
            vectors[phase,policy]=np.array([[per[i],*sentence_counts(sentences[i])] for i in ordered],dtype=np.float64)
            if policy=='fixed_rerank':
                selected=load(d+'selected_ranks.json')
                for r in rows:
                    i=r['coco_id'];rank=fixed_rank(lookup[i],scored[i]['phases'][phase]['clipscore'],imap[i]['detected'])
                    check(d+'select/'+str(i),rank==selected[str(i)]==scored[i]['phases'][phase]['selected_rank'] and r['caption']==lookup[i]['candidates'][rank]['caption'])
            if phase=='original':check(d+'historical',all(abs(point[k]-SOURCE['expected_metrics'][policy][k])<1e-12 for k in ('CIDEr','CHAIRs','CHAIRi','Recall','NLL')))
        for rank in (0,1):
            w=load(f'evaluations/{phase}/worker{rank}.json');check(phase+'/cursor'+str(rank),w['updates']==step and w['images']==2500 and w['complete'])
    selected,eligible=select({p:m['fixed_rerank'] for p,m in result['metrics'].items()});lock=load('locked_selection.json')
    check('selection',lock==result['selection'] and lock['selected_name']==selected and lock['eligible_names']==eligible and selected=='original' and lock['sha256']==SOURCES['PROMPT']['sha256'])
    for p,n in (('mid','mid_receipt.json'),('end','checkpoint_receipt.json')):
        r=load(n);check('receipt/'+p,r['updates']==PHASES[p] and r['image_events']==min(PHASES[p]*8,113287))
    def aggregate(x):
        s=x.sum(axis=0);return np.array([x[:,0].mean()*100,s[1]/len(x)*100,s[2]/s[3]*100,s[4]/s[5]*100])
    comparisons={}
    for phase in ('mid','end'):
        for policy in ('decoder_rank0','fixed_rerank'):
            a=vectors['original',policy];b=vectors[phase,policy];rng=np.random.default_rng(20261108)
            draws=np.array([aggregate(b[ix])-aggregate(a[ix]) for ix in (rng.integers(0,5000,5000) for _ in range(2000))])
            comparisons[phase+'/'+policy]=dict(metric_order=['CIDEr_points','CHAIRs_pp','CHAIRi_pp','Recall_pp'],delta=(aggregate(b)-aggregate(a)).tolist(),CI95=np.quantile(draws,[.025,.975],axis=0).tolist(),
                flags_repaired=int(((a[:,1]>0)&(b[:,1]==0)).sum()),flags_introduced=int(((a[:,1]==0)&(b[:,1]>0)).sum()))
    report=dict(status='PROMPT60_CONTINUE_REVIEW_COMPLETE',source_version=version,source_notebook='ducanh3/notebook7bc83c4c76',archive_sha256=stored['archive_sha256'],checks=len(checks),metrics=result['metrics'],selection=lock,comparisons=comparisons,
        mid_sha256=load('mid_receipt.json')['sha256'],end_sha256=load('checkpoint_receipt.json')['sha256'],worker_seconds=load('training_complete.json')['seconds'],
        limits=['One seed/reused VALID; conditional fixed-corpus paired2000bootstrap, not training variance or new TEST.','Checkpoint binaries not in review ZIP: SHA and Adam/RNG restoration are runtime receipts/pinned code, not independently loaded weights.','No local model, CUDA or official scorer execution.'])
    (HERE/'PROMPT60_CONTINUE_RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS',len(checks),'checks');print(json.dumps(comparisons,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('--version',type=int);a=p.parse_args();audit(a.cache,a.version)
