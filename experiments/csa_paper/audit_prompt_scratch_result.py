"""Audit cached review text, counts and hashes; no neural or metric execution."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from full_xe_data import validate_state,validate_predictions,learning_rate

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def audit(cache):
    count=0;arms={};checks=[]
    def check(label,condition):
        nonlocal count
        if not condition:raise ValueError(label)
        count+=1;checks.append(label)
    for arm in ('CONTROL','PROMPT'):
        record=json.loads((Path(cache)/(arm+'.json')).read_text(encoding='utf-8'));files=record['files']
        read=lambda n:json.loads(files[n])
        result=read('result.json');identity=read('input_identity.json');source=read('source_receipt.json');state=read('progress.json')
        check(arm+'/complete',read('run_status.json')==dict(complete=True,arm=arm,requested_epochs=10,test_used=False) and result['status']=='PROMPT_SCRATCH_XE10_COMPLETE')
        check(arm+'/full10',result['completed_epochs']==10 and result['updates']==177110 and result['caption_events']==5667470 and result['train_images']==113287 and result['val_images']==5000 and result['test_images']==0 and not result['rl'])
        validate_state(state);check(arm+'/state',state['history']==result['history'] and state['epoch']==11 and result['initial_updates_this_session']==0)
        check(arm+'/source',source['commit']=='8420162b6760af0b6cb2199ba07e8e511b4a2509' and len(source['sha256'])==50 and source['sha256']==identity['source'])
        for n,h in source['sha256'].items():check(arm+'/source/'+n,hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
        refs=read('validation_references.json');ids={r['coco_id'] for r in refs};vocab=read('vocabulary.json')
        check(arm+'/refs',len(refs)==len(ids)==5000 and sum(len(r['references']) for r in refs)==25010 and len(vocab)==10000)
        for n,key in [('validation_references.json','validation_references_sha256'),('validation_ptb.json','ptb_sha256')]:check(arm+'/'+n,hashlib.sha256(files[n].encode()).hexdigest()==identity[key])
        check(arm+'/guards',read('prompt_preflight_result.json')==dict(passed=True,cases=4,arm=arm,training_started=False,guard_weights_discarded=True) and len(read('prompt_preflight.json'))==4 and all(r['passed'] for r in read('prompt_preflight.json')) and read('beam_preflight.json')['passed'] and read('metric_preflight.json')['passed'])
        epochs=[];predictions={}
        for e,h in enumerate(result['history'],1):
            rows=[];metrics=read(f'metrics_epoch{e:02d}.json')
            check(arm+f'/epoch{e}/schedule',h['epoch']==e and h['updates']==e*17711 and h['val_target_tokens']==286871 and h['validation_images']==5000 and h['references']==25010 and h['beam']==5 and h['max_new_tokens']==20 and all(h[k]==v for k,v in metrics.items()))
            for rank in (0,1):
                n=f'val_epoch{e:02d}_rank{rank}.json';r=read(n)
                check(arm+'/'+n,hashlib.sha256(files[n].encode()).hexdigest()==h['prediction_sha256'][n] and len(r)==2500);rows+=r
            validate_predictions(rows,refs)
            for row in rows:
                tokens=row['ids'];end=tokens.index(2) if 2 in tokens else len(tokens)
                if not 1<=len(tokens)<=20 or any(type(t)!=int or not 0<=t<len(vocab) for t in tokens) or any(t in (0,1) for t in tokens[:end]) or any(tokens[end+1:]) or row['eos']!=(2 in tokens) or row['caption']!=' '.join(vocab[t] for t in tokens[:end]):raise ValueError('Caption/token content differs')
            check(arm+f'/epoch{e}/captions',sum(r['eos'] for r in rows)==h['eos_images'] and abs(sum(len(r['caption'].split()) for r in rows)/5000-h['mean_words'])<1e-12)
            epochs.append(dict(epoch=e,CIDEr_x100=100*h['cider'],val_nll=h['val_nll'],train_nll=h['observed_train_nll'],mean_words=h['mean_words']))
            predictions[e]=rows
        best=max(result['history'],key=lambda h:h['cider']);br=read('best_receipt.json');cr=read('checkpoint_receipt.json')
        check(arm+'/best',br['validated_epoch']==best['epoch']==result['best_epoch'] and br['cider']==best['cider']==result['best_cider'] and br['updates']==best['updates'])
        check(arm+'/latest',cr['epoch']==11 and cr['updates']==177110 and cr['caption_events']==5667470 and cr['batch_cursor']==0)
        training=[ast.literal_eval(m.group(1)) for m in re.finditer(r'^TRAIN (\{.*\})$',files['notebook.log'],re.M)]
        check(arm+'/training_log',len(training)==1771 and [r['step'] for r in training]==list(range(100,177111,100)) and all(abs(r['lr']-learning_rate(r['step']))<1e-14 and math.isfinite(r['nll']) and math.isfinite(r['grad_norm']) for r in training))
        arms[arm]=dict(record=record,result=result,identity=identity,epochs=epochs,predictions=predictions,best=br,latest=cr)
    a,b=[arms[k] for k in ('CONTROL','PROMPT')]
    check('matched_identity_except_arm',{k:v for k,v in a['identity'].items() if k!='arm'}=={k:v for k,v in b['identity'].items() if k!='arm'})
    for name in ('validation_references.json','validation_ptb.json','vocabulary.json','bank_identity.json','prompt_stage.json'):check('matched/'+name,a['record']['files'][name]==b['record']['files'][name])
    output=dict(status='PROMPT_SCRATCH_XE10_REVIEW_COMPLETE',metadata_checks=count,checks=checks,source_commit='8420162b6760af0b6cb2199ba07e8e511b4a2509',test_used=False,chair_measured=False,model_or_scorer_executed=False,
        sources={'CONTROL':'ducanh2403/2-vi-cai-kia-qua-nhieu/178','PROMPT':'ducanh3/notebook7bc83c4c76/58'},
        arms={arm:dict(archive_sha256=x['record']['archive_sha256'],best=x['best'],latest=x['latest'],epochs=x['epochs'],worker_seconds=x['result']['session_seconds']) for arm,x in arms.items()},
        delta=dict(best_CIDEr_points=100*(b['result']['best_cider']-a['result']['best_cider']),endpoint_CIDEr_points=100*(b['result']['history'][-1]['cider']-a['result']['history'][-1]['cider']),endpoint_val_nll=b['epochs'][-1]['val_nll']-a['epochs'][-1]['val_nll'],endpoint_train_nll=b['epochs'][-1]['train_nll']-a['epochs'][-1]['train_nll']))
    output['source_files_sha256']=a['identity']['source'];output['common_init_sha256']=a['identity']['common_init_sha256']
    (HERE/'PROMPT_SCRATCH_XE10_RESULT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Metadata checks:',count,'PASS; deltas:',output['delta'])
    return arms,output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args();audit(a.cache)
