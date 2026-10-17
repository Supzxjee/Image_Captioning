"""Artifact-only audit of manually run full-train SCST experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from run_scst_full_train import select_checkpoint


def audit(source,output):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        digest=lambda n:hashlib.sha256(z.read(n)).hexdigest()
        cfg,summary,locked=read('protocol.json'),read('run_summary.json'),read('locked_selection.json')
        assert cfg['code_commit']=='025c5597568800d3ef13ad0d62080a1541efebcd'
        assert cfg['variant']=='scst_ce_count_w05_full_train_v1' and not cfg['test_used']
        assert summary['status']=='FULL_TRAIN_COMPARISON_COMPLETE' and not summary['test_used']
        assert summary['protocol_sha256']==digest('protocol.json')
        assert summary['locked_selection']==locked and locked['tune_metrics_sha256']==digest('tune1000/metrics.json')
        identity=json.loads((Path(__file__).parent/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
        assert all(identity[k]==v for k,v in cfg['input_sha256'].items())
        groups=cfg['groups']
        assert {k:len(v) for k,v in groups.items()}=={'train':113287,'val':1000,'validation_remainder':4000}
        assert len({r['coco_id'] for rows in groups.values() for r in rows})==118287
        assert len({r['image_id'] for rows in groups.values() for r in rows})==118287
        assert cfg['snapshot_steps']==[7081] and cfg['reward_reference_cache_size']==256
        assert cfg['strict_reproducibility'] and cfg['train_same_gpu']
        assert (cfg['scst_weight'],cfg['ce_weight'],cfg['object_reward_weight'])==(.05,1.,.5)
        assert (cfg['object_reward_parser'],cfg['object_reward_penalty'])==('aliases_v2','unique_count_v2')
        targets=read('train_object_targets.json')
        assert digest('train_object_targets.json')==cfg['train_object_targets_sha256']
        assert targets['metadata']==cfg['object_reward_metadata']
        assert [{k:r[k] for k in ('image_id','coco_id','filename','filepath')} for r in targets['data']]==groups['train']
        from captioning.object_reward import TrainingObjectReward,CACHE_SHA256
        TrainingObjectReward(targets['data'],'aliases_v2','unique_count_v2')
        assert targets['metadata']['annotation_cache_sha256']==CACHE_SHA256
        assert not targets['metadata']['heldout_reward_labels_exported']
        assert sum(bool(r['supported']) for r in targets['data'])==targets['metadata']['reward_eligible_images']
        training={}
        for arm in ('scst_ce','scst_ce_object'):
            hist=read(arm+'/history.json');trained=read(arm+'/training_summary.json');runtime=read(arm+'/runtime.json')
            assert len(hist)==trained['steps']==14161 and trained['train_images']==113287
            assert trained['protocol_sha256']==digest('protocol.json')
            assert trained['source_checkpoint_sha256']==identity['checkpoint']
            assert runtime['deterministic_algorithms'] and not runtime['warn_only']
            assert runtime['cublas_workspace']==':4096:8' and runtime['sdp_math']
            weight=.5 if arm=='scst_ce_object' else 0.
            assert trained['object_reward_weight']==weight and trained['scst_weight']==.05 and trained['ce_weight']==1.
            for i,r in enumerate(hist,1):
                assert r['step']==i and r['images_seen']==min(i*8,113287)
                assert all(math.isfinite(v) for v in r.values() if isinstance(v,(float,int)))
                assert abs(r['loss']-(r['ce_loss']+.05*r['scst_loss']))<1e-5
                for p in ('sample','greedy'):
                    assert abs(r[p+'_reward']-r[p+'_cider_reward']-weight*r.get(p+'_object_reward',0.))<1e-6
                    if weight:
                        assert abs(r[p+'_object_reward']-(.5*r[p+'_salient_recall']-r[p+'_unsupported_count']))<1e-6
                assert abs(r['advantage_mean']-r['sample_reward']+r['greedy_reward'])<1e-6
            training[arm]=trained
        assert read('scst_ce/runtime.json')==read('scst_ce_object/runtime.json')
        assert read('scst_ce/initial_model_identity.json')==read('scst_ce_object/initial_model_identity.json')
        metrics={};per={};chairs={};captions={}
        for phase,group,report in [('tune1000','val','tune_metrics'),('validation_remainder4000','validation_remainder','remainder_metrics')]:
            phasecfg=read(phase+'/evaluation_protocol.json');savedmetrics=read(phase+'/metrics.json')
            assert savedmetrics==summary[report]
            images=groups[group];n=len(images);ids=[r['image_id'] for r in images]
            assert phasecfg['groups']['val']==images
            if group=='validation_remainder':
                assert phasecfg['evaluation_checkpoints']==locked['checkpoint_paths']
            refs={i:[] for i in ids}
            for r in read(phase+'/val_gt.json')['annotations']:refs[r['image_id']].append({'caption':r['caption']})
            assert all(len(v)==5 for v in refs.values())
            tokenizer=PTBTokenizer();truth=tokenizer.tokenize(refs)
            metrics[phase]={}
            for arm,path in phasecfg['evaluation_checkpoints'].items():
                prefix=phase+'/'+arm+'/'
                beams=read(prefix+'beams.json')['data'];scored=read(prefix+'clip_beams.json')['data']
                assert [r['image_id'] for r in beams]==ids==[r['image_id'] for r in scored]
                for a,b in zip(beams,scored):
                    assert a['candidates']==[{k:v for k,v in c.items() if k!='clipscore'} for c in b['candidates']]
                worker=read(prefix+'beams_worker_config.json')
                assert worker['checkpoint']==path
                if group=='validation_remainder':assert worker['checkpoint_sha256']==locked['checkpoint_sha256'][arm]
                for rank in range(2):
                    assert [json.loads(line) for line in z.read(prefix+f'beams_{rank}.jsonl').decode().splitlines()]==beams[rank::2]
                for policy in ('rank0','fixed'):
                    name=arm+'_'+policy;pred=read(prefix+policy+'_predictions.json')
                    assert [r['image_id'] for r in pred]==ids
                    assert all(p['caption'] in {c['caption'] for c in b['candidates']} for p,b in zip(pred,beams))
                    if policy=='rank0':assert all(p['caption']==b['candidates'][0]['caption'] for p,b in zip(pred,beams))
                    tokenpred=tokenizer.tokenize({r['image_id']:[{'caption':r['caption']}] for r in pred})
                    cider,scores=Cider().compute_score(truth,tokenpred)
                    bleu,_=Bleu(4).compute_score(truth,tokenpred);rouge,_=Rouge().compute_score(truth,tokenpred)
                    saved=read(prefix+policy+'_metrics.json');assert saved==savedmetrics[name]['caption']
                    for k,v in {'CIDEr':cider,'ROUGE_L':rouge,**{f'Bleu_{i+1}':v for i,v in enumerate(bleu)}}.items():assert abs(saved[k]-v)<1e-10
                    ch=read(prefix+policy+'_chair_output.json');rows=ch['sentences'];byid={r['image_id']:r for r in rows}
                    assert len(rows)==len(byid)==n and set(byid)=={r['coco_id'] for r in images}
                    assert all(byid[i['coco_id']]['caption']==p['caption'] for i,p in zip(images,pred))
                    bad=sum(bool(r['mscoco_hallucinated_words']) for r in rows);hall=sum(len(r['mscoco_hallucinated_words']) for r in rows)
                    mentions=sum(len(r['mscoco_generated_words']) for r in rows)
                    correct=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in rows)
                    total=sum(len(set(r['mscoco_gt_words'])) for r in rows)
                    agg={'CHAIRs':bad/n,'CHAIRi':hall/mentions,'Recall':correct/total}
                    assert agg==ch['overall_metrics']==savedmetrics[name]['chair']
                    metrics[phase][name]={'caption':saved,'chair':agg,'hallucinated_captions':bad,'hallucinated_mentions':hall,'object_mentions':mentions,'correct_unique_objects':correct,'reference_unique_objects':total}
                    per[(phase,name)]=np.asarray(scores);chairs[(phase,name)]=byid;captions[(phase,name)]=pred
                    print('Audited',phase,name,flush=True)
        for arm in ('scst_ce','scst_ce_object'):
            name,eligible=select_checkpoint(summary['tune_metrics'],[arm+'_mid',arm+'_end'])
            assert name==locked['selected_names'][arm] and eligible==locked['eligible_names'][arm]
        assert locked['checkpoint_sha256']['scst_ce_object']==training['scst_ce_object']['checkpoint_sha256']
        assert locked['selected_names']['scst_ce']=='original'
        phase='validation_remainder4000';paired={}
        for policy in ('rank0','fixed'):
            left,right='original_'+policy,'scst_ce_object_'+policy
            assert captions[(phase,left)]==captions[(phase,'scst_ce_'+policy)]
            delta=per[(phase,right)]-per[(phase,left)];rng=np.random.default_rng(20261017)
            ci=np.percentile([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(5000)],[2.5,97.5])
            a,b=chairs[(phase,left)],chairs[(phase,right)]
            paired[policy]={'CIDEr_delta':float(delta.mean()),'conditional_bootstrap95':ci.tolist(),
                'hallucination_removed':sum(bool(a[i]['mscoco_hallucinated_words']) and not b[i]['mscoco_hallucinated_words'] for i in a),
                'hallucination_introduced':sum(bool(b[i]['mscoco_hallucinated_words']) and not a[i]['mscoco_hallucinated_words'] for i in a),
                'changed_captions':sum(a['caption']!=b['caption'] for a,b in zip(captions[(phase,left)],captions[(phase,right)]))}
        treatment,baseline=(metrics[phase][k+'_fixed'] for k in ('scst_ce_object','original'))
        gate=treatment['caption']['CIDEr']>baseline['caption']['CIDEr'] and all(treatment['chair'][k]<=baseline['chair'][k] for k in ('CHAIRs','CHAIRi')) and treatment['chair']['Recall']>=baseline['chair']['Recall']-.01
        assert gate==summary['fixed_beats_both_with_hallucination_control']
        result={'status':'AUDIT_PASS','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'source_bytes':Path(source).stat().st_size,'metrics':metrics,'training':training,'locked_selection':locked,
                'remainder_paired_comparisons':paired,'gate_passed':bool(gate),
                'target_metadata':targets['metadata'],'test_used':False,
                'limits':['One seed; reused development validation, not never-seen test.',
                         'SCST+CE selected original fallback: no comparison with independently selected newly trained control on remainder.',
                         'No trained tensors/source dataset/detections in review: cannot replay weights, split provenance or fixed-rerank formula.',
                         'Checkpoint selection arithmetic and evaluator config consistency verified; physical execution chronology not independently proven.',
                         'METEOR cross-checked only;CHAIR reaggregated,not retokenized.',
                         'Bootstrap fixed-IDF paired images conditional on checkpoint/pool;not seed uncertainty.']}
        Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(paired,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.output)
