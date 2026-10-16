"""Audit user-run gain-guard review without running a model or ZIP instructions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from captioning.caption_gain_ranker import CaptionGainRanker, calibrate, propose


def audit(source, output):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read = lambda n: json.loads(z.read(n))
        digest = lambda n: hashlib.sha256(z.read(n)).hexdigest()
        cfg, summary = read('protocol.json'), read('run_summary.json')
        assert cfg['project_commit'] == '205d59e5381d7563b6c47067afc1acaf1fc2c71b'
        assert cfg['variant'] == 'caption_gain_guard_v2' and not cfg['test_used']
        assert summary['status'] == 'GAIN_PILOT_COMPLETE' and not summary['test_used']
        assert summary['protocol_sha256'] == digest('protocol.json')
        assert summary['fit_IDF_sha256'] == digest('locked_fit_IDF.json')
        assert summary['locked_policies_sha256'] == digest('locked_gain_policies.json')
        groups = cfg['groups']
        assert {k: len(v) for k,v in groups.items()} == {'fit':1024,'tune':128,'calibration':128,'validation':512}
        assert len({r['coco_id'] for rows in groups.values() for r in rows}) == 1792
        assert len({r['image_id'] for rows in groups.values() for r in rows}) == 1792
        assert all(r['split'] in ('train','restval') for k,rows in groups.items() if k!='validation' for r in rows)
        assert all(r['split']=='val' for r in groups['validation'])
        rows = read('feature_candidates.json')['data'];lookup={r['image_id']:r for r in rows}
        assert len(rows)==len(lookup)==1792
        assert set(lookup)=={r['image_id'] for v in groups.values() for r in v}
        policies=read('locked_gain_policies.json');assert policies==summary['locked_policies']
        validation=groups['validation'];ids=[r['image_id'] for r in validation]
        traces={};ranks={};head_checks={};selection_checks={}
        for mode in ('scores','global','regional'):
            payload=torch.load(io.BytesIO(z.read(mode+'_gain.pt')),map_location='cpu',weights_only=True)
            model=CaptionGainRanker(mode);model.load_state_dict(payload['state_dict'],strict=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert digest(mode+'_gain.pt')==summary['selections'][mode]['checkpoint_sha256']
            assert payload['protocol_sha256']==digest('protocol.json')
            history=read(mode+'_history.json');assert [r['epoch'] for r in history]==list(range(11))
            best=max(history,key=lambda r:r['tune_train_IDF_CIDEr'])
            assert best['epoch']==summary['selections'][mode]['epoch']==payload['selection']['epoch']
            for split in ('tune','calibration','validation'):
                data=read(mode+'_'+split+'_scores.json');assert [r['image_id'] for r in data]==[r['image_id'] for r in groups[split]]
                gains=torch.tensor([r['predicted_gains'] for r in data]);base=torch.tensor([r['baseline_rank'] for r in data])
                assert torch.isfinite(gains).all() and (gains[torch.arange(len(base)),base]==0).all()
                raw=propose(gains,base);guard=propose(gains,base,0. if split=='tune' else policies[mode]['correction'])
                assert raw.tolist()==[r['raw_rank'] for r in data]
                assert guard.tolist()==[r['guarded_rank'] for r in data]
                if split in ('tune','calibration'):
                    targets=read(split+'_CIDEr_targets.json');assert targets['image_ids']==[r['image_id'] for r in data]
                    truth=np.asarray(targets['scores'])
                    if split=='calibration': assert calibrate(gains.numpy(),base.numpy(),truth)==policies[mode]
                    else: assert abs(truth[np.arange(len(raw)),raw.numpy()].mean()-best['tune_train_IDF_CIDEr'])<1e-10
                else:
                    if 'fixed' in ranks: assert ranks['fixed']==base.tolist()
                    ranks['fixed']=base.tolist();ranks[mode+'_raw']=raw.tolist();ranks[mode+'_guarded']=guard.tolist()
                    traces[mode]=data
            head_checks[mode]={'strict_load':True,'finite':True,'selected_epoch':best['epoch'],'correction':policies[mode]['correction']}
        gt=read('validation_gt.json');refs={i:[] for i in ids}
        for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
        assert all(len(v)==5 for v in refs.values())
        tokenizer=PTBTokenizer();tokens=tokenizer.tokenize(refs)
        results={};per={};chair_data={}
        for name,picks in ranks.items():
            pred=read(name+'_predictions.json');assert [r['image_id'] for r in pred]==ids
            assert [r['caption'] for r in pred]==[lookup[i]['candidates'][k]['caption'] for i,k in zip(ids,picks)]
            texts=tokenizer.tokenize({r['image_id']:[{'caption':r['caption']}] for r in pred})
            cider,scores=Cider().compute_score(tokens,texts);bleu,_=Bleu(4).compute_score(tokens,texts);rouge,_=Rouge().compute_score(tokens,texts)
            saved=read(name+'_metrics.json');assert saved==summary['metrics'][name]['caption']
            for k,v in {'CIDEr':cider,'ROUGE_L':rouge,**{f'Bleu_{i+1}':b for i,b in enumerate(bleu)}}.items():assert abs(saved[k]-v)<1e-10
            chair=read(name+'_chair_output.json');sentences=chair['sentences'];cl={r['image_id']:r for r in sentences}
            assert len(sentences)==len(cl)==512 and set(cl)=={r['coco_id'] for r in validation}
            assert all(cl[r['coco_id']]['caption']==p['caption'] for r,p in zip(validation,pred))
            bad=sum(bool(r['mscoco_hallucinated_words']) for r in sentences)
            mentions=sum(len(r['mscoco_hallucinated_words']) for r in sentences)
            generated=sum(len(r['mscoco_generated_words']) for r in sentences)
            correct=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in sentences)
            total=sum(len(set(r['mscoco_gt_words'])) for r in sentences)
            agg={'CHAIRs':bad/512,'CHAIRi':mentions/generated,'Recall':correct/total}
            assert agg==chair['overall_metrics']==summary['metrics'][name]['chair']
            changed=sum(a!=b for a,b in zip(picks,ranks['fixed']));assert changed==summary['metrics'][name]['changed_vs_fixed']
            results[name]={'caption':saved,'chair':agg,'changed':changed,'hallucinated_captions':bad,'hallucinated_mentions':mentions}
            per[name]=np.asarray(scores);chair_data[name]=cl
        for name in results:
            delta=per[name]-per['fixed'];changed=np.asarray(ranks[name])!=np.asarray(ranks['fixed'])
            results[name]['CIDEr_delta_vs_fixed']=float(delta.mean())
            results[name]['changed_images_improve_decline_tie']=[int(((delta>1e-10)&changed).sum()),int(((delta< -1e-10)&changed).sum()),int(((abs(delta)<=1e-10)&changed).sum())]
            results[name]['hallucination_removed']=sum(bool(chair_data['fixed'][i]['mscoco_hallucinated_words']) and not chair_data[name][i]['mscoco_hallucinated_words'] for i in chair_data[name])
            results[name]['hallucination_introduced']=sum(bool(chair_data[name][i]['mscoco_hallucinated_words']) and not chair_data['fixed'][i]['mscoco_hallucinated_words'] for i in chair_data[name])
        changes=[]
        for mode in ('global','regional'):
            for j,r in enumerate(traces[mode]):
                if r['guarded_rank']!=r['baseline_rank']:
                    i=ids[j];cs=lookup[i]['candidates']
                    changes.append({'mode':mode,'image_id':i,'baseline':cs[r['baseline_rank']]['caption'],
                        'selected':cs[r['guarded_rank']]['caption'],'predicted_gain':r['predicted_gains'][r['guarded_rank']],
                        'validation_IDF_actual_CIDEr_gain':float(per[mode+'_guarded'][j]-per['fixed'][j])})
        primary,fixed=results['regional_guarded'],results['fixed']
        gate=(primary['caption']['CIDEr']>=fixed['caption']['CIDEr']+.005 and
              all(primary['chair'][k]<=fixed['chair'][k] for k in ('CHAIRs','CHAIRi')) and
              primary['chair']['Recall']>=fixed['chair']['Recall']-.01 and
              all(primary['caption']['CIDEr']>results[m+'_guarded']['caption']['CIDEr'] for m in ('scores','global')))
        assert summary['expansion_gate_passed']==gate
        result={'status':'AUDIT_PASS','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'source_bytes':Path(source).stat().st_size,'head_checks':head_checks,'metrics':results,'guarded_changes':changes,
                'expansion_gate_passed':gate,'feature_cache_regenerated':cfg['feature_cache_regenerated'],
                'limits':['Feature tensors absent: no independent feature hash/model forward replay.',
                          'Full train references absent: train IDF/target creation not independently rebuilt; calibration arithmetic rechecked.',
                          'METEOR cross-checked only; CHAIR reaggregated from supplied labels, not retokenized.',
                          'Repeated validation512;development result only;no new GPU run or training.']}
        Path(output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.output)
