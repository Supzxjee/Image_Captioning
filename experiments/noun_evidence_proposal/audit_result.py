"""Audit cached G/S review contents without loading a model or executing ZIP code."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from evidence_data import A_SHA,A_BASE_SHA,encoded,digest,save
from evidence_metrics import paired

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def audit(arm):
    cache=json.loads((HERE/f'review/loaded_{arm}.json').read_text(encoding='utf-8'))
    data,text=cache['json'],cache['text'];cfg=data['protocol.json'];summary=data['run_summary.json']
    assert cfg['arm']==summary['arm']==arm and cfg['updates']==2500 and cfg['seed']==42
    assert cfg['source_checkpoint_sha256']==cfg['input_sha256']['checkpoint']==A_SHA
    assert hashlib.sha256(encoded(cfg)).hexdigest()==summary['protocol_sha256']
    locked=dict(global_batch_size=8,eval_batch_size=16,head_lr=1e-4,weight_decay=.01,max_gradient_norm=1.,warmup_updates=200,
        scst_weight=.05,object_reward_weight=.5,presence_weight=.1,region_weight=.1,logit_weight=.5,correction_clip=2.,head_parameters=108752)
    assert all(cfg[k]==v for k,v in locked.items())
    locations={name:HERE/name for name in ('evidence_data.py','evidence_metrics.py','evidence_model.py','evidence_stage.py','run_evidence.py','A_secondary_reference.json')}
    locations.update({name:ROOT/'experiments/upt_alignment_pilot'/name for name in ('pilot_data.py','pilot_model.py','run_pilot.py')})
    locations['hidden_metric_smoke.py']=ROOT/'experiments/upt_hidden_pair/hidden_metric_smoke.py'
    for name,expected in cfg['implementation_sha256'].items():
        source=(json.dumps(data['source/'+name],ensure_ascii=False,separators=(',',':'))+'\n') if name.endswith('.json') else text['source/'+name]
        assert hashlib.sha256(source.encode()).hexdigest()==expected,name
        local=locations.get(name,ROOT/name)
        assert hashlib.sha256(local.read_text(encoding='utf-8').encode()).hexdigest()==expected,name
    package=json.loads((HERE/'DATA_PACKAGE.json').read_text())
    assert hashlib.sha256(encoded(data['label_receipt.json'])).hexdigest()==cfg['label_receipt_sha256']==package['receipt_sha256']
    with zipfile.ZipFile(HERE/'data/noun_evidence_train10k_v1_dataset.zip') as archive:
        groups=json.loads(archive.read('groups.json'));targets=json.loads(archive.read('targets.json'))
    assert cfg['groups']==data['groups.json']==groups
    assert hashlib.sha256(encoded(groups)).hexdigest()==cfg['cohort_sha256']
    assert hashlib.sha256(encoded(data['train_object_targets.json'])).hexdigest()==cfg['object_targets_sha256']
    train_ids={r['image_id'] for r in groups['train']};val_ids={r['image_id'] for r in groups['val']}
    assert len(train_ids)==10000 and len(val_ids)==5000 and not train_ids&val_ids
    reward_rows=data['train_object_targets.json']['data'];assert {r['image_id'] for r in reward_rows}==train_ids
    assert data['runtime_rank0.json']==data['runtime_rank1.json']
    for rank in (0,1):
        pre=data[f'preflight_rank{rank}.json'];done=data[f'worker_complete_rank{rank}.json']
        assert pre['status']=='PREFLIGHT_PASS' and pre['base_sha256']==A_BASE_SHA
        assert done['status']=='COMPLETE' and done['updates']==2500 and done['final_base_sha256']==A_BASE_SHA
        assert done['off_parity_after_training'] and done['trainable_base_parameters']==0 and done['trainable_head_parameters']==108752
    done=[data[f'worker_complete_rank{r}.json'] for r in (0,1)]
    assert all(done[0][k]==done[1][k] for k in ('initial_head_sha256','final_head_sha256'))
    history=[data[f'history_rank{r}.json'] for r in (0,1)]
    assert all(len(h)==2500 and [v['step'] for v in h]==list(range(1,2501)) for h in history)
    for epoch in (0,1):
        ids=[i for h in history for v in h if v['epoch']==epoch for i in v['local_image_ids']]
        assert Counter(ids)==Counter({i:1 for i in train_ids})
    max_loss_error=0.
    for step,records in enumerate(zip(*history),1):
        ids=[i for r in records for i in r['local_image_ids']]
        regions=sum(len(targets[str(i)]['regions']) for i in ids);observed=sum(sum(targets[str(i)]['observed']) for i in ids)
        for rank,r in enumerate(records):
            assert len(r['local_image_ids'])==4 and r['global_images_seen']==step*8
            assert r['epoch']==(step-1)//1250 and r['scst_weight']==(0. if step<=200 else .05)
            assert r['global_regions']==regions and r['global_observed_labels']==observed
            assert r['local_regions']==sum(len(targets[str(i)]['regions']) for i in r['local_image_ids'])
            assert all(np.isfinite(v) for v in r.values() if isinstance(v,(float,int)))
        assert records[0]['presence_loss_global']==records[1]['presence_loss_global']
        assert records[0]['region_loss_global']==records[1]['region_loss_global']
        # Uneven region counts intentionally produce different local losses. Compare the DDP mean.
        expected=np.mean([r['ce_loss']+r['scst_weight']*r['scst_loss'] for r in records])+.1*records[0]['presence_loss_global']+.1*records[0]['region_loss_global']
        error=abs(np.mean([r['loss'] for r in records])-expected);max_loss_error=max(max_loss_error,error)
        assert error<2e-6,error
    assert sum(r['global_regions'] for r in history[0])==11204
    assert sum(r['global_observed_labels'] for r in history[0])==1597744
    metrics={};per_cider={};chair={};predictions={}
    id_to_coco={r['image_id']:r['coco_id'] for r in groups['val']}
    for condition in ('origin','on'):
        rows=data[f'{condition}_predictions.json'];predictions[condition]={r['image_id']:r['caption'] for r in rows}
        assert len(rows)==5000 and [r['image_id'] for r in rows]==sorted(val_ids)
        assert sorted(data[f'{condition}_rank0.json']+data[f'{condition}_rank1.json'],key=lambda r:r['image_id'])==rows
        m=data[f'{condition}_metrics.json'];assert m==summary['metrics'][condition]
        assert hashlib.sha256(encoded(rows)).hexdigest()==m['predictions_sha256']
        per=data[f'{condition}_per_image_cider.json'];assert [r['image_id'] for r in per]==sorted(val_ids)
        values=np.array([r['CIDEr'] for r in per]);assert np.isfinite(values).all()
        assert abs(values.mean()-m['CIDEr'])<1e-12;per_cider[condition]=values
        output=data[f'{condition}_chair_output.json'];sentences=output['sentences'];chair[condition]={s['image_id']:s for s in sentences}
        assert len(sentences)==5000 and set(chair[condition])==set(id_to_coco.values())
        assert all(chair[condition][id_to_coco[i]]['caption']==caption for i,caption in predictions[condition].items())
        bad=sum(len(s['mscoco_hallucinated_words']) for s in sentences);mentions=sum(len(s['mscoco_generated_words']) for s in sentences)
        count=sum(bool(s['mscoco_hallucinated_words']) for s in sentences)
        recall=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences);den=sum(len(set(s['mscoco_gt_words'])) for s in sentences)
        recomputed=dict(CHAIRs=count/5000,CHAIRi=bad/mentions,Recall=recall/den)
        assert all(abs(recomputed[k]-m['chair'][k])<1e-12 and abs(recomputed[k]-output['overall_metrics'][k])<1e-12 for k in recomputed)
        assert m['counts']==dict(images=5000,hallucinated_captions=count,hallucinated_mentions=bad,object_mentions=mentions,recalled_unique_objects=recall,reference_unique_objects=den)
        assert abs(np.mean([len(r['caption'].split()) for r in rows])-m['mean_caption_words'])<1e-12
        metrics[condition]=m
    old=json.loads((HERE/'A_secondary_reference.json').read_text(encoding='utf-8'))
    mismatches=[r['image_id'] for r in old if predictions['origin'][r['image_id']]!=r['caption']]
    assert mismatches==summary['historical_A_secondary_caption_mismatches']==[]
    difference=paired(per_cider['on'],per_cider['origin']);assert difference==summary['on_minus_A']
    a=metrics['origin']['chair'];s=metrics['on']['chair']
    gate=difference['mean']>=.01 and difference['paired95'][0]>0 and s['CHAIRs']<=a['CHAIRs'] and s['CHAIRi']<=a['CHAIRi'] and s['Recall']>=a['Recall']-.005
    assert gate==summary['S_A_gate']
    changed=sum(predictions['origin'][i]!=predictions['on'][i] for i in val_ids);assert changed==summary['changed_captions']
    introduced=[i for i in chair['origin'] if not chair['origin'][i]['mscoco_hallucinated_words'] and chair['on'][i]['mscoco_hallucinated_words']]
    removed=[i for i in chair['origin'] if chair['origin'][i]['mscoco_hallucinated_words'] and not chair['on'][i]['mscoco_hallucinated_words']]
    shared=[i for i in chair['origin'] if chair['origin'][i]['mscoco_hallucinated_words'] and chair['on'][i]['mscoco_hallucinated_words']]
    trends={}
    for name,sl in (('first100',slice(0,100)),('last100',slice(-100,None))):
        records=[r for h in history for r in h[sl]]
        trends[name]={k:float(np.mean([r[k] for r in records])) for k in ('ce_loss','presence_loss_global','region_loss_global','correction_abs_mean','correction_saturation_fraction')}
    result=dict(status='AUDIT_PASS_S_FAILS_A_GATE' if arm=='S' and not gate else 'AUDIT_PASS',arm=arm,zip_sha256=cache['zip_sha256'],
        source_checkpoint_sha256=A_SHA,frozen_base_sha256=A_BASE_SHA,initial_head_tensor_sha256=done[0]['initial_head_sha256'],final_head_tensor_sha256=done[0]['final_head_sha256'],
        label_receipt_sha256=cfg['label_receipt_sha256'],protocol_sha256=summary['protocol_sha256'],stage_sha256=cfg['stage_sha256'],updates=2500,epochs=2,
        validation_images=5000,historical_A_dev_caption_mismatches=0,metrics=metrics,on_minus_A=difference,S_A_gate=gate,
        changed_captions=changed,hallucination_transitions=dict(introduced=len(introduced),removed=len(removed),shared=len(shared)),
        training_trends=trends,max_ddp_mean_loss_reconstruction_error=max_loss_error,
        training_seconds=max(h[-1]['seconds'] for h in history),preparation_seconds=cfg['preparation_seconds'],test_used=False,
        checks=['source bytes and local implementation','pinned protocol/label/reward hashes','exact A tensor digest reported by both ranks','runtime/rank state equality',
            'two epochs actual image coverage','every step region/observed counts','DDP mean loss reconstruction','5000 ordered IDs','CIDEr mean and paired bootstrap',
            'CHAIR numerator/denominator reaggregation','historical A1000 captions identical'],
        limits=['One seed; historically reused validation. CI measures image uncertainty, not training-seed variation.',
            'No checkpoint/head bytes included; reported state hashes cannot be independently recomputed here.',
            'No new visual judgment: hallucination transitions are CHAIR labels, not human-verified image correctness.',
            'G is not supplied; no spatial-supervision attribution. Controls correctly skipped after failed S/A gate.'])
    save(HERE/f'{arm}_RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['G','S']);audit(p.parse_args().arm)
