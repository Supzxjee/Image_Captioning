"""Verify a returned independent-test bundle and recompute caption/CHAIR evidence.

No notebook execution or model inference. Bootstrap intervals use paired test images with corpus IDF held fixed.
No hyperparameters are selected using test references.
"""
import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from audit_caption_progress import chair_aggregate
from captioning.hoi_caption_reranking import METHOD,extract_claims,rerank_with_hoi
from captioning.object_consistency import index_detections


def digest(data): return hashlib.sha256(data).hexdigest()


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as archive:
        assert archive.testzip() is None
        assert len(set(archive.namelist()))==len(archive.namelist())
        raw={name:archive.read(name) for name in archive.namelist()}
    def read(name): return json.loads(raw[name])
    summary=read('run_summary.json');cfg=read('protocol.json');locked=read('applied_validation_selection.json')
    assert summary['status']=='TEST_COMPLETE' and summary['count']==5000
    assert cfg['split']==summary['split']=='test' and locked['split']=='val'
    assert cfg['test_run'] and summary['test_run'] and not cfg['training_performed']
    assert summary['protocol_sha256']==digest(raw['protocol.json'])
    from run_hoi_caption_validation import load_test_selection
    lock_path=out/'applied_validation_selection.json'
    lock_path.write_bytes(raw['applied_validation_selection.json'])
    assert load_test_selection(lock_path)==locked
    assert cfg['locked_selection_sha256']==digest(raw['applied_validation_selection.json'])
    assert cfg['locked_upt_weight']==.1 and cfg['upt_grid'] is None
    assert summary['selection']=={'upt_weight':.1,'selected_on':'validation','retuned_on_test':False}
    assert summary['trials']==[] and not summary['retuned_on_test']
    identity=json.loads((Path(__file__).resolve().parent/'audit/hoi_validation_input_identity.json').read_text())
    assert cfg['input_sha256']==identity['input_sha256']
    assert cfg['clip_revision']==identity['clip_revision'] and cfg['method']==identity['method']
    with zipfile.ZipFile(a.validation_archive) as validation:
        assert digest(validation.read('val_hoi_candidates.json'))==locked['candidate_sha256']
        val_ids={r['coco_id'] for r in json.loads(validation.read('protocol.json'))['images']}
    assert not val_ids.intersection(r['coco_id'] for r in cfg['images'])
    assert cfg['method']==METHOD and locked['grid_fixed_before_run']==[0.,.05,.1,.2]
    assert locked['fixed_base']=={'decoder':.3,'clip':.4,'occ':.3}
    base=read('test_beam5_clipscore_candidates.json');bundle=read('test_hoi_candidates.json')
    rows=bundle['data'];images=cfg['images'];ids=[r['image_id'] for r in rows]
    assert len(ids)==len(set(ids))==5000
    assert [r['image_id'] for r in images]==ids
    assert bundle['metadata']['test_protocol_sha256']==summary['protocol_sha256']
    assert base['metadata']['checkpoint_sha256']==cfg['input_sha256']['checkpoint']
    assert base['metadata']['clipscore_revision']==cfg['clip_revision']
    for phase,expected in [('beams',read('test_beam5_candidates.json')['data']),('upt',rows)]:
        worker_cfg=read(phase+'_worker_config.json')
        for rank in range(2):
            journal=[json.loads(line) for line in raw[f'{phase}_{rank}.jsonl'].splitlines()]
            stamp=read(f'{phase}_{rank}.protocol.json')
            assert stamp=={'phase':phase,'rank':rank,'protocol':worker_cfg}
            assert journal==expected[rank::2] and len(journal)==2500
    beams=read('test_beam5_candidates.json')['data']
    assert len(beams)==len(base['data'])==5000
    for beam,scored in zip(beams,base['data']):
        assert beam['image_id']==scored['image_id']
        assert beam['candidates']==[{k:v for k,v in c.items() if k!='clipscore'} for c in scored['candidates']]
    features=Counter();claim_count=unknown=0;feature_variation=[]
    for image,old,row in zip(images,base['data'],rows):
        assert all(image[k]==row[k]==old[k] for k in ('image_id','coco_id','filename'))
        assert [{k:v for k,v in c.items() if k!='hoi_evidence'} for c in row['candidates']]==old['candidates']
        for c in row['candidates']:
            evidence=c['hoi_evidence'];claims=evidence['claims'];assert evidence['method']==METHOD
            assert [{k:r[k] for k in ('subject','verb','object')} for r in claims]==extract_claims(c['caption'])
            values=[]
            for claim in claims:
                assert claim['status'] in ('available','unknown');claim_count+=1
                if claim['status']=='unknown':unknown+=1;continue
                assert claim['subject_index']!=claim['object_index']
                assert abs(1/(1+math.exp(-claim['raw_logit']))-claim['probability'])<1e-6
                expected=0. if claim['competitor_object'] is None else math.tanh((claim['raw_logit']-claim['competitor_logit'])/2)
                if claim['competitor_object'] is not None:assert claim['competitor_object']!=claim['object']
                assert abs(expected-claim['contrast'])<1e-12
                values.append(claim['contrast']);features[claim['verb']]+=1
            assert len(values)==evidence['available_claims']
            expected=sum(values)/len(claims) if claims else 0.
            assert abs(expected-evidence['feature'])<1e-12
        if row['hoi_pair_count']==0:
            assert row['hoi_image_status']=='no_human_object_pairs'
            assert all(c['hoi_evidence']['available_claims']==0 and c['hoi_evidence']['feature']==0 for c in row['candidates'])
        fs=[c['hoi_evidence']['feature'] for c in row['candidates']]
        if max(fs)-min(fs)>1e-12:feature_variation.append(row['image_id'])
    coverage={'images_without_pairs':sum(r['hoi_pair_count']==0 for r in rows),
        'images_with_available_parsed_claim':sum(any(c['hoi_evidence']['available_claims'] for c in r['candidates']) for r in rows),
        'images_with_nonzero_relation_feature':sum(any(c['hoi_evidence']['feature']!=0 for c in r['candidates']) for r in rows)}
    assert coverage==summary['upt_coverage']
    predictions={name:{r['image_id']:r['caption'] for r in read(name+'_predictions.json')} for name in ('rank0','fixed_rerank','upt_rerank')}
    ranks={name:[] for name in predictions}
    for row in rows:
        for name in predictions:
            matching=[i for i,c in enumerate(row['candidates']) if c['caption']==predictions[name][row['image_id']]]
            assert len(matching)==1;ranks[name].append(matching[0])
        assert ranks['rank0'][-1]==0
    # Detection cache is an independent input, hash-bound to the run protocol.
    path=Path(a.detections)
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('objectdetectioncache.json')];assert len(names)==1
            detection_bytes=z.read(names[0])
    else:detection_bytes=path.read_bytes()
    assert digest(detection_bytes)==cfg['input_sha256']['detections']
    detections=index_detections(json.loads(detection_bytes))
    grid_ranks={weight:[] for weight in [0.,locked['upt_weight']]}
    for row in rows:
        selected,enriched=rerank_with_hoi(row,detections[row['filename']],locked['upt_weight'])
        assert selected['caption']==predictions['upt_rerank'][row['image_id']]
        for weight in grid_ranks:
            best=max(enriched,key=lambda c:(c['fixed_rerank_score']+weight*c['hoi_evidence']['feature'],-c['original_rank']))
            grid_ranks[weight].append(best['original_rank'])
    assert grid_ranks[0.]==ranks['fixed_rerank']
    assert grid_ranks[.1]==ranks['upt_rerank']
    gt=read('test_gt.json');refs={i:[] for i in ids}
    assert {r['id'] for r in gt['images']}==set(ids)
    for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
    assert all(len(v)==5 for v in refs.values())
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    token_refs=PTBTokenizer().tokenize(refs)
    token_beams=PTBTokenizer().tokenize({r['image_id']:[{'caption':c['caption']} for c in r['candidates']] for r in rows})
    remeasured={};per_image={};trial=[]
    chair={};counts={};chair_records={}
    for name,choice in ranks.items():
        text={i:[token_beams[i][rank]] for i,rank in zip(ids,choice)}
        cider,scores=Cider().compute_score(token_refs,text);bleu,_=Bleu(4).compute_score(token_refs,text);rouge,_=Rouge().compute_score(token_refs,text)
        measured={'CIDEr':float(cider),'ROUGE_L':float(rouge),**{f'Bleu_{k+1}':float(v) for k,v in enumerate(bleu)}}
        assert all(abs(value-summary['metrics'][name]['caption'][key])<1e-10 for key,value in measured.items())
        assert read(name+'_metrics.json')==summary['metrics'][name]['caption']
        remeasured[name]=measured;per_image[name]=np.asarray(scores)
        chair[name],counts[name]=chair_aggregate(read(name+'_chair_output.json'))
        assert chair[name]==summary['metrics'][name]['chair']
        cr={r['image_id']:r for r in read(name+'_chair_output.json')['sentences']};chair_records[name]=cr
        chair_input={r['image_id']:r['caption'] for r in read(name+'_chair_input.json')}
        assert len(cr)==len(chair_input)==5000
        for row in rows:assert cr[row['coco_id']]['caption']==chair_input[row['coco_id']]==predictions[name][row['image_id']]
        chosen=[r['candidates'][rank]['caption'] for r,rank in zip(rows,choice)]
        assert abs(sum(len(c.split()) for c in chosen)/5000-summary['metrics'][name]['mean_words'])<1e-12
        assert sum(bool(extract_claims(c)) for c in chosen)==summary['metrics'][name]['captions_with_parsed_hoi']
    changes=[]
    for j,row in enumerate(rows):
        coco=row['coco_id'];truth=chair_records['rank0'][coco]['mscoco_gt_words']
        assert all(set(chair_records[name][coco]['mscoco_gt_words'])==set(truth) for name in predictions)
        if ranks['fixed_rerank'][j]!=ranks['upt_rerank'][j]:
            changes.append({'image_id':row['image_id'],'coco_id':coco,'filename':row['filename'],
                'fixed_caption':predictions['fixed_rerank'][row['image_id']],'upt_caption':predictions['upt_rerank'][row['image_id']],
                'fixed_rank':ranks['fixed_rerank'][j],'upt_rank':ranks['upt_rerank'][j],
                'cider_delta':float(per_image['upt_rerank'][j]-per_image['fixed_rerank'][j]),
                'fixed_chairs':chair_records['fixed_rerank'][coco]['metrics']['CHAIRs'],
                'upt_chairs':chair_records['upt_rerank'][coco]['metrics']['CHAIRs']})
    for baseline in ('rank0','fixed_rerank'):
        left=chair_records[baseline];right=chair_records['upt_rerank']
        assert summary['transitions_upt_vs_baseline'][baseline]=={
            'hallucination_removed':sum(left[i]['metrics']['CHAIRs']==1 and right[i]['metrics']['CHAIRs']==0 for i in left),
            'hallucination_introduced':sum(left[i]['metrics']['CHAIRs']==0 and right[i]['metrics']['CHAIRs']==1 for i in left),
            'caption_changed':sum(left[i]['caption']!=right[i]['caption'] for i in left)}
    delta=per_image['upt_rerank']-per_image['fixed_rerank'];unique,freq=np.unique(delta,return_counts=True)
    rng=np.random.default_rng(42);samples=rng.multinomial(5000,freq/5000,size=20000)@unique/5000
    interval=np.quantile(samples,[.025,.975]).tolist()
    result={'archive_sha256':digest(Path(a.archive).read_bytes()),'archive_member_sha256':{n:digest(b) for n,b in raw.items()},
        'summary':summary,'checks':{'all5000_ids_and_two_shards':True,'candidate_pool_and_scores_unchanged':True,
        'claim_parser_and_feature_formula_replayed':True,'two_frozen_choices_replayed':True,'validation_decision_hash_unchanged':True,'test_validation_ids_disjoint':True,'no_test_grid_selection':True,
        'three_arm_CIDEr_BLEU_ROUGE_recomputed':True,'CHAIR_reaggregated_same_ids_objects_captions':True,
        'meteor':'Stored metric cross-checked against summary; not independently rerun.',
        'UPT_model':'No5000-image model rerun; raw teacher tensors absent from review ZIP.'},
        'recomputed_metrics':remeasured,'chair_counts':counts,'coverage':{**coverage,'images_with_candidate_feature_variation':len(feature_variation)},
        'parsed_claims':claim_count,'unknown_claims':unknown,'available_claims_by_verb':dict(features),
        'changed_vs_fixed':len(changes),'changed_cases_cider':{'better':sum(r['cider_delta']>1e-12 for r in changes),
        'worse':sum(r['cider_delta']< -1e-12 for r in changes),'tied':sum(abs(r['cider_delta'])<=1e-12 for r in changes)},
        'cider_delta_upt_minus_fixed':float(delta.mean()),'paired_image_bootstrap':{'replicates':20000,'seed':42,'fixed_corpus_IDF':True,
        'delta_95_percentile_interval':interval,'selection_adjusted':False,'interpretation':'Paired held-out test bootstrap; fixed corpus IDF; conditional on this checkpoint, candidate pool and frozen validation decision. Not evidence across training seeds.'},
        'manual_notebook_execution_only':True,'new_notebook_started':False}
    (out/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'changed_captions_33.json').write_text(json.dumps(sorted(changes,key=lambda r:r['cider_delta'],reverse=True),indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('archive_member_sha256','summary')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--validation-archive',required=True);p.add_argument('--detections',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
