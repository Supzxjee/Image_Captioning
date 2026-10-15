"""Audit returned pilot without running notebooks, training, or model inference."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from audit_caption_progress import chair_aggregate
from captioning.learned_caption_ranker import CaptionRanker,MODES
from captioning.multiscore_reranking import rerank_multiscore
from captioning.object_consistency import index_detections


def digest(b):return hashlib.sha256(b).hexdigest()
def bootstrap(delta):
    values,counts=np.unique(delta,return_counts=True)
    samples=np.random.default_rng(42).multinomial(len(delta),counts/len(delta),size=20000)@values/len(delta)
    return np.quantile(samples,[.025,.975]).tolist()


def run(a):
    torch.set_num_threads(2);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        raw={n:z.read(n) for n in z.namelist()}
    read=lambda n:json.loads(raw[n])
    s=read('run_summary.json');cfg=read('protocol.json');groups=cfg['groups']
    assert s['status']=='PILOT_COMPLETE' and not s['test_used'] and not cfg['test_used']
    assert (s['fit_images'],s['tune_images'],s['validation_images'])==(1024,256,512)
    assert s['protocol_sha256']==digest(raw['protocol.json'])
    assert {k:len(v) for k,v in groups.items()}=={'fit':1024,'tune':256,'validation':512}
    assert cfg['images']==sum(groups.values(),[])
    assert all(r['split'] in ('train','restval') for k in ('fit','tune') for r in groups[k])
    assert all(r['split']=='val' for r in groups['validation'])
    assert len({r['coco_id'] for r in cfg['images']})==1792
    with zipfile.ZipFile(a.test_archive) as z:
        test_ids={r['coco_id'] for r in json.loads(z.read('protocol.json'))['images']}
    assert not test_ids&{r['coco_id'] for r in cfg['images']}
    identity=json.loads((Path(__file__).resolve().parent/'audit/hoi_validation_input_identity.json').read_text())
    assert all(v==identity['input_sha256'][k] for k,v in cfg['input_sha256'].items())
    beams=read('beam_candidates.json')['data'];features=read('feature_candidates.json')['data']
    assert len(beams)==len(features)==1792
    for phase,records in [('beams',beams),('features',features)]:
        worker=read(phase+'_worker_config.json');assert worker==cfg
        for rank in range(2):
            journal=[json.loads(line) for line in raw[f'{phase}_{rank}.jsonl'].splitlines()]
            assert journal==records[rank::2] and len(journal)==896
            assert read(f'{phase}_{rank}.protocol.json')=={'phase':phase,'rank':rank,'protocol':cfg}
    for original,r,image in zip(beams,features,cfg['images']):
        assert all(original[k]==r[k]==image[k] for k in ('image_id','coco_id','filename'))
        assert original['candidates']==[{k:v for k,v in c.items() if k!='clipscore'} for c in r['candidates']]
    checkpoints={}
    for mode in MODES:
        b=raw[mode+'_ranker.pt'];assert digest(b)==s['selections'][mode]['checkpoint_sha256']
        checkpoint=torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
        assert checkpoint['protocol_sha256']==s['protocol_sha256'] and checkpoint['mode']==mode
        model=CaptionRanker(mode);model.load_state_dict(checkpoint['state_dict'],strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert sum(p.numel() for p in model.parameters())==s['selections'][mode]['parameters']
        history=read(mode+'_history.json');assert [r['epoch'] for r in history]==list(range(11))
        best=max(history,key=lambda r:(r['tune_CIDEr'],-r['epoch']))
        assert best['epoch']==checkpoint['selection']['epoch']==s['selections'][mode]['selected_epoch']
        assert best['tune_CIDEr']==checkpoint['selection']['tune_CIDEr']==s['selections'][mode]['tune_CIDEr']
        assert all(np.isfinite(r['tune_CIDEr']) and (r['epoch']==0 or np.isfinite(r['training_loss'])) for r in history)
        zero=not torch.count_nonzero(model.head[-1].weight) and not torch.count_nonzero(model.head[-1].bias)
        assert zero if mode=='scores' else not zero
        checkpoints[mode]={'hash_and_strict_load':True,'selected_epoch':best['epoch'],'zero_residual_head':bool(zero)}
        (out/(mode+'_ranker.pt')).write_bytes(b)
    lookup={r['image_id']:r for r in features};val=[lookup[r['image_id']] for r in groups['validation']]
    assert read('validation_images.json')==groups['validation']
    gt=read('validation_gt.json');ids=[r['image_id'] for r in val]
    refs={i:[] for i in ids}
    for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
    assert len(gt['images'])==512 and all(len(v)==5 for v in refs.values())
    with zipfile.ZipFile(a.validation_archive) as z:
        original=json.loads(z.read('val_gt.json'));old={i:[] for i in ids}
        for r in original['annotations']:
            if r['image_id'] in old:old[r['image_id']].append({'caption':r['caption']})
    assert old==refs
    predictions={name:read(name+'_predictions.json') for name in ('fixed',*MODES)}
    assert predictions['scores']==predictions['fixed']
    choices={}
    for name,pred in predictions.items():
        assert len(pred)==512 and [r['image_id'] for r in pred]==ids
        choices[name]=[next(k for k,c in enumerate(r['candidates']) if c['caption']==p['caption']) for r,p in zip(val,pred)]
        assert sum(i!=j for i,j in zip(choices[name],choices['fixed']))==s['metrics'][name]['changed_vs_fixed']
    detection_bytes=Path(a.detections).read_bytes();assert digest(detection_bytes)==cfg['input_sha256']['detections']
    detected=index_detections(json.loads(detection_bytes))
    assert [rerank_multiscore(r,detected[r['filename']],.4,.3)[0]['original_rank'] for r in val]==choices['fixed']
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    token_refs=PTBTokenizer().tokenize(refs)
    tokens=PTBTokenizer().tokenize({r['image_id']:[{'caption':c['caption']} for c in r['candidates']] for r in val})
    scores={};chairs={};counts={};metrics={}
    for name,choice in choices.items():
        text={i:[tokens[i][k]] for i,k in zip(ids,choice)}
        cider,per_image=Cider().compute_score(token_refs,text);bleu,_=Bleu(4).compute_score(token_refs,text);rouge,_=Rouge().compute_score(token_refs,text)
        measured={'CIDEr':float(cider),'ROUGE_L':float(rouge),**{f'Bleu_{i+1}':float(v) for i,v in enumerate(bleu)}}
        assert all(abs(v-s['metrics'][name]['caption'][k])<1e-10 for k,v in measured.items())
        assert read(name+'_metrics.json')==s['metrics'][name]['caption']
        aggregate,counts[name]=chair_aggregate(read(name+'_chair_output.json'))
        assert aggregate==s['metrics'][name]['chair']
        cr={r['image_id']:r for r in read(name+'_chair_output.json')['sentences']};chairs[name]=cr
        ci=read(name+'_chair_input.json');assert len(ci)==len(cr)==512
        assert all(cr[r['coco_id']]['caption']==p['caption']==c['caption'] and c['image_id']==r['coco_id'] for r,p,c in zip(val,predictions[name],ci))
        scores[name]=np.asarray(per_image);metrics[name]=measured
        print(name,'caption metric replay PASS',flush=True)
    comparisons={}
    for name,base in [('global','fixed'),('regional','fixed'),('regional','global')]:
        delta=scores[name]-scores[base]
        cs=np.asarray([chairs[name][r['coco_id']]['metrics']['CHAIRs']-chairs[base][r['coco_id']]['metrics']['CHAIRs'] for r in val])
        assert all(set(chairs[name][r['coco_id']]['mscoco_gt_words'])==set(chairs[base][r['coco_id']]['mscoco_gt_words']) for r in val)
        comparisons[name+'_minus_'+base]={'CIDEr_delta':float(delta.mean()),'CIDEr_95_percentile_interval':bootstrap(delta),
            'CHAIRs_delta':float(cs.mean()),'CHAIRs_95_percentile_interval':bootstrap(cs),
            'hallucination_removed':sum(chairs[base][r['coco_id']]['metrics']['CHAIRs']==1 and chairs[name][r['coco_id']]['metrics']['CHAIRs']==0 for r in val),
            'hallucination_introduced':sum(chairs[base][r['coco_id']]['metrics']['CHAIRs']==0 and chairs[name][r['coco_id']]['metrics']['CHAIRs']==1 for r in val)}
    m=s['metrics'];regional=m['regional'];fixed=m['fixed']
    gate=(regional['caption']['CIDEr']>=fixed['caption']['CIDEr']+.005 and regional['chair']['CHAIRs']<=fixed['chair']['CHAIRs']
        and regional['chair']['CHAIRi']<=fixed['chair']['CHAIRi'] and regional['chair']['Recall']>=fixed['chair']['Recall']-.01
        and all(regional['caption']['CIDEr']>m[k]['caption']['CIDEr'] for k in ('scores','global')))
    assert gate==s['expansion_gate_passed']==False
    result={'source_archive_sha256':digest(Path(a.archive).read_bytes()),'summary':s,
        'checks':{'cohort_counts_disjoint_no_test':True,'original_input_hashes':True,'two_phase_shards_and_candidate_integrity':True,
                  'frozen_epoch_selection_matches_tune_histories':True,'fixed_reranking_replayed':True,'scores_epoch0_exact_fixed_predictions':True,
                  'validation_GT_matches_prior_bundle':True,'CIDEr_BLEU_ROUGE_recomputed':True,'CHAIR_reaggregated_and_captions_checked':True,
                  'gate_recomputed':True,'feature_forward':'Unavailable:large private features absent;no independent global/regional forward.',
                  'METEOR':'Stored file/summary cross-check only;not independently recomputed.'},
        'checkpoint_checks':checkpoints,'recomputed_metrics':metrics,'chair_counts':counts,'paired_comparisons':comparisons,
        'bootstrap':{'replicates':20000,'seed':42,'fixed_corpus_IDF':True,'scope':'One-seed reused validation512,not independent test or across-seed evidence.'},
        'new_notebook_started':False,'new_training_performed':False}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('archive','test-archive','validation-archive','detections','output'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
