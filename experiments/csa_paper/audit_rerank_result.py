"""Stored reranking text/count/selector audit and NumPy bootstrap; no models/scorers."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from candidate_data import baseline
from rerank_data import candidates,select,MODELS,POLICIES,INPUT_SHA,DATA_SHA,DETECTION_SHA
from real_bank_data import read,save,sha
from saved_quality_data import chair_counts,check_rows

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def points(total,n):
    return np.stack([total[...,0]/n,total[...,1]/n,total[...,2]/np.maximum(1,total[...,3]),
        total[...,4]/np.maximum(1,total[...,5])],axis=-1)


def run():
    folder=HERE/'rerank163_review_text';checks=0
    def check(value):
        nonlocal checks
        if not value:raise ValueError('Rerank audit failed at check'+str(checks+1))
        checks+=1
    result=read(folder/'result.json');source=read(folder/'source_receipt.json');saved=baseline()
    check(read(folder/'saved_text_receipt.json')['crc_pass'])
    check(result['status']=='CSA_FIXED_RERANK_VALID_COMPLETE' and result['images']==5000)
    check(result['actual_reranking'] and not any(result[k] for k in ('test_used','training','oracle_used_for_selection')))
    check(read(folder/'run_status.json')==dict(complete=True,training=False,test_used=False,oracle_used_for_selection=False,actual_reranking=True))
    check(result['candidate_source']=='ducanh2403/2-vi-cai-kia-qua-nhieu/163' and result['models']==MODELS)
    for n,h in source['sha256'].items():check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
    for n in ('experiments/csa_paper/rerank_data.py','captioning/multiscore_reranking.py','captioning/object_consistency.py'):
        check(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==source['sha256'][n])
    pool=candidates(folder/'candidates163');check(len(pool)==5000)
    prepared=read(folder/'prepared.json');inputs=read(folder/'inference_inputs.json')
    check(prepared['status']=='RERANK_VALID_PREPARED' and prepared['images']==5000 and not prepared['test_used'])
    check(prepared['candidate_files_sha256']==INPUT_SHA and prepared['dataset_sha256']==DATA_SHA and prepared['detections_sha256']==DETECTION_SHA)
    check(prepared['detector_threshold']==.5 and prepared['models']==MODELS)
    check(sha(folder/'inference_inputs.json')==prepared['inference_inputs_sha256'])
    check(read(folder/'validation_references.json')==saved['references'] and sha(folder/'validation_references.json')==prepared['references_sha256'])
    for row,old in zip(inputs,pool):
        if row['coco_id']!=old['coco_id'] or row['candidates']!=old['candidates']:raise ValueError('Candidate inference payload differs')
        if set(row)!={'coco_id','candidates','filename','image_path','image_sha256','width','height','detected'}:raise ValueError('Unexpected inference fields')
        if int(Path(row['filename']).stem.split('_')[-1])!=row['coco_id'] or Path(row['image_path']).name!=row['filename']:raise ValueError('Filename/ID mismatch')
        if min(row['width'],row['height'])<=0 or len(row['image_sha256'])!=64:raise ValueError('Invalid image metadata')
        if any(not np.isfinite(v) or not .5<=v<=1 for v in row['detected'].values()):raise ValueError('Invalid accepted detector confidence')
    check(len(inputs)==5000)
    models=read(folder/'models.json');coverage=read(folder/'token_coverage.json')
    check(set(models)==set(MODELS))
    for name,m in models.items():
        check(m['id']==MODELS[name]['id'] and m['revision']==MODELS[name]['revision'] and Path(m['path']).name==m['revision'])
        check({'config.json','preprocessor_config.json','pytorch_model.bin'}<=set(m['files_sha256']))
        check(all(len(h)==64 for h in m['files_sha256'].values()))
        check(0<coverage[name]['max_tokens']<=coverage[name]['limit'])
    scored={};workers=[]
    for rank in (0,1):
        w=read(folder/f'worker{rank}.json');workers.append(w);path=folder/f'scores_rank{rank}.jsonl'
        part=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        check(w['complete'] and w['images']==2500 and w['scored_pairs']==12500 and w['models']==MODELS and w['fp32'] and not w['training'])
        check(sha(path)==w['scores_sha256'])
        check([r['coco_id'] for r in part]==[r['coco_id'] for r in inputs[rank::2]])
        scored.update({r['coco_id']:r for r in part})
    check(len(scored)==5000)
    ranks={k:[] for k in POLICIES};probability_error=0.
    for row in inputs:
        s=scored[row['coco_id']]
        if s['image_sha256']!=row['image_sha256']:raise ValueError('Raw-image identity differs')
        chosen=select(dict(row,clipscore=s['clipscore'],itm_logits=s['itm_logits']))
        if chosen!=s['selected_ranks']:raise ValueError('Selector replay differs')
        logits=np.asarray(s['itm_logits']);p=1/(1+np.exp(-(logits[:,1]-logits[:,0])))
        probability_error=max(probability_error,float(np.abs(p-np.asarray(s['itm_probability'])).max()))
        for policy,k in chosen.items():ranks[policy].append(k)
    check(ranks==read(folder/'selected_ranks.json') and probability_error<2e-7)
    check(read(folder/'metric_preflight.json')['passed'] and read(folder/'metric_preflight.json')['metrics']=={k:v for k,v in saved['metrics'].items() if k!='NLL'})
    stats={};chair_by={};predictions_by={}
    ids=sorted(r['coco_id'] for r in inputs)
    for policy in POLICIES:
        predictions=read(folder/(policy+'_captions.json'));check_rows(predictions,ids)
        check(predictions==[dict(coco_id=r['coco_id'],caption=r['candidates'][k]['caption']) for r,k in zip(inputs,ranks[policy])])
        check(read(folder/(policy+'_chair_input.json'))==[dict(image_id=r['coco_id'],caption=r['caption']) for r in predictions])
        chair=read(folder/(policy+'_chair_output.json'));m,c=chair_counts(chair,predictions);reported=result['metrics'][policy]
        check(c==reported['counts'] and all(abs(v-reported[k])<1e-12 for k,v in m.items()))
        per=read(folder/(policy+'_cider_per_image.json'));per_map={r['coco_id']:r['CIDEr'] for r in per}
        check(len(per)==len(per_map)==5000 and set(per_map)==set(ids))
        check(abs(np.mean(list(per_map.values()))-reported['CIDEr'])<1e-12)
        lengths=[len(r['caption'].split()) for r in predictions]
        check(abs(np.mean(lengths)-reported['mean_words'])<1e-12 and sum(n<=2 for n in lengths)==reported['short_le2'])
        sentences={r['image_id']:r for r in chair['sentences']};chair_by[policy]=sentences
        predictions_by[policy]={r['coco_id']:r['caption'] for r in predictions}
        stats[policy]=np.asarray([[per_map[i],bool(sentences[i]['mscoco_hallucinated_words']),len(sentences[i]['mscoco_hallucinated_words']),
            len(sentences[i]['mscoco_generated_words']),len(set(sentences[i]['mscoco_generated_words'])&set(sentences[i]['mscoco_gt_words'])),
            len(set(sentences[i]['mscoco_gt_words']))] for i in ids],dtype=float)
    base=chair_by['decoder_rank0'];comparisons={}
    check(all(set(base[i]['mscoco_gt_words'])==set(chair_by[p][i]['mscoco_gt_words']) for i in ids for p in POLICIES))
    check(result['metrics']['decoder_rank0']=={k:v for k,v in saved['metrics'].items() if k!='NLL'})
    for p in POLICIES[1:]:
        other=chair_by[p];changes=result['changes'][p]
        check(changes['changed_captions']==sum(k!=0 for k in ranks[p]))
        check(changes['rank_histogram']=={str(k):ranks[p].count(k) for k in range(5)})
        check(changes['CHAIR_repairs']==sum(bool(base[i]['mscoco_hallucinated_words']) and not other[i]['mscoco_hallucinated_words'] for i in ids))
        check(changes['CHAIR_new_flags']==sum(not base[i]['mscoco_hallucinated_words'] and bool(other[i]['mscoco_hallucinated_words']) for i in ids))
    old_per={r['coco_id']:r['CIDEr'] for r in read(HERE/'scst_full1_ce_review_text/evaluations/end/cider_per_image.json')}
    check(all(abs(old_per[i]-v)<1e-12 for i,v in zip(ids,stats['decoder_rank0'][:,0])))
    log=(folder/'notebook.log').read_text(encoding='utf-8')
    check(all(s in log for s in ('Ran 6 tests','Exact163 API routing mock PASS','Exact rank0 VALID5000 metric replay PASS','Isolated pinned HF dependencies PASS')))
    check('Traceback' not in log and 'FAILED' not in log)
    # Conditional paired image bootstrap; recompute count ratios after resampling.
    pairs=[('clip_object_fixed','decoder_rank0'),('blip_itm','decoder_rank0'),('clip_object_fixed','blip_itm')]
    rng=np.random.default_rng(20261105);draws={pair:[] for pair in pairs};repeats=10000
    for _ in range(repeats//100):
        ix=rng.integers(0,5000,size=(100,5000));samples={p:points(stats[p][ix].sum(axis=1),5000) for p in POLICIES}
        for pair in pairs:draws[pair].append(samples[pair[0]]-samples[pair[1]])
    names=['CIDEr','CHAIRs','CHAIRi','Recall']
    for a,b in pairs:
        delta=points(stats[a].sum(0),5000)-points(stats[b].sum(0),5000)
        intervals=np.quantile(np.concatenate(draws[(a,b)]),[.025,.975],axis=0)
        comparisons[a+'_minus_'+b]=dict(delta={k:float(v) for k,v in zip(names,delta)},
            paired_image_bootstrap95={k:[float(intervals[0,j]),float(intervals[1,j])] for j,k in enumerate(names)})
    original=read(HERE/'scst_full1_ce_review_text/evaluations/original/result.json')['metrics']
    check(read(HERE/'scst_full1_ce_review_text/validation_references.json')==saved['references'])
    whole_recipe=dict(CIDEr_points=(result['metrics']['clip_object_fixed']['CIDEr']-original['CIDEr'])*100,
        CHAIR_flags_before=original['counts']['hallucinated_captions'],CHAIR_flags_after=206,
        note='CSA15 XE -> SCST+CE full1 + fixed reranker; combined stages, not isolated rerank effect')
    report=dict(status='RERANK_STORED_TEXT_AUDIT_PASS',checks=checks,source=source['commit'],
        zip_sha256=read(folder/'saved_text_receipt.json')['zip_sha256'],metrics=result['metrics'],changes=result['changes'],
        comparisons=comparisons,bootstrap=dict(repeats=repeats,seed=20261105,unit='paired image',ratios_recomputed=True,
            limitation='conditional unadjusted intervals; omit training-seed variance, repeated VALID exposure and policy-choice uncertainty'),
        whole_recipe_vs_CSA15=whole_recipe,models=result['models'],worker_seconds=[w['seconds'] for w in workers],
        max_saved_probability_replay_error=probability_error,actual_reranking=True,test_used=False,training=False,
        oracle_used_for_selection=False,limitations=['No author model/scorer execution; stored CIDEr per-image means/CHAIR counts/selection rechecked, not official scorers rerun.',
            'No raw JPEGs, detector source or model weights in ZIP: their SHA/forward claims are user-run evidence, not independently verified here.',
            'External BLIP COCO finetuning split unverified; detector absence is not true absence; CLIP processor differs old cached-CLS source.',
            'CHAIR is object-category hallucination, not full relation/attribute truth or human confirmation.'])
    save(HERE/'RERANK_VALID_RESULT.json',report)
    print(json.dumps(dict(checks=checks,comparisons=comparisons,whole_recipe=whole_recipe,probability_error=probability_error),indent=2))


if __name__=='__main__':run()
