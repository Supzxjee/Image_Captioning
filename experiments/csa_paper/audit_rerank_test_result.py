"""Audit cached TEST166 text/receipts and paired counts; no models or scorers."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from real_bank_data import read,save,sha
from candidate_data import baseline,content
from full_xe_decode import check_beam_rows
from rerank_test_data import fixed_rank,POLICIES
from saved_quality_data import chair_counts,check_rows
from audit_rerank_result import points

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def run():
    folder=HERE/'rerank_test_v2_review_text';checks=0
    def check(value):
        nonlocal checks
        if not value:raise ValueError('TEST audit failed at check'+str(checks+1))
        checks+=1
    result=read(folder/'result.json');source=read(folder/'source_receipt.json');lock=read(folder/'test_lock.json');expected=read(HERE/'RERANK_TEST_INPUTS.json')
    check(read(folder/'saved_text_receipt.json')['crc_pass'])
    check(result['status']=='CSA161_FIXED_RERANK_TEST5000_COMPLETE' and result['images']==5000 and result['test_used'])
    check(not any(result[k] for k in ('training','tuned_on_test','oracle_used_for_selection')) and result['prior_project_test_exposure'])
    check(read(folder/'run_status.json')==dict(complete=True,training=False,test_used=True,tuned_on_test=False))
    for n,h in source['sha256'].items():check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
    for n in ('experiments/csa_paper/rerank_test_data.py','captioning/multiscore_reranking.py','captioning/object_consistency.py'):
        check(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==source['sha256'][n])
    check(source['gpu']==dict(built=True,available=True,count=2))
    check(lock['checkpoint_sha256']==expected['checkpoint_sha256']==result['checkpoint_sha256'])
    check(lock['checkpoint_version']==expected['checkpoint_version'] and lock['cache_version']==expected['test_cache_version'])
    check(lock['cache_input_mode']=='CPU_original_H5_exact_byte_match' and lock['selector']==result['selector']==expected['frozen_selector'])
    check(lock['clip']==expected['clip'] and lock['images']==5000 and lock['beam']==5 and lock['max_new_tokens']==20)
    check(lock['cache_sha256']==expected['test_receipt']['files'])
    restored=read(folder/'cache_restoration.json');receipt=read(folder/'restored_test_cache/test_receipt.json')
    check(restored['status']=='EXACT_TEST_CACHE_RESTORED' and restored['mode']=='original_H5_CPU_pooling' and not restored['neural_extraction'])
    check(restored['file_sha256']==lock['cache_sha256'] and restored['reference_cache_version']==lock['cache_version'])
    old=expected['test_receipt']
    for k in ('status','images','group','dataset_json_sha256','detector_sha256','feature_profile','training_images','validation_images','files','test_id_order_sha256'):
        check(receipt[k]==old[k])
    for n in ('test_metadata.json','vocabulary.json'):check(sha(folder/'restored_test_cache'/n)==lock['cache_sha256'][n])
    metadata=read(folder/'restored_test_cache/test_metadata.json');vocab=read(folder/'restored_test_cache/vocabulary.json');inputs=read(folder/'inference_inputs.json')
    refs=read(folder/'test_references.json');ids=[r['coco_id'] for r in metadata]
    check(len(metadata)==len(set(ids))==len(inputs)==len(refs)==5000)
    check(hashlib.sha256(str(ids).encode()).hexdigest()==old['test_id_order_sha256'])
    check(not set(ids)&set(baseline()['order']) and vocab==baseline()['vocabulary'])
    check([r['coco_id'] for r in inputs]==ids and all(r['source_split']=='test' and r['group']=='test' for r in metadata))
    check(refs==[dict(coco_id=r['coco_id'],references=r['captions']) for r in metadata])
    check(sha(folder/'inference_inputs.json')==lock['inference_sha256'] and sha(folder/'test_references.json')==lock['references_sha256'])
    for r,m in zip(inputs,metadata):
        if set(r)!={'coco_id','filename','image_path','image_sha256','detected'}:raise ValueError('Unexpected inference fields')
        if r['filename']!=m['filename'] or Path(r['image_path']).name!=r['filename'] or len(r['image_sha256'])!=64:raise ValueError('Wrong image metadata')
        if any(not np.isfinite(v) or not .5<=v<=1 for v in r['detected'].values()):raise ValueError('Invalid detector consistency input')
    check(True)
    beams={};scores={};decode_workers=[];score_workers=[]
    for rank in (0,1):
        path=folder/f'beams_rank{rank}.jsonl';rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()];w=read(folder/f'decode_worker{rank}.json');decode_workers.append(w)
        check(w['complete'] and w['images']==2500 and w['optimizer_steps']==0 and w['beam']==5 and w['max_new_tokens']==20)
        check(w['checkpoint_sha256']==lock['checkpoint_sha256'] and sha(path)==w['beams_sha256'])
        check([r['coco_id'] for r in rows]==ids[rank::2])
        for r in rows:
            c=r['candidates'];check_beam_rows([([x['ids'] for x in c],[x['logprob'] for x in c])],1,10000)
            if any(x['caption']!=' '.join(vocab[t] for t in content(x['ids'])) for x in c):raise ValueError('Beam caption/token mismatch')
            beams[r['coco_id']]=r
        path=folder/f'scores_rank{rank}.jsonl';rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()];w=read(folder/f'worker{rank}.json');score_workers.append(w)
        check(w['complete'] and w['images']==2500 and w['scored_pairs']==12500 and not w['training'])
        check(w['clip_revision']==lock['clip']['revision'] and sha(path)==w['scores_sha256'])
        check([r['coco_id'] for r in rows]==ids[rank::2]);scores.update({r['coco_id']:r for r in rows})
    check(len(beams)==len(scores)==5000)
    chosen=[]
    for row in inputs:
        s=scores[row['coco_id']]
        if s['image_sha256']!=row['image_sha256']:raise ValueError('Image-score identity mismatch')
        k=fixed_rank(beams[row['coco_id']],s['clipscore'],row['detected'])
        if k!=s['selected_rank']:raise ValueError('TEST selector replay differs')
        chosen.append(k)
    check(chosen==read(folder/'selected_ranks.json'))
    asset=read(folder/'clip_assets.json');check({k:v for k,v in asset.items() if k!='path'}==lock['clip'])
    coverage=read(folder/'token_coverage.json');check(coverage['images']==5000 and coverage['captions']==25000 and coverage['max_tokens']<=coverage['limit']==77)
    check(read(folder/'metric_preflight.json')==dict(passed=True,dependencies_only=True,no_test_metrics_yet=True))
    stats={};sentences_by={}
    for policy in POLICIES:
        predictions=read(folder/(policy+'_captions.json'));check_rows(predictions,set(ids))
        check(predictions==[dict(coco_id=i,caption=beams[i]['candidates'][0 if policy=='decoder_rank0' else k]['caption']) for i,k in zip(ids,chosen)])
        check(read(folder/(policy+'_chair_input.json'))==[dict(image_id=r['coco_id'],caption=r['caption']) for r in predictions])
        chair=read(folder/(policy+'_chair_output.json'));m,c=chair_counts(chair,predictions);point=result['metrics'][policy]
        check(c==point['counts'] and all(abs(v-point[k])<1e-12 for k,v in m.items()))
        per=read(folder/(policy+'_cider_per_image.json'));index={r['coco_id']:r['CIDEr'] for r in per}
        check(len(per)==len(index)==5000 and set(index)==set(ids) and abs(np.mean(list(index.values()))-point['CIDEr'])<1e-12)
        lengths=np.array([len(r['caption'].split()) for r in predictions]);check(abs(lengths.mean()-point['mean_words'])<1e-12 and int((lengths<=2).sum())==point['short_le2'])
        sentences={r['image_id']:r for r in chair['sentences']};sentences_by[policy]=sentences
        stats[policy]=np.array([[index[i],bool(sentences[i]['mscoco_hallucinated_words']),len(sentences[i]['mscoco_hallucinated_words']),len(sentences[i]['mscoco_generated_words']),
            len(set(sentences[i]['mscoco_generated_words'])&set(sentences[i]['mscoco_gt_words'])),len(set(sentences[i]['mscoco_gt_words']))] for i in ids],dtype=float)
    a,b=[sentences_by[p] for p in POLICIES];change=result['changes']
    check(all(set(a[i]['mscoco_gt_words'])==set(b[i]['mscoco_gt_words']) for i in ids))
    check(change['changed_captions']==sum(k!=0 for k in chosen) and change['rank_histogram']=={str(k):chosen.count(k) for k in range(5)})
    check(change['CHAIR_repairs']==sum(bool(a[i]['mscoco_hallucinated_words']) and not b[i]['mscoco_hallucinated_words'] for i in ids))
    check(change['CHAIR_new_flags']==sum(not a[i]['mscoco_hallucinated_words'] and bool(b[i]['mscoco_hallucinated_words']) for i in ids))
    check(change['CHAIR_repairs']-change['CHAIR_new_flags']==58)
    log=(folder/'notebook.log').read_text(encoding='utf-8')
    check('Ran 7 tests' in log and 'Restored TEST NPZ/metadata/vocab byte-for-byte match' in log and 'TEST ROI CACHE COMPLETE' in log)
    check('Traceback' not in log and 'FAILED' not in log and 'Isolated pinned HF dependencies PASS' in log)
    old_test=read(HERE/'csa10_test_review_text/result.json')['metrics']
    check(old_test['counts']['reference_unique']==15746 and result['metrics']['decoder_rank0']['counts']['reference_unique']==15746)
    rng=np.random.default_rng(20261105);draws=[]
    for _ in range(100):
        ix=rng.integers(0,5000,size=(100,5000))
        draws.append(points(stats['clip_object_fixed'][ix].sum(1),5000)-points(stats['decoder_rank0'][ix].sum(1),5000))
    ci=np.quantile(np.concatenate(draws),[.025,.975],axis=0);delta=points(stats['clip_object_fixed'].sum(0),5000)-points(stats['decoder_rank0'].sum(0),5000)
    names=('CIDEr','CHAIRs','CHAIRi','Recall')
    report=dict(status='RERANK_TEST_STORED_TEXT_AUDIT_PASS',checks=checks,user_reported_output_source='ducanh2403/2-vi-cai-kia-qua-nhieu/166',
        zip_sha256=read(folder/'saved_text_receipt.json')['zip_sha256'],source=source['commit'],checkpoint_sha256=lock['checkpoint_sha256'],
        metrics=result['metrics'],changes=change,delta={k:float(v) for k,v in zip(names,delta)},
        paired_image_bootstrap95={k:[float(ci[0,j]),float(ci[1,j])] for j,k in enumerate(names)},
        bootstrap=dict(repeats=10000,seed=20261105,unit='paired image',ratios_recomputed=True,limitation='conditional; training seeds and prior project TEST exposure not represented'),
        descriptive_vs_CSA10=dict(CIDEr_points=(result['metrics']['clip_object_fixed']['CIDEr']-old_test['CIDEr'])*100,
            CHAIR_flags_before=old_test['counts']['hallucinated_captions'],CHAIR_flags_after=result['metrics']['clip_object_fixed']['counts']['hallucinated_captions'],
            limitation='extra XE epochs, SCST and reranking; not isolated CSA/alignment or matched training budget'),
        cache_mode=lock['cache_input_mode'],cache_reference_version=lock['cache_version'],cache_files_sha256=lock['cache_sha256'],
        cache_restore_seconds=receipt['seconds'],decode_seconds=[r['seconds'] for r in decode_workers],score_seconds=[r['seconds'] for r in score_workers],
        test_used=True,training=False,tuned_on_test=False,oracle_used_for_selection=False,prior_project_test_exposure=True,
        limitations=['No author Torch/model/scorer execution; selection/counts/metric means replayed, not official scorers rerun.',
            'Cache metadata/vocabulary SHA independently checked; NPZ/weights/JPEG/H5/detector binaries absent ZIP, their checks are user-run evidence.',
            'Detector absence may be a miss/topK; CHAIR object-category metric does not establish all relation/attribute truth or human quality.',
            'Frozen choice was made on VALID164; TEST has earlier project exposure, no clean-never-seen or multi-seed guarantee.'])
    save(HERE/'RERANK_TEST_RESULT.json',report)
    print(json.dumps(dict(checks=checks,delta=report['delta'],CI=report['paired_image_bootstrap95'],descriptive_vs_CSA10=report['descriptive_vs_CSA10']),indent=2))


if __name__=='__main__':run()
