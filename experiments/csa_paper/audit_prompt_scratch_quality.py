"""Audit stored scorer outputs and recompute counts/paired statistics only; no scorers/models."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from saved_quality_data import check_rows,chair_counts
from score_prompt_scratch_quality import paired

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def audit(cache):
    record=json.loads((Path(cache)/'archive_text.json').read_text(encoding='utf-8'));f=record['files'];read=lambda n:json.loads(f[n]);checks=[]
    def check(label,condition):
        if not condition:raise ValueError(label)
        checks.append(label)
    result=read('comparison.json');source=read('source_receipt.json');provenance=read('evaluator_provenance.json')
    check('complete',result['status']=='PROMPT_SCRATCH_QUALITY_COMPLETE' and read('run_status.json')==dict(complete=True,model_executed=False,new_generation=False,test_used=False))
    check('no_replay_mismatch',result['replay_mismatches']==read('replay_mismatches.json')==[])
    check('expected_source',source['commit']=='17bf52c3f17c7b44e67f873d2405dcda2bb55a92' and len(source['sha256'])==6)
    for n,h in source['sha256'].items():check('source/'+n,hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=ROOT)).hexdigest()==h)
    check('evaluator',provenance['packages']['pycocoevalcap']=='1.2' and provenance['chair_cache_sha256']=='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3' and provenance['reference_tokens_reused_exactly'])
    payload=json.loads(gzip.decompress((HERE/'prompt_scratch_valid5000_saved.json.gz').read_bytes()));ids={int(i) for i in payload['reference_ptb']}
    per={};rows={}
    for name,model in payload['models'].items():
        rows[name]=model['rows'];check_rows(rows[name],ids)
        inp=read(name+'_chair_input.json');chair=read(name+'_chair_output.json');point=read(name+'_metrics.json');raw=read(name+'_per_image.json')
        check(name+'/point',point==result['metrics'][name] and point['epoch']==model['epoch'] and abs(point['CIDEr']-model['expected_CIDEr'])<1e-12)
        check(name+'/input',inp==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows[name]])
        metric,counts=chair_counts(chair,rows[name]);check(name+'/counts',counts==point['counts'] and all(abs(v-point[k])<1e-12 for k,v in metric.items()))
        per[name]={int(i):v for i,v in raw.items()};check(name+'/per_image_ids',len(per[name])==5000 and set(per[name])==ids)
        sentences={r['image_id']:r for r in chair['sentences']}
        check(name+'/per_image_values',all(np.isfinite(v['CIDEr']) and v['hallucination']==bool(sentences[i]['mscoco_hallucinated_words']) for i,v in per[name].items()) and abs(np.mean([v['CIDEr'] for v in per[name].values()])-point['CIDEr'])<1e-12)
        check(name+'/caption_lengths',abs(np.mean([len(r['caption'].split()) for r in rows[name]])-point['mean_words'])<1e-12)
    comparisons={}
    for name in ('PROMPT9','PROMPT10'):
        computed=paired(per['CONTROL10'],per[name]);stored=result['paired'][name+'_minus_CONTROL10']
        check(name+'/paired_statistics',all(np.allclose(v,stored[k],rtol=0,atol=1e-12) if isinstance(v,(float,list)) else v==stored[k] for k,v in computed.items()))
        check(name+'/net_hallucination_counts',computed['repaired']-computed['introduced']==result['metrics']['CONTROL10']['counts']['hallucinated_captions']-result['metrics'][name]['counts']['hallucinated_captions'])
        counts=result['metrics'][name]['counts'];base=result['metrics']['CONTROL10']['counts']
        comparisons[name]=dict(**stored,CIDEr_delta_points=stored['CIDEr_delta']*100,CIDEr_paired95_points=(np.array(stored['CIDEr_paired95'])*100).tolist(),CHAIRs_delta_pp=stored['CHAIRs_delta']*100,
            hallucinated_caption_relative_reduction=(base['hallucinated_captions']-counts['hallucinated_captions'])/base['hallucinated_captions'],
            recalled_unique_delta=counts['recalled_unique']-base['recalled_unique'],object_mentions_delta=counts['object_mentions']-base['object_mentions'],
            changed_captions=sum(a['caption']!=b['caption'] for a,b in zip(sorted(rows['CONTROL10'],key=lambda r:r['coco_id']),sorted(rows[name],key=lambda r:r['coco_id']))))
    report=dict(status='PROMPT_SCRATCH_QUALITY_REVIEW_COMPLETE',archive_sha256=record['archive_sha256'],metadata_checks=len(checks),checks=checks,metrics=result['metrics'],comparisons=comparisons,
        evaluator_provenance=provenance,model_or_scorer_executed_by_author=False,selection_unchanged=dict(CONTROL=10,PROMPT=9),test_used=False,
        limits=result['limits']+['CIDEr CI including zero does not establish equivalence or noninferiority.','No text-zero/shuffle/off result yet; semantic text contribution versus adapter/ROI capacity is unresolved.'])
    (HERE/'PROMPT_SCRATCH_QUALITY_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Quality metadata/count/statistics checks:',len(checks),'PASS');print(json.dumps(comparisons))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args();audit(a.cache)
