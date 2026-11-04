"""Stored text/JSON/NumPy checks only; no ZIP reread, model, features or scorer."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from grounding_labels import positive_set

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def audit():
    d=json.loads((HERE/'review/grounding_audit_v2_loaded.json').read_text(encoding='utf-8'))
    text={Path(n).name:t for n,t in d['text'].items()}
    s=json.loads(text['summary.json']);receipt=json.loads(text['source_receipt.json'])
    checks=0
    def check(value):
        nonlocal checks
        if not value:raise AssertionError('Stored audit check failed at '+str(checks+1))
        checks+=1
    for name,h in receipt['sha256'].items():
        check(hashlib.sha256(subprocess.check_output(['git','show',receipt['commit']+':experiments/csa_paper/'+name],cwd=ROOT)).hexdigest()==h)
    check(s['status']=='CSA_GROUNDING_DATA_AUDIT_COMPLETE' and s['images']==10000)
    check(json.loads(text['run_status.json'])==dict(complete=True,model_executed=False,training_ready=False,manual_review_pending=True))
    check('failure.txt' not in text and not s['test_used'] and not s['model_executed'])
    cohort=json.loads(text['cohort.json']);ids={r['coco_id'] for r in cohort}
    check(len(cohort)==len(ids)==10000)
    check({r['split'] for r in cohort}<={'train','restval'})
    keys=[hashlib.sha256(('csa-grounded-attention-v1/42/'+r['filename']).encode()).hexdigest() for r in cohort]
    check(keys==sorted(keys))
    for n,k in [('cohort.json','cohort_sha256'),('lexicon.json','lexicon_sha256')]:check(hashlib.sha256(text[n].encode()).hexdigest()==s[k])
    counts=s['counts'];statuses=('eligible','missing_gt','no_positive','crowd','unknown_token','invalid_gt','truncated')
    check(sum(counts.get(k,0) for k in statuses)==s['mapped_mentions'])
    total=Counter()
    for row in s['per_class'].values():total.update(row)
    check(all(total[k]==counts.get(k,0) for k in statuses))
    check(s['positive_coverage_denominator']==counts['eligible']+counts['no_positive'])
    check(abs(s['positive_coverage']-counts['eligible']/s['positive_coverage_denominator'])<1e-15)
    check(s['coverage_gate_pass'] and s['positive_coverage']>=.8)
    expected=json.loads((HERE/'SAVE_FULL_ROI_V134_PACKAGE.json').read_text())['files']
    for name,h in s['checked_roi_payloads'].items():check(name.startswith('train_') and expected[name]==h)
    check(len(s['checked_roi_payloads'])==228)
    provenance=json.loads(text['annotation_provenance.json'])
    check(provenance['archive_sha256']=='031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009')
    lookup=json.loads(text['lexicon.json']);sample_stats={}
    for name in ('review_100.json','review_eligible_100.json'):
        rows=json.loads(text[name]);check(len(rows)==len({r['mention_id'] for r in rows})==100)
        sample_stats[name]=dict(Counter(r['status'] for r in rows))
        for r in rows:
            check(r['coco_id'] in ids and lookup[r['phrase']]==r['category_id'])
            check(r['target_position']==r['end']-1 and r['target_id']>=4)
            if r['status']=='eligible':
                gt=[dict(category_id=r['category_id'],iscrowd=0,bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]]) for b in r['gt_boxes']]
                p,status,_=positive_set(r['proposal_boxes'],['predicted_roi']*len(r['proposal_boxes']),gt,r['category_id'])
                # Samples lack region_kinds: equality here only validates saved box geometry.
                check(status=='eligible' and p==r['positive_regions'])
        if name=='review_eligible_100.json':check(all(r['status']=='eligible' for r in rows))
    target=HERE/'grounding_audit_v2_review_text';target.mkdir(exist_ok=True)
    for name in ('summary.json','source_receipt.json','annotation_provenance.json','run_status.json'):
        (target/name).write_text(text[name],encoding='utf-8')
    result=dict(status='DATA_PREPARATION_COMPLETE_REVIEW_PENDING',zip_sha256=d['zip_sha256'],metadata_checks=checks,
        source_commit=receipt['commit'],images=s['images'],mapped_mentions=s['mapped_mentions'],counts=counts,
        positive_coverage=s['positive_coverage'],eligible_fraction_of_mapped=counts['eligible']/s['mapped_mentions'],
        caption_mapping_fraction=counts['captions_with_mapped_mention']/counts['captions'],classes_observed=len(s['per_class']),
        eligible_person_fraction=s['per_class']['person']['eligible']/counts['eligible'],worker_seconds=s['seconds'],sample_statuses=sample_stats,
        training_ready=False,human_review_pending=True,model_executed=False,
        limits='Actual NPZ/GT/full mention labels absent ZIP; sample geometry checks do not verify image semantics or region_kinds. No human labels or caption-quality result.')
    (HERE/'GROUNDING_AUDIT_V2_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
