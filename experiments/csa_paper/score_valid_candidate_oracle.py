"""USER-RUN CPU CIDEr on five saved VALID candidates; no selection tuning."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import math


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def oracle_summary(rows,values):
    if len(rows)!=len(values) or any(len(v)!=5 or not all(math.isfinite(s) for s in v) for v in values):
        raise ValueError('Incomplete/nonfinite candidate scores')
    n=len(rows)
    if not n:raise ValueError('Empty cohort')
    raw=sum(v[0] for v in values)/n*100
    selected=sum(v[r['selected_rank']] for r,v in zip(rows,values))/n*100
    best=sum(max(v) for v in values)/n*100
    return dict(images=n,rank0_CIDEr_x100=raw,selected_CIDEr_x100=selected,
        oracle_CIDEr_x100=best,oracle_minus_selected_points=best-selected,
        available_gain_over_rank0_points=best-raw,
        fraction_oracle_gain_captured=(selected-raw)/(best-raw) if best>raw+1e-10 else None,
        selected_is_CIDEr_best=sum(v[r['selected_rank']]>=max(v)-1e-10 for r,v in zip(rows,values)))


def run(bundle_path,expected_sha,out):
    raw=Path(bundle_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Changed VALID input package')
    bundle=json.loads(gzip.decompress(raw))
    if bundle['schema']!='saved_VALID5000_candidates_v1' or bundle['test_used'] or bundle['training']:
        raise ValueError('Wrong bundle/split')
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from importlib.metadata import version
    if version('pycocoevalcap')!='1.2':raise ValueError('Scorer version differs')
    out.mkdir(parents=True,exist_ok=True)
    results={};tokenizer=PTBTokenizer()
    for arm,source in bundle['arms'].items():
        rows=source['records'];refs=source['references'];ids=[r['coco_id'] for r in refs]
        if len(rows)!=5000 or len(set(ids))!=5000 or {r['coco_id'] for r in rows}!=set(ids):raise ValueError('Wrong cohort')
        rows_by_id={r['coco_id']:r for r in rows}
        ref=tokenizer.tokenize({r['coco_id']:[dict(caption=s) for s in r['references']] for r in refs})
        # Every rank uses the SAME 5000-reference corpus/IDF, not a 25000-item expansion.
        per={i:[None]*5 for i in ids}
        for rank in range(5):
            print('CPU saved VALID candidates:',arm,'rank',rank,'/4',flush=True)
            hyp=tokenizer.tokenize({i:[dict(caption=rows_by_id[i]['candidates'][rank]['caption'])] for i in ids})
            point,scores=Cider().compute_score(ref,hyp)
            if len(scores)!=5000 or not math.isfinite(float(point)):raise ValueError('Scorer incomplete')
            for i,s in zip(ref,scores):per[i][rank]=float(s)
            if rank==0 and any(abs(per[i][0]-rows_by_id[i]['rank0_CIDEr'])>1e-7 for i in ids):
                raise ValueError('Rank0 historical per-image replay differs')
        if any(abs(per[i][rows_by_id[i]['selected_rank']]-rows_by_id[i]['selected_CIDEr'])>1e-7 for i in ids):
            raise ValueError('Selected historical per-image replay differs')
        values=[per[r['coco_id']] for r in rows]
        result=oracle_summary(rows,values)
        result['source_version']=source['source_version']
        results[arm]=result
        save(out/(arm+'_per_candidate_CIDEr.json'),[dict(coco_id=r['coco_id'],selected_rank=r['selected_rank'],CIDEr=per[r['coco_id']]) for r in rows])
        save(out/'progress.json',dict(complete_arms=list(results),results=results))
    save(out/'result.json',dict(status='VALID_CANDIDATE_ORACLE_COMPLETE',arms=results,bundle_sha256=expected_sha,
         training=False,decoding=False,test_used=False,selector_changed=False,
         limitation='Oracle uses VALID references; diagnostic only, not deployable selection or proof of visual correctness.'))
    print('VALID ORACLE COMPLETE',results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bundle,a.sha256,a.output)
