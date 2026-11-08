"""User-run CPU metrics and one locked VALID reranking calibration."""
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from weight_calibration_data import ORIGINAL,split_ids,grid,choose,chair_counts,metrics,fit,accept

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(a):
    bundle=json.loads(gzip.decompress(a.bundle.read_bytes()))
    expected=json.loads(gzip.decompress(a.parity.read_bytes()))
    if bundle['schema']!='saved_VALID5000_candidates_v1' or bundle['test_used'] or bundle['training']:raise ValueError('Wrong input scope')
    if sha(a.bundle)!=expected['bundle_sha256']:raise ValueError('Changed candidate bundle')
    from importlib.metadata import version
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    import chair
    # Trusted pinned pickle was created by standalone chair.py as __main__.CHAIR.
    globals()['CHAIR']=chair.CHAIR
    if version('pycocoevalcap')!='1.2':raise ValueError('Changed scorer version')
    if sha(a.chair_cache)!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    with a.chair_cache.open('rb') as f:evaluator=pickle.load(f)
    sources=bundle['arms']
    if set(sources)!={'CONTROL','PROMPT'}:raise ValueError('Wrong arms')
    rows={arm:{r['coco_id']:r for r in src['records']} for arm,src in sources.items()}
    ids=list(rows['CONTROL'])
    if set(ids)!=set(rows['PROMPT']):raise ValueError('Cohorts differ')
    splits=split_ids(ids)
    if splits!=json.loads(a.bundle.with_name('CALIBRATION_SPLIT.json').read_text(encoding='utf-8')):raise ValueError('Prelocked split differs')
    save(a.output/'split.json',splits)
    tokenizer=PTBTokenizer();refs={};hyps={};chairs={}
    for arm,src in sources.items():
        if src['references']!=sources['CONTROL']['references']:raise ValueError('References differ')
        refs[arm]=tokenizer.tokenize({r['coco_id']:[dict(caption=s) for s in r['references']] for r in src['references']})
        if set(refs[arm])!=set(ids):raise ValueError('Reference cohort differs')
        hyps[arm]=[];chairs[arm]={i:[None]*5 for i in ids}
        for rank in range(5):
            print('Historical replay / CHAIR:',arm,'rank',rank,flush=True)
            captions=[dict(image_id=i,caption=rows[arm][i]['candidates'][rank]['caption']) for i in ids]
            hyps[arm].append(tokenizer.tokenize({x['image_id']:[dict(caption=x['caption'])] for x in captions}))
            point,values=Cider().compute_score(refs[arm],hyps[arm][rank])
            for i,s in zip(refs[arm],values):
                row=rows[arm][i]
                if rank==0 and abs(float(s)-row['rank0_CIDEr'])>1e-7:raise ValueError('Rank0 CIDEr replay differs')
                if rank==row['selected_rank'] and abs(float(s)-row['selected_CIDEr'])>1e-7:raise ValueError('Selected CIDEr replay differs')
            cap=a.output/'chair_input.json';save(cap,captions)
            result=evaluator.compute_chair(str(cap),'image_id','caption')
            save(a.output/f'{arm}_CHAIR_rank{rank}.json',result)
            if len(result['sentences'])!=5000:raise ValueError('Incomplete CHAIR')
            seen=set()
            for sentence in result['sentences']:
                i=sentence['image_id']
                if i in seen or i not in rows[arm] or sentence['caption']!=rows[arm][i]['candidates'][rank]['caption']:raise ValueError('CHAIR output mapping differs')
                seen.add(i);counts=chair_counts(sentence);chairs[arm][i][rank]=counts
                old=expected['arms'][arm][str(i)]
                if rank==0 and counts!=old['rank0']:raise ValueError('Rank0 CHAIR replay differs')
                if rank==rows[arm][i]['selected_rank'] and counts!=old['selected']:raise ValueError('Selected CHAIR replay differs')
        if any(choose(row,ORIGINAL)!=row['selected_rank'] for row in rows[arm].values()):raise ValueError('Original selector replay differs')
    save(a.output/'replay.json',dict(status='PASS',arms=list(sources),images=5000))
    def score_split(name):
        subset=splits[name];data={}
        for arm in sources:
            r={i:refs[arm][i] for i in subset};data[arm]={i:[None]*5 for i in subset}
            for rank in range(5):
                print('Independent CIDEr corpus:',name,arm,rank,flush=True)
                _,values=Cider().compute_score(r,{i:hyps[arm][rank][i] for i in subset})
                for i,v in zip(r,values):data[arm][i][rank]=float(v)
            save(a.output/f'{name}_{arm}_CIDEr.json',data[arm])
        return data
    def aggregate(data,subset,weights):return {arm:metrics(rows[arm],data[arm],chairs[arm],subset,weights) for arm in sources}
    tune=score_split('TUNE')
    scores=[dict(weights=w,arms=aggregate(tune,splits['TUNE'],w)) for w in grid()]
    save(a.output/'tune_grid.json',scores);chosen=fit(scores)
    lock=dict(weights=chosen['weights'],tune=chosen['arms'],bundle_sha256=sha(a.bundle),
              split_sha256=sha(a.output/'split.json'),protocol='VALID_WEIGHT_CALIBRATION_PROTOCOL.md',test_used=False)
    save(a.output/'selection_lock.json',lock)
    check=score_split('CHECK')
    baseline=aggregate(check,splits['CHECK'],ORIGINAL);candidate=aggregate(check,splits['CHECK'],chosen['weights'])
    accepted=accept(candidate,baseline)
    save(a.output/'result.json',dict(status='VALID_WEIGHT_CALIBRATION_COMPLETE',selection_lock_sha256=sha(a.output/'selection_lock.json'),
         proposed_weights=chosen['weights'],retained_weights=chosen['weights'] if accepted else ORIGINAL,
         check_accept=accepted,CHECK_original=baseline,CHECK_proposed=candidate,
         training=False,decoding=False,test_used=False,
         limitation='Internal reused VALID split; point-estimate acceptance, no significance or new TEST claim.'))
    print('CALIBRATION COMPLETE; accepted:',accepted,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('bundle','parity','chair-cache','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a)
