"""Recompute saved candidate decisions locally without model execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/noun_evidence_proposal'))
from candidate_data import A_SHA,BASE_SHA,ARCHIVE_SHA,digest,choose_pair,oracle,draw_seed


def audit(loaded):
    files=loaded['files'];obj={n:json.loads(s) for n,s in files.items() if n.endswith('.json')}
    cfg,summary=obj['protocol.json'],obj['numeric_summary.json']
    assert summary['status']=='CANDIDATE_AUDIT_COMPLETE' and not summary['trained'] and not cfg['test_used']
    assert cfg['input_sha256']['checkpoint']==summary['source_A_sha256']==A_SHA
    for name,sha in cfg['implementation_sha256'].items():
        assert hashlib.sha256(files['source/'+name].encode()).hexdigest()==sha,name
    archive=ROOT/'experiments/noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'
    assert digest(archive)==ARCHIVE_SHA
    with zipfile.ZipFile(archive) as z:groups=json.loads(z.read('groups.json'));sealed_targets=json.loads(z.read('targets.json'));classes=json.loads(z.read('receipt.json'))['classes']
    selected=sorted(groups['train'],key=lambda r:hashlib.sha256(('preference-bank-train-v1:'+r['filename']).encode()).hexdigest())[:2000]
    assert cfg['sample']==selected and cfg['train_ids']==[r['image_id'] for r in groups['train']]
    assert not {r['coco_id'] for r in selected}&{r['coco_id'] for r in groups['val']}
    original=[]
    for rank in (0,1):
        done=obj[f'complete_rank{rank}.json'];rows=obj[f'candidates_rank{rank}.json']
        assert done['status']=='COMPLETE' and done['images']==len(rows)==1000 and not done['trained']
        assert done['base_sha256']==BASE_SHA and done['greedy_parity'] and done['sampling_repeatable']
        assert hashlib.sha256(files[f'candidates_rank{rank}.json'].encode()).hexdigest()==done['candidates_sha256']
        assert [r['coco_id'] for r in rows]==[r['coco_id'] for r in selected[rank::2]]
        original+=rows
    original.sort(key=lambda r:r['image_id']);scored=obj['scored_candidates.json']
    assert len(scored)==2000 and len({r['image_id'] for r in scored})==2000
    valid=0;endless=0;unknown=0
    for raw,row in zip(original,scored):
        assert {k:v for k,v in row.items() if k!='candidates'}=={k:v for k,v in raw.items() if k!='candidates'}
        assert len(row['candidates'])==5
        for draw,(before,after) in enumerate(zip(raw['candidates'],row['candidates'])):
            assert {k:v for k,v in after.items() if k!='CIDEr'}==before and after['draw']==draw
            ids=after['token_ids'];assert ids[0]==1 and 1<len(ids)<=30 and np.isfinite(after['CIDEr'])
            ended=2 in ids[1:];assert ended==after['ended']
            actions=ids[1:ids.index(2,1)] if ended else ids[1:]
            assert after['valid']==(ended and bool(actions) and not bool(set(actions)&{0,1,3}))
            valid+=after['valid'];endless+=not ended;unknown+=3 in actions
    policy=json.loads(files['source/lexical_policy.json']);labels=obj['labels_train_sample.json'];recomputed=[];gains=[];counts=Counter()
    assert set(labels)=={str(r['image_id']) for r in selected}
    for row in scored:
        label=labels[str(row['image_id'])];sealed=sealed_targets[str(row['image_id'])]
        assert set(label['support'])=={c for c,y in zip(classes,sealed['y']) if y} and set(label['salient'])<=set(label['support'])
        args=(row['candidates'],label['support'],label['salient'],policy,label['conflict'])
        pair=choose_pair(*args);best=oracle(*args);gains.append(best['CIDEr']-row['candidates'][0]['CIDEr'])
        if pair:
            recomputed.append(dict(image_id=row['image_id'],coco_id=row['coco_id'],filename=row['filename'],**pair));counts.update(label['salient'])
    assert recomputed==obj['eligible_pairs.json']
    assert len(recomputed)==summary['valid_pair_images'] and counts==Counter(summary['salient_classes_in_pairs'])
    assert summary['valid_pair_fraction']==len(recomputed)/2000 and abs(summary['oracle_CIDEr_gain']-np.mean(gains))<1e-12
    assert abs(summary['greedy_CIDEr']-np.mean([r['candidates'][0]['CIDEr'] for r in scored]))<1e-12
    assert summary['numeric_screen_pass']==(len(recomputed)>=400 and np.mean(gains)>=.05)
    review=sorted(recomputed,key=lambda p:hashlib.sha256(('candidate-review-v1:'+str(p['coco_id'])).encode()).hexdigest())[:30]
    key={str(p['coco_id']):('Y' if draw_seed(p['coco_id'],9)%2 else 'X') for p in review}
    assert key==obj['review_key.json'] and len(key)==30
    fingerprint=hashlib.sha256(files['eligible_pairs.json'].encode()).hexdigest()
    assert fingerprint in files['review_30.html']
    (HERE/'review/review_30.html').write_text(files['review_30.html'],encoding='utf-8')
    return dict(status='SAVED_ARTIFACT_AUDIT_PASS',zip_sha256=loaded['zip_sha256'],trained=False,
        sample_images=2000,valid_pair_images=len(recomputed),valid_pair_fraction=len(recomputed)/2000,
        distinct_salient_classes=len(counts),numeric_screen_pass=True,
        greedy_TRAIN_CIDEr=summary['greedy_CIDEr'],guarded_oracle_TRAIN_CIDEr=summary['greedy_CIDEr']+summary['oracle_CIDEr_gain'],
        guarded_oracle_gain=summary['oracle_CIDEr_gain'],valid_candidates=valid,total_candidates=10000,
        no_EOS_candidates=endless,UNK_candidates=unknown,mean_unique_captions=float(np.mean([len({c['caption'] for c in r['candidates']}) for r in scored])),
        pairs_with_greedy_winner=sum(p['winner']['draw']==0 for p in recomputed),
        eligible_pair_images_with_non_greedy_winner=sum(p['winner']['draw']!=0 for p in recomputed),
        images_with_guarded_candidate_better_than_greedy=sum(g>0 for g in gains),
        human_review_status='PENDING_30_BLINDED_PAIRS',review_manifest_sha256=fingerprint,
        preparation_seconds=cfg['preparation_seconds'],worker_seconds=max(obj[f'complete_rank{r}.json']['seconds'] for r in (0,1)),
        decision='Numeric feasibility PASS; human gate and novelty unresolved. No automatic training/full10k bank.',
        limits=['CIDEr scorer not rerun; saved per-candidate scores and source provenance verified, decisions recomputed.',
                'GPU model/input tensor identities supported by worker receipts; model/HDF5 absent from review.',
                'Salience reference counts and detector conflicts supported by saved preparation/source; original full raw inputs not reopened.',
                'TRAIN-only oracle on five candidates, not a learned model gain or a validation result.'])


if __name__=='__main__':
    result=audit(json.loads((HERE/'review/loaded.json').read_text(encoding='utf-8')))
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
