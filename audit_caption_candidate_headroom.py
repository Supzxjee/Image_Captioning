"""Reference-aware validation oracle: diagnostic ceiling, never a deployable ranker."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from captioning.hoi_caption_reranking import extract_claims
from captioning.object_consistency import extract_caption_objects


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        cfg=read('protocol.json');summary=read('run_summary.json')
        assert cfg['split']==summary['split']=='val' and not cfg['test_run']
        assert summary['status']=='VALIDATION_COMPLETE' and summary['count']==5000
        rows=read('val_hoi_candidates.json')['data'];gt=read('val_gt.json')
        predictions={n:read(n+'_predictions.json') for n in ('rank0','fixed_rerank','upt_rerank')}
    ids=[r['image_id'] for r in rows]
    assert len(ids)==len(set(ids))==5000
    refs={i:[] for i in ids}
    for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
    assert all(len(v)==5 for v in refs.values())
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    token_refs=PTBTokenizer().tokenize(refs)
    beams=PTBTokenizer().tokenize({r['image_id']:[{'caption':c['caption']} for c in r['candidates']] for r in rows})
    k=max(len(r['candidates']) for r in rows);scores=[]
    # Always score the same 5000 references, so each rank uses the same corpus IDF.
    for rank in range(k):
        text={r['image_id']:[beams[r['image_id']][min(rank,len(r['candidates'])-1)]] for r in rows}
        _,per_image=Cider().compute_score(token_refs,text)
        scores.append(per_image)
        print(f'Candidate rank {rank+1}/{k} scored',flush=True)
    scores=np.asarray(scores).T
    for j,r in enumerate(rows):scores[j,len(r['candidates']):]=-np.inf
    measured={};choices={}
    for name,pred in predictions.items():
        lookup={r['image_id']:r['caption'] for r in pred}
        assert len(pred)==len(lookup)==5000 and set(lookup)==set(ids)
        choice=[next(i for i,c in enumerate(r['candidates']) if c['caption']==lookup[r['image_id']]) for r in rows]
        choice=np.asarray(choice);choices[name]=choice
        measured[name]=float(scores[np.arange(len(rows)),choice].mean())
        assert abs(measured[name]-summary['metrics'][name]['caption']['CIDEr'])<1e-10
    oracle=scores.argmax(1);best=scores.max(1)
    measured['reference_oracle']=float(best.mean())
    comparisons={};records=[]
    for name,choice in choices.items():
        gain=best-scores[np.arange(len(rows)),choice]
        assert np.all(gain>=-1e-12)
        comparisons[name]={'mean_CIDEr_gap_to_oracle':float(gain.mean()),
            'images_with_strictly_better_CIDEr_candidate':int((gain>1e-10).sum()),
            'images_with_CIDEr_gain_at_least_point1':int((gain>=.1).sum()),
            'images_already_tied_for_best_CIDEr':int((gain<=1e-10).sum())}
    normalized_counts=[];object_changes=claim_changes=0;oracle_object_change=oracle_claim_change=0
    for j,r in enumerate(rows):
        texts=beams[r['image_id']];normalized_counts.append(len(set(texts)))
        objects=[tuple(sorted(extract_caption_objects(c['caption']))) for c in r['candidates']]
        claims=[tuple(sorted((v['subject'],v['verb'],v['object']) for v in extract_claims(c['caption']))) for c in r['candidates']]
        object_changes+=len(set(objects))>1;claim_changes+=len(set(claims))>1
        fixed=int(choices['fixed_rerank'][j]);best_rank=int(oracle[j])
        oracle_object_change+=objects[fixed]!=objects[best_rank]
        oracle_claim_change+=claims[fixed]!=claims[best_rank]
        records.append({'image_id':r['image_id'],'coco_id':r['coco_id'],'filename':r['filename'],
            'fixed_rank':fixed,'oracle_rank':best_rank,'fixed_caption':r['candidates'][fixed]['caption'],
            'oracle_caption':r['candidates'][best_rank]['caption'],
            'fixed_CIDEr':float(scores[j,fixed]),'oracle_CIDEr':float(best[j])})
    result={'source_archive_sha256':hashlib.sha256(Path(a.archive).read_bytes()).hexdigest(),
        'split':'val','images':5000,'candidate_count_distribution':dict(Counter(len(r['candidates']) for r in rows)),
        'CIDEr':measured,'oracle_gaps':comparisons,
        'diversity':{'mean_unique_PTB_captions':float(np.mean(normalized_counts)),
            'images_with_PTB_duplicate_candidates':sum(n<len(r['candidates']) for n,r in zip(normalized_counts,rows)),
            'images_with_different_explicit_object_class_sets':object_changes,
            'images_with_different_six_verb_claim_sets':claim_changes,
            'oracle_vs_fixed_different_object_sets':oracle_object_change,
            'oracle_vs_fixed_different_claim_sets':oracle_claim_change},
        'limitations':['Reference-aware CIDEr ceiling, not an executable selector or measured new model.',
            'CIDEr improvement does not establish visual truth, relation correctness or lower CHAIR.',
            'Lexical object/claim diversity is partial; does not measure all semantic differences.',
            'Existing reused validation, not independent test; no training or hyperparameter search.'],
        'new_notebook_started':False,'test_used':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'oracle_cases.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(out/'per_candidate_CIDEr.npz',image_ids=np.asarray(ids),scores=scores)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
