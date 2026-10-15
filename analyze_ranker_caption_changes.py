"""Explain pilot CIDEr changes descriptively; no threshold/model selection."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from captioning.object_consistency import extract_caption_objects
from captioning.hoi_caption_reranking import extract_claims


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        s=read('run_summary.json');cfg=read('protocol.json');gt=read('validation_gt.json')
        assert s['status']=='PILOT_COMPLETE' and not s['test_used']
        rows={r['image_id']:r for r in read('feature_candidates.json')['data']}
        val=cfg['groups']['validation'];ids=[r['image_id'] for r in val]
        predictions={m:{r['image_id']:r['caption'] for r in read(m+'_predictions.json')} for m in ('fixed','regional')}
        chairs={m:{r['image_id']:r for r in read(m+'_chair_output.json')['sentences']} for m in predictions}
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    refs={i:[] for i in ids}
    for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
    token_refs=PTBTokenizer().tokenize(refs)
    text={m:PTBTokenizer().tokenize({i:[{'caption':predictions[m][i]}] for i in ids}) for m in predictions}
    per_image={}
    for m in predictions:
        mean,scores=Cider().compute_score(token_refs,text[m]);per_image[m]=scores
        assert abs(mean-s['metrics'][m]['caption']['CIDEr'])<1e-10
    groups=defaultdict(list);changes=[]
    for j,image in enumerate(val):
        i=image['image_id'];coco=image['coco_id'];left=predictions['fixed'][i];right=predictions['regional'][i]
        if left==right:continue
        delta=float(per_image['regional'][j]-per_image['fixed'][j])
        a_chair=chairs['fixed'][coco]['metrics']['CHAIRs'];b_chair=chairs['regional'][coco]['metrics']['CHAIRs']
        chair='removed_hallucination' if a_chair and not b_chair else 'introduced_hallucination' if b_chair and not a_chair else 'both_hallucinated' if a_chair else 'both_clean'
        object_same=set(extract_caption_objects(left))==set(extract_caption_objects(right))
        claim=lambda x:{tuple(r[k] for k in ('subject','verb','object')) for r in extract_claims(x)}
        claim_same=claim(left)==claim(right)
        length=len(right.split())-len(left.split())
        for key in (chair,'same_objects' if object_same else 'different_objects','same_sixverb_claims' if claim_same else 'different_sixverb_claims',
                    'shorter' if length<0 else 'longer' if length>0 else 'same_length',
                    'same_PTB' if text['fixed'][i]==text['regional'][i] else 'different_PTB'):
            groups[key].append(delta)
        rank=lambda caption:next(k for k,c in enumerate(rows[i]['candidates']) if c['caption']==caption)
        changes.append({'image_id':i,'coco_id':coco,'filename':image['filename'],'fixed':left,'regional':right,
            'fixed_rank':rank(left),'regional_rank':rank(right),'CIDEr_delta':delta,'chair_transition':chair,
            'same_object_classes':object_same,'same_sixverb_claims':claim_same,'word_count_delta':length})
    d=np.asarray([r['CIDEr_delta'] for r in changes]);assert len(changes)==s['metrics']['regional']['changed_vs_fixed']==165
    assert abs(d.sum()/512-(s['metrics']['regional']['caption']['CIDEr']-s['metrics']['fixed']['caption']['CIDEr']))<1e-10
    result={'source_sha256':hashlib.sha256(Path(a.archive).read_bytes()).hexdigest(),'images':512,'changed':165,
        'CIDEr_changes':{'better':int((d>1e-10).sum()),'worse':int((d< -1e-10).sum()),'tied':int((abs(d)<=1e-10).sum()),
            'positive_contribution_to_corpus':float(d[d>0].sum()/512),'negative_contribution_to_corpus':float(d[d<0].sum()/512),
            'net_delta':float(d.sum()/512)},
        'descriptive_groups':{k:{'images':len(v),'contribution_to_corpus_CIDEr_delta':float(sum(v)/512)} for k,v in groups.items()},
        'decision':'Diagnostic only. No new model,threshold,epoch,weight or validation-based selection.',
        'limitations':['Reference CIDEr changes do not establish visual correctness.','Lexical class/claim grouping is incomplete.',
                       'CHAIR clean does not imply correct relation,attribute or whole caption.','Repeated inspection of development validation512.'],
        'new_training_or_notebook':False,'test_used':False}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'changed_cases.json').write_text(json.dumps(sorted(changes,key=lambda r:r['CIDEr_delta']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    print('TOP LOSSES',json.dumps(sorted(changes,key=lambda r:r['CIDEr_delta'])[:3],ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
