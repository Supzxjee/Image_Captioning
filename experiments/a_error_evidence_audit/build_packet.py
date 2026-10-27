"""Reuse saved predictions only. No inference, training, parser model or metric run."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'review'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def order(i):return hashlib.sha256(('A-evidence-audit-20261027:'+str(i)).encode()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load(path):
    p=ROOT/path;x=json.loads(p.read_text(encoding='utf-8'))
    if 'json' in x:objects=x['json']
    else:objects={k:json.loads(v) for k,v in x.get('texts',x.get('files',{})).items() if k.endswith('.json') and isinstance(v,str)}
    return objects,dict(path=path,cache_sha256=sha(p),archive_sha256=x.get('zip_sha256'),archive=x.get('archive'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base,source=load('experiments/native_detail_pilot/review/loaded.json');sources=[source]
    cfg=base['protocol.json'];group={r['image_id']:r for r in cfg['groups']['val']}
    assert len(group)==5000
    branches={};scores={};chair={}
    def add(name,data,prefix,scope):
        pred=data[prefix+'_predictions.json'];rows={r['image_id']:r['caption'] for r in pred}
        assert len(rows)==len(pred) and set(rows)<=set(group)
        branches[name]=rows;scores[name]={r['image_id']:r['CIDEr'] for r in data[prefix+'_per_image_cider.json']}
        chair[name]={r['image_id']:r for r in data[prefix+'_chair_output.json']['sentences']}
        assert set(rows)==set(scores[name]) and {group[i]['coco_id'] for i in rows}==set(chair[name])
        return dict(images=len(rows),scope=scope)
    inventory={}
    for name in ('A','H','L'):inventory['A' if name=='A' else 'native_'+name]=add('A' if name=='A' else 'native_'+name,base,name,'same reused val5000')
    for arm in ('G','S'):
        data,src=load(f'experiments/noun_evidence_proposal/review/loaded_{arm}.json');sources.append(src)
        assert all(branches['A'][r['image_id']]==r['caption'] for r in data['origin_predictions.json'])
        inventory['noun_'+arm]=add('noun_'+arm,data,'on','same reused val5000')
    data,src=load('experiments/vg_region_caption/review/loaded_A.json');sources.append(src)
    assert all(branches['A'][r['image_id']]==r['caption'] for r in data['on_predictions.json'])
    pair_presence={r['image_id']:r['valid_pairs'] for rank in (0,1) for r in data[f'on_diagnostic_rank{rank}.json']}
    for arm in ('G','T','GV'):
        data,src=load(f'experiments/vg_region_caption/review/loaded_{arm}.json');sources.append(src)
        inventory['VG_'+arm]=add('VG_'+arm,data,'on','reused dev1000 subset; per-image CIDEr IDF differs from val5000')
    for arm in ('C','T'):
        data,src=load(f'experiments/stale_prompt_pilot/review/{arm}_loaded.json');sources.append(src)
        assert all(branches['A'][r['image_id']]==r['caption'] for r in data['A_predictions.json'])
        inventory['stale_'+arm]=add('stale_'+arm,data,'end','reused dev1000 subset; not rank-comparable CIDEr to val5000')
    dataset_path=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')
    assert sha(dataset_path)=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    dataset=json.loads(dataset_path.read_text(encoding='utf-8'))['images']
    candidates=[]
    for i,r in group.items():
        image=dataset[i];assert image['filename']==r['filename'] and image['split']=='val'
        a=chair['A'][r['coco_id']];refs=[s['raw'] for s in image['sentences']]
        generated=set(a['mscoco_generated_words']);gt=set(a['mscoco_gt_words'])
        row=dict(image_id=i,coco_id=r['coco_id'],filename=r['filename'],filepath='val2014',caption_A=branches['A'][i],references=refs,
            automatic=dict(chair_flagged=bool(a['mscoco_hallucinated_words']),hallucinated_words=a['mscoco_hallucinated_words'],gt_object_classes=sorted(gt),generated_classes=sorted(generated),unmentioned_classes=sorted(gt-generated),A_cider_val5000=scores['A'][i]),
            captions={name:rows[i] for name,rows in branches.items() if i in rows},
            branch_chair_flags={name:bool(rows[r['coco_id']]['mscoco_hallucinated_words']) for name,rows in chair.items() if r['coco_id'] in rows},
            upt_valid_pairs_recorded=pair_presence.get(i),
            scope='Diagnostic candidates, not established errors. References/CHAIR are not human truth; missing mentions may be appropriate.')
        candidates.append(row)
    # Fixed purposeful strata BEFORE seeing images. No score/gain population claim.
    selected=[]
    def take(label,population,count):
        used={r['coco_id'] for r in selected}
        chosen=sorted((r for r in population if r['coco_id'] not in used),key=lambda r:order(r['coco_id']))[:count]
        selected.extend([dict(r,selection_stratum=label) for r in chosen])
    joint=[r for r in candidates if 'VG_G' in r['captions']]
    take('CHAIR-positive with cross-branch coverage',[r for r in joint if r['automatic']['chair_flagged']],10)
    take('Unflagged possible action/relation problem',[r for r in candidates if not r['automatic']['chair_flagged'] and re.search(r'holding.+\band\b',r['caption_A'])],6)
    take('Unflagged low reference-overlap, possible missing content',[r for r in joint if not r['automatic']['chair_flagged'] and r['automatic']['unmentioned_classes'] and r['automatic']['A_cider_val5000']<.5],10)
    take('Unflagged low reference-overlap outside dev1000',[r for r in candidates if not r['automatic']['chair_flagged'] and r['automatic']['A_cider_val5000']<.3 and r['automatic']['unmentioned_classes']],4)
    assert len(selected)==30 and len({r['coco_id'] for r in selected})==30
    save(OUT/'candidate30.json',selected)
    save(OUT/'inventory.json',dict(date='2026-10-27',sources=sources,branches=inventory,dataset_sha256=sha(dataset_path),
        images=5000,diagnostic_candidates=30,selection_policy='hash-first within 10 CHAIR+/6 holding-and/10 unflagged low-CIDEr dev/4 unflagged low-CIDEr nondev strata; inspect before confirming any error',
        local_neural_execution=False,fresh_evaluation=False,novel_method=False))
    print('Saved30 candidates; aligned branches',inventory)
    for n,r in enumerate(selected,1):print(n,r['coco_id'],r['selection_stratum'],'A:',r['caption_A'],'REF:',r['references'][0])

if __name__=='__main__':main()
