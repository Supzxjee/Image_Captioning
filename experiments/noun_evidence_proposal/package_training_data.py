"""Seal small TRAIN-only labels/regions and vocabulary; no torch or model execution."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import zipfile
import numpy as np
from evidence_data import DATA_SHA,encoded,digest,save,labels,lexicon

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    parent=ROOT/'experiments/coco_grounding_feasibility'
    data_path=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')
    assert digest(data_path)==DATA_SHA
    data=json.loads(data_path.read_text(encoding='utf-8'))
    a=json.loads((ROOT/'experiments/vg_region_caption/review/loaded_A.json').read_text(encoding='utf-8'))['json']
    groups=dict(train=a['groups.json']['train'],val=[dict(image_id=i,coco_id=r['cocoid'],filename=r['filename'],split='val')
        for i,r in enumerate(data['images']) if r['split']=='val'])
    groups['val'].sort(key=lambda r:r['image_id'])
    assert len(groups['train'])==10000 and len(groups['val'])==5000
    assert not {r['coco_id'] for r in groups['train']}&{r['coco_id'] for r in groups['val']}
    vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3}
    counter=Counter(w for r in data['images'] if r['split'] in ('train','restval') for s in r['sentences'][:5] for w in s['raw'].lower().split())
    vocab.update({w:i for i,(w,n) in enumerate(counter.most_common(9996),4)})
    objects=runpy.run_path(str(ROOT/'captioning/object_consistency.py'))
    aliases=objects['ALIASES'];classes=list(objects['COCO_LABELS'])
    policy=json.loads((parent/'MAPPING_POLICY_V11.json').read_text(encoding='utf-8'))
    instances_path=parent/'review/full_annotations/train10k_instances.json'
    instances=json.loads(instances_path.read_text(encoding='utf-8'))
    categories={r['id']:r['name'] for r in instances['categories']};present=defaultdict(set)
    for r in instances['annotations']:present[r['image_id']].add(categories[r['category_id']])
    span_dir=parent/'review/noun_box_train10k_v1_1'
    spans=json.loads((span_dir/'spans.json').read_text(encoding='utf-8'));masks=json.loads((span_dir/'mask_bank.json').read_text(encoding='utf-8'))
    unique={}
    for row in spans:
        mask=masks[row['mask_key']];key=(row['coco_id'],mask['annotation_id'],row['category'])
        unique[key]=dict(annotation_id=mask['annotation_id'],class_id=classes.index(row['category']),mask=mask['patch_area_fraction'])
    regions=defaultdict(list)
    for (iid,aid,c),r in sorted(unique.items()):regions[iid].append(r)
    targets={};ys=[];observed=[]
    for row in groups['train']:
        refs=[s['raw'] for s in data['images'][row['image_id']]['sentences'][:5]]
        y,o=labels(present[row['coco_id']],refs,aliases,policy,classes)
        targets[str(row['image_id'])]=dict(coco_id=row['coco_id'],y=y,observed=o,regions=regions[row['coco_id']])
        ys.append(y);observed.append(o)
    y=np.asarray(ys);o=np.asarray(observed);prior=(y.sum(0)+.5)/(o.sum(0)+1)
    assert (y<=o).all() and len(unique)==5602 and int(sum(len(r['regions']) for r in targets.values()))==5602
    lexical=lexicon(vocab,aliases,policy,classes)
    positive_only,_=labels([],['a hot dog on a plate'],aliases,policy,classes)
    assert positive_only[classes.index('hot dog')]==1 and positive_only[classes.index('dog')]==0
    payloads=dict(groups=groups,lexicon=lexical,targets=targets,prior=dict(probability=prior.tolist(),positive=y.sum(0).tolist(),observed=o.sum(0).tolist(),
        smoothing='.5/1; each observed TRAIN image/class once; no class weighting'))
    out=HERE/'review/dataset';out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,value in payloads.items():
        raw=encoded(value);path=out/(name+'.json')
        if path.exists():assert path.read_bytes()==raw,'Preserve sealed data; version changes explicitly.'
        else:path.write_bytes(raw)
        hashes[path.name]=hashlib.sha256(raw).hexdigest()
    receipt=dict(schema='noun_evidence_train10k_v1',payload_sha256=hashes,train_images=10000,val_images=5000,
        region_count=5602,region_images=len(regions),classes=classes,candidate_vocab_entries=len(lexical['candidate_ids']),
        source_dataset_sha256=DATA_SHA,source_instances_sha256=digest(instances_path),
        source_span_sha256=digest(span_dir/'spans.json'),source_mask_sha256=digest(span_dir/'mask_bank.json'),
        source_alias_sha256=digest(ROOT/'captioning/object_consistency.py'),source_policy_sha256=digest(parent/'MAPPING_POLICY_V11.json'),
        secondary_dev_ids=[r['image_id'] for r in a['groups.json']['val']],vocabulary_sha256=lexical['vocabulary_sha256'],
        val_reused=True,test_used=False,weak_negatives_not_certified_absence=True)
    save(out/'receipt.json',receipt)
    archive=HERE/'review/noun_evidence_train10k_v1_dataset.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(out.glob('*.json')):
            info=zipfile.ZipInfo(path.name,date_time=(2026,10,23,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
    summary=dict(status='LABEL_PACKAGE_SEALED_NO_MODEL_RUN',receipt_sha256=digest(out/'receipt.json'),
        archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,files=5,region_count=5602,
        candidate_vocab_entries=len(lexical['candidate_ids']),positive_labels=int(y.sum()),masked_labels=int((1-o).sum()),
        person_regions=sum(r['class_id']==0 for items in regions.values() for r in items))
    save(HERE/'DATA_PACKAGE.json',summary);print(json.dumps(summary,indent=2));print(archive)


if __name__=='__main__':main()
