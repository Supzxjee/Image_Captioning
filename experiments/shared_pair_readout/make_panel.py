"""Build the locked I/J caption panel from cached exports; no model execution."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'review'
CACHE=ROOT/'experiments/relation_frames_inventory/review/shared_readout_v2'

def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode('utf-8')).hexdigest()

def assemble():
    texts={a:json.loads((CACHE/f'{a}_loaded.json').read_text(encoding='utf-8'))['texts'] for a in ('I','J')}
    meta=json.loads(texts['I']['metadata.json'])
    assert texts['I']['metadata.json']==texts['J']['metadata.json']
    panels={a:json.loads(t['semantic_panel.json']) for a,t in texts.items()}
    assert all(p['preselected'] and not p['human_scores_available'] for p in panels.values())
    byarm={a:{p['pair_id']:p for p in panel['pairs']} for a,panel in panels.items()}
    selected=set(meta['semantic_panel_pair_ids']); assert len(selected)==32
    assert all(set(p)==selected for p in byarm.values())
    manifest=json.loads(gzip.decompress((ROOT/'experiments/relation_frames_inventory/V2_MANIFEST.json.gz').read_bytes()))
    frozen={p['pair_id']:p for p in manifest['heldout']}
    cards=[]; key=[]; downloads=[]
    for pid in sorted(selected):
        p=byarm['I'][pid]; other=byarm['J'][pid]; f=frozen[pid]
        assert p['statement']==other['statement']==f['statement']
        for role in ('positive','negative'):
            r=p[role]; rr=other[role]; origin=f[role]
            assert r['coco_id']==rr['coco_id']==origin['image_id']
            assert r['captions']['A']==rr['captions']['A']
            cid=digest('shared-panel-image-v1:'+pid+':'+str(r['coco_id']))[:24]
            captions={'A':r['captions']['A'],'I':r['captions']['end'],'J':rr['captions']['end']}
            groups={}
            for arm,caption in captions.items():groups.setdefault(caption,[]).append(arm)
            units=[]
            for caption,arms in groups.items():
                uid=digest('shared-panel-caption-v1:'+cid+':'+caption)[:24]
                units.append({'id':uid,'caption':caption})
                key.append({'id':uid,'card_id':cid,'pair_id':pid,'stratum':p['stratum'],'role':role,'coco_id':r['coco_id'],'arms':arms,'caption':caption})
            units.sort(key=lambda u:digest('shared-panel-order-v1:'+u['id']))
            boxes={name:origin['predicted'][name]['bbox_xyxy'] for name in ('subject','object')}
            card={'id':cid,'assertion':p['statement'],'subject':origin['subject'],'object':origin['object'],
                'image':'images/'+cid+'.jpg','width':origin['image']['width'],'height':origin['image']['height'],
                'boxes':boxes,'captions':units}
            cards.append(card)
            downloads.append((card,origin['image']['file_name']))
    cards.sort(key=lambda c:digest('shared-panel-image-order-v1:'+c['id']))
    public={'schema':'shared-readout-blind-panel-v1','cards':cards,'source_captions':192,'unique_ratings':len(key)}
    public['panel_sha256']=digest(json.dumps(public,sort_keys=True,ensure_ascii=False,separators=(',',':')))
    assert len(cards)==64 and len({c['id'] for c in cards})==64
    assert sum(len(k['arms']) for k in key)==192
    assert {k['pair_id'] for k in key}==selected
    assert all(not any(word in c for word in ('role','arms','pair_id','stratum','coco_id')) for c in cards)
    return public,key,downloads

def fetch(item):
    card,filename=item; path=OUT/card['image'];path.parent.mkdir(parents=True,exist_ok=True)
    directory=filename.split('_')[1]
    assert directory in ('train2014','val2014') and filename.endswith('.jpg')
    url=f'https://s3.amazonaws.com/images.cocodataset.org/{directory}/{filename}'
    local=ROOT/'experiments/relation_frames_inventory/review/images'/filename
    raw=path.read_bytes() if path.exists() else (local.read_bytes() if local.exists() else urllib.request.urlopen(url,timeout=45).read())
    with Image.open(io.BytesIO(raw)) as im:
        assert im.format=='JPEG' and im.size==(card['width'],card['height'])
        im.verify()
    if not path.exists():path.write_bytes(raw)
    return {'card_id':card['id'],'url':url,'sha256':digest(raw),'bytes':len(raw)}

def main():
    public,key,downloads=assemble();OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:provenance=list(pool.map(fetch,downloads))
    (OUT/'private_key.json').write_text(json.dumps({'panel_sha256':public['panel_sha256'],'mapping':key},indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (OUT/'panel.json').write_text(json.dumps(public,indent=2,ensure_ascii=False),encoding='utf-8')
    template=(HERE/'panel_template.html').read_text(encoding='utf-8')
    (OUT/'index.html').write_text(template.replace('__PAYLOAD__',json.dumps(public,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    print(json.dumps({'images':len(public['cards']),'source_captions':192,'unique_ratings':public['unique_ratings'],'panel_sha256':public['panel_sha256'],'page':str(OUT/'index.html')},indent=2))

if __name__=='__main__':main()
