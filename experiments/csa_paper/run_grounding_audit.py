"""User-run CPU metadata/box audit; never loads ROI feature arrays or a model."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import time
from urllib.request import urlopen
import zipfile
import numpy as np
from grounding_labels import cohort, lexicon, caption_labels

HERE = Path(__file__).resolve().parent
URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
MEMBERS = {'train':'777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732',
           'val':'e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47'}
DATA_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
ARCHIVE_SHA = '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def annotations_for(wanted, work, previous_archive=None):
    import ijson
    archive=Path(previous_archive) if previous_archive and Path(previous_archive).is_file() else work/'annotations_trainval2014.zip'
    if not archive.is_file():
        print('Downloading official COCO annotations only; no images/features.',flush=True)
        with urlopen(URL,timeout=90) as response,archive.open('wb') as f:
            while block:=response.read(4<<20): f.write(block)
    if sha(archive)!=ARCHIVE_SHA: raise ValueError('Changed official annotation archive')
    print('Official archive verified (existing copy reused when present).',flush=True)
    images={}; annotations=defaultdict(list); categories=None; provenance={}
    with zipfile.ZipFile(archive) as z:
        for split, expected in MEMBERS.items():
            name=f'annotations/instances_{split}2014.json'; h=hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda:f.read(4<<20),b''): h.update(block)
            if h.hexdigest()!=expected: raise ValueError('Official annotation member changed: '+name)
            provenance[name]=expected
            for section in ('categories','images','annotations'):
                with z.open(name) as f:
                    values=ijson.items(f,section+'.item',use_float=True)
                    if section=='categories':
                        current=list(values)
                        if categories is not None and categories!=current: raise ValueError('Category lists differ')
                        categories=current
                    else:
                        for row in values:
                            identity=row['id'] if section=='images' else row['image_id']
                            if identity not in wanted: continue
                            if section=='images':
                                if identity in images: raise ValueError('Duplicate image ID')
                                images[identity]={k:row[k] for k in ('id','file_name','width','height')}
                            else:
                                annotations[identity].append({k:row[k] for k in ('id','category_id','bbox','iscrowd')})
            print('Filtered official',split,'annotations to TRAIN cohort.',flush=True)
    if set(images)!=set(wanted) or len(categories)!=80: raise ValueError('GT image/category coverage differs')
    return images,annotations,categories,dict(url=URL,members=provenance,archive_sha256=sha(archive))


def sample(rows, seed, n=100):
    return sorted(rows,key=lambda r:hashlib.sha256((seed+'/'+r['mention_id']).encode()).hexdigest())[:n]


def review_html(rows):
    data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><meta charset="utf-8"><title>CSA word–region audit</title>
<style>body{font:16px sans-serif;background:#eef3f6;margin:24px}article{background:white;padding:20px;margin:20px 0}canvas{max-width:100%;height:auto}select{margin:8px}p{max-width:1000px}</style>
<h1>Kiểm nhãn từ–vùng CSA: chưa huấn luyện</h1>
<p>Xanh lá: tất cả GT cùng lớp; cam: proposal dương; xám: proposal khác. Kiểm nghĩa từ trong câu và tập vùng dương. Đây là nhãn theo lớp, không chọn một instance cụ thể. Ảnh không tải được: chọn Không xem được. Giữ nguyên các nhãn chưa chấm.</p>
<label>Người kiểm tra <input id="reviewer"></label><button id="export">Xuất JSON</button><div id="cases"></div>
<script>const rows=DATA;const key='csa-grounding-review/'+rows.map(r=>r.mention_id).join('|');const answers=JSON.parse(localStorage.getItem(key)||'{}');const root=document.getElementById('cases');
for(const [i,r] of rows.entries()){const card=document.createElement('article');root.append(card);
const title=document.createElement('h2');title.textContent=`${i+1}. ${r.category} — ${r.phrase} (${r.status})`;card.append(title);
const p=document.createElement('p');p.textContent=r.caption;card.append(p);const detail=document.createElement('p');detail.textContent=`ID ${r.coco_id}; token target ${r.target_position}; positives [${r.positive_regions}]`;card.append(detail);
const c=document.createElement('canvas');c.width=r.width;c.height=r.height;card.append(c);const ctx=c.getContext('2d');const im=new Image();
im.onload=()=>{ctx.drawImage(im,0,0,c.width,c.height);for(let j=0;j<r.proposal_boxes.length;j++){const b=r.proposal_boxes[j];ctx.strokeStyle=r.positive_regions.includes(j)?'#ee7900':'#aaa';ctx.lineWidth=r.positive_regions.includes(j)?3:1;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(j,b[0]+2,b[1]+12)}for(const b of r.gt_boxes){ctx.strokeStyle='#00a050';ctx.lineWidth=3;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}};
im.onerror=()=>{detail.textContent+=' — Không tải được ảnh';};im.src=r.image_url;
const s=document.createElement('select');for(const [v,t] of [['','Chưa chấm'],['confirmed','Mapping và tập dương hợp lý'],['incorrect','Cần sửa nhãn/tập dương'],['uncertain','Chưa rõ'],['unreviewable','Không xem được ảnh']]){const o=document.createElement('option');o.value=v;o.textContent=t;s.append(o)}card.append(s);const note=document.createElement('input');note.placeholder='Ghi chú';card.append(note);if(answers[r.mention_id]){s.value=answers[r.mention_id].status;note.value=answers[r.mention_id].notes}const update=()=>{answers[r.mention_id]={status:s.value,notes:note.value};localStorage.setItem(key,JSON.stringify(answers))};s.onchange=update;note.oninput=update;}
document.getElementById('export').onclick=()=>{const payload={schema:'csa_grounding_audit_human_v1',reviewer:document.getElementById('reviewer').value,annotations:answers,case_ids:rows.map(r=>r.mention_id)};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='csa_grounding_audit_review.json';a.click()};</script>'''.replace('DATA',data)


def run(cfg):
    start=time.monotonic(); out=Path(cfg['output']); work=Path(cfg['work']);work.mkdir(parents=True,exist_ok=True)
    expected=read(HERE/'SAVE_FULL_ROI_V134_PACKAGE.json')['files']
    roots=[p.parent for p in Path(cfg['full_root']).rglob('receipt.json') if sha(p)==expected['receipt.json']]
    if len(roots)!=1: raise ValueError('Expected one exact v134 ROI receipt; attach csa-full-roi-karpathy-v134')
    root=roots[0]
    for name in ('manifest.json','vocabulary.json'):
        if sha(root/name)!=expected[name]: raise ValueError('Changed v134 '+name)
    if sha(cfg['dataset_json'])!=DATA_SHA: raise ValueError('Changed Karpathy dataset JSON')
    image_count=cfg.get('image_count',10000)
    if image_count not in (10000,30000):raise ValueError('Only approved10k/30k cohorts supported')
    dataset=read(cfg['dataset_json'])
    selected=cohort(dataset['images'],count=30000) if image_count==30000 else cohort(dataset['images'])
    wanted={int(r['cocoid']):r for r in selected}
    if image_count==30000:
        prefix=[dict(coco_id=int(r['cocoid']),filename=r['filename'],split=r['split']) for r in selected[:10000]]
        prefix_sha=hashlib.sha256((json.dumps(prefix,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
        if prefix_sha!='1faedfb6d48c1d66aecffab139195bc154cf357af73d0f695b36af7d5535c834':raise ValueError('30k cohort must extend exact original10k prefix')
    save(out/'cohort.json',[dict(coco_id=i,filename=r['filename'],split=r['split']) for i,r in wanted.items()])
    images,anns,categories,provenance=annotations_for(wanted,work,cfg.get('annotation_archive'))
    save(out/'annotation_provenance.json',provenance); lookup=lexicon(categories);save(out/'lexicon.json',lookup)
    instances_name='train10k_instances.json' if image_count==10000 else 'train_instances.json'
    save(out/instances_name,dict(images=list(images.values()),annotations=dict(anns),categories=categories))
    vocab=read(root/'vocabulary.json'); word2id={w:i for i,w in enumerate(vocab)}
    mapped=[];seen=set(); payloads={}; counts=Counter();per_class=defaultdict(Counter);catnames={c['id']:c['name'] for c in categories}
    for entry in read(root/'manifest.json')['shards']:
        if entry['group']!='train': continue
        meta_name=next(n for n in entry['files'] if n.endswith('.json'))
        if sha(root/meta_name)!=expected[meta_name]: raise ValueError('Changed shard metadata')
        metadata=read(root/meta_name)['images']
        if not any(r['coco_id'] in wanted for r in metadata):continue
        npz_name=next(n for n in entry['files'] if n.endswith('.npz'))
        if sha(root/npz_name)!=expected[npz_name]:raise ValueError('Changed ROI shard')
        payloads.update({n:expected[n] for n in (meta_name,npz_name)})
        with np.load(root/npz_name,allow_pickle=False) as z:
            offsets=z['offsets'];ids=z['ids'];boxes=z['boxes'] # features deliberately never accessed
        if ids.tolist()!=[r['coco_id'] for r in metadata] or len(offsets)!=len(ids)+1 or offsets[0]!=0 or offsets[-1]!=len(boxes) or (np.diff(offsets)<=0).any():raise ValueError('ROI row/offset alignment differs')
        for j,m in enumerate(metadata):
            identity=m['coco_id']
            if identity not in wanted:continue
            if identity in seen:raise ValueError('Repeated TRAIN image')
            seen.add(identity); original=wanted[identity]; gt=images[identity]
            if m['source_split']!=original['split'] or m['filename']!=gt['file_name'] or (m['width'],m['height'])!=(gt['width'],gt['height']):raise ValueError('Image identity/dimensions mismatch')
            b=boxes[offsets[j]:offsets[j+1]]
            if (b<0).any() or (b[:,[0,2]]>m['width']).any() or (b[:,[1,3]]>m['height']).any():raise ValueError('Out-of-bound cached boxes')
            if len(original['sentences'])!=len(m['sequences']):raise ValueError('Caption counts differ')
            for ci,s in enumerate(original['sentences']):
                tokens=s['tokens']; content=[word2id.get(unicodedata_token(t),3) for t in tokens[:52]]
                seq=[1]+content+[2]+[0]*(52-len(content))
                if seq!=m['sequences'][ci] or s['raw']!=m['captions'][ci]:raise ValueError('Caption/token provenance differs')
                counts['captions']+=1;counts['content_tokens']+=len(tokens)
                labels=list(caption_labels(tokens,seq,vocab,b,m['region_kinds'],anns[identity],lookup))
                counts['captions_with_mapped_mention']+=bool(labels)
                for label in labels:
                    label.update(coco_id=identity,caption_index=ci,caption=s['raw'],category=catnames[label['category_id']],
                        mention_id=f'{identity}:{ci}:{label["start"]}:{label["end"]}',width=m['width'],height=m['height'],
                        proposal_boxes=b.tolist(),image_url='https://images.cocodataset.org/'+('train2014/' if 'train2014' in m['filename'] else 'val2014/')+m['filename'])
                    mapped.append(label);counts[label['status']]+=1;per_class[label['category']][label['status']]+=1
        print('Audited ROI boxes:',len(seen),'/',image_count,flush=True)
    if seen!=set(wanted):raise ValueError('TRAIN coverage incomplete')
    with (out/'mentions.jsonl').open('w',encoding='utf-8') as f:
        for r in mapped:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    # General sample includes skipped cases. Eligible sample evaluates positive-set precision separately.
    general=sample(mapped,'general'); eligible=sample([r for r in mapped if r['status']=='eligible'],'eligible')
    for name,rows in [('review_100',general),('review_eligible_100',eligible)]:
        save(out/(name+'.json'),rows);(out/(name+'.html')).write_text(review_html(rows),encoding='utf-8')
    denom=counts['eligible']+counts['no_positive']
    summary=dict(status='CSA_GROUNDING_DATA_AUDIT_COMPLETE',images=len(seen),mapped_mentions=len(mapped),counts=dict(counts),
        per_class={k:dict(v) for k,v in sorted(per_class.items())},positive_coverage_denominator=denom,
        positive_coverage=counts['eligible']/denom if denom else None,
        coverage_gate_pass=bool(denom and counts['eligible']/denom>=.8),manual_review_pending=True,
        invalid_gt_policy='Skip entire category mention if any same-category noncrowd GT has nonpositive width/height; retain XE',
        training_ready=False,model_executed=False,test_used=False,seconds=time.monotonic()-start,
        checked_roi_payloads=payloads,lexicon_sha256=sha(out/'lexicon.json'),cohort_sha256=sha(out/'cohort.json'))
    if image_count==30000:
        summary.update(original10k_prefix_sha256=prefix_sha,instances_file=instances_name,
            planned_updates=2000,planned_caption_events=64000,sampler='round_robin_one_caption_per_image_seed42',sampler_decision_pending=False)
    save(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_class','checked_roi_payloads')},indent=2),flush=True)


def unicodedata_token(t):
    import unicodedata
    return unicodedata.normalize('NFKC',t).lower()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(read(p.parse_args().config))
