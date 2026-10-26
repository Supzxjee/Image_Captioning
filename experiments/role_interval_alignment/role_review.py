"""Offline, unfilled derivation audit. Never impersonates human annotations."""
import html
import json
from pathlib import Path
from role_logic import identity,POLICY


def write_review(folder,cases,image_root):
    from PIL import Image,ImageDraw
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    wanted={r['scene']['filename'] for r in cases}
    paths={}
    for p in Path(image_root).rglob('*.jpg'):
        if p.name in wanted:
            if p.name in paths:raise ValueError('Ambiguous image path '+p.name)
            paths[p.name]=p
    if set(paths)!=wanted:raise FileNotFoundError('Missing review images: '+str(sorted(wanted-set(paths))[:5]))
    manifest=dict(schema_version=1,kind='role_interval_derivation_review',cases=cases,
        instructions='Judge parser binding, annotation lookup, interval arithmetic and decision. Not full-caption truth; unknown is not false. Purposive audit, not representative accuracy.')
    fingerprint=identity(manifest);(folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for c in cases:
        scene=c['scene'];iid=c['coco_id'];image=Image.open(paths[scene['filename']]).convert('RGB');image.thumbnail((900,650));draw=ImageDraw.Draw(image)
        labels=[]
        for aid,category,b in scene['instances']:
            if category=='person' or category in {e['category'] for p in c['parsed'].values() for e in p['graph']['entities'].values()}:
                x1,y1,x2,y2=[max(0,min(1,v)) for v in b];bounds=(x1*image.width,y1*image.height,x2*image.width,y2*image.height)
                if bounds[2]>bounds[0] and bounds[3]>bounds[1]:
                    draw.rectangle(bounds,outline='lime' if category=='person' else 'orange',width=2)
                    draw.text((bounds[0],bounds[1]),str(aid),fill='black',stroke_width=1,stroke_fill='white')
                labels.append(f'{aid}: {category}')
        image.save(folder/f'{iid}.jpg',quality=90)
        escaped=lambda obj:html.escape(json.dumps(obj,ensure_ascii=False,indent=2))
        cards.append(f'''<article data-id="{iid}"><h2>COCO {iid}</h2><img src="{iid}.jpg" alt="COCO {iid}">
        <p>{html.escape('; '.join(labels))}</p><p><b>Greedy:</b> {html.escape(c['greedy'])}<br><b>Sampled:</b> {html.escape(c['sample'])}</p>
        <p><b>Δ interval:</b> [{c['comparison']['lower']}, {c['comparison']['upper']}] · <b>aG:</b> {c['comparison']['role_advantage']}</p>
        <details><summary>1. Parser: entities, edges, opaque units + dependency tokens</summary><pre>{escaped(c['parsed'])}</pre></details>
        <details><summary>2. Annotation: positive links and explicit actor/action labels (missing ≠ false)</summary><pre>{escaped(dict(positive_edges=scene['positive_edges'],actor_labels=scene['actor_labels'],actor_inventory_complete=scene['actor_inventory_complete']))}</pre></details>
        <details><summary>3. Consistent assignments, cancelled/added/removed units and interval decision</summary><pre>{escaped(c['comparison'])}</pre></details>
        <label>Kết luận phép suy diễn <select class="decision"><option value="">Chưa chấm</option><option value="correct">Đúng</option><option value="incorrect">Sai</option><option value="unclear">Chưa đủ rõ</option></select></label>
        <p><label><input type="checkbox" class="critical">Có lỗi nghiêm trọng: biến thiếu liên kết/nhãn thành phủ định</label></p>
        <p><label><input type="checkbox" class="parser-critical">Có lỗi nghiêm trọng: parser biến cấu trúc bất định/chưa xử lý thành khẳng định chắc chắn</label></p>
        <textarea placeholder="Ghi lỗi cụ thể: parser, gán actor, nhãn, khoảng hay phép trừ..."></textarea></article>''')
    document='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Role interval — audit 30</title>
    <style>body{font:17px Arial;background:#edf2f7;max-width:1050px;margin:24px auto;padding:0 16px}article{background:white;padding:22px;margin:20px 0;border-radius:12px}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}textarea{width:95%;height:65px}select,input,button{font:inherit}details{margin:12px 0}header{background:white;padding:20px}button{margin:14px}</style>
    <header><h1>Kiểm tra 30 phép suy diễn role/interval</h1><p>Đây không phải chọn caption hay hơn. Với mỗi ảnh, kiểm tra: parser có giữ đúng người dùng chung giữa các hành động; nhãn có được tra đúng; thiếu nhãn có giữ unknown; khoảng và aG có tính đúng không.</p>
    <p>Đúng ở đây chỉ là đúng phép suy diễn theo ontology/annotation. Không xác nhận giới tính, tính từ, toàn bộ câu hay định vị người cụ thể. Khung xanh: person; cam: đối tượng được parser dùng. Số trên box là COCO annotation ID.</p>
    <p>Nhãn actor/action 0 mới có thể là bằng chứng phủ định. Thiếu liên kết hoặc không có box cùng loại phải để unknown. Nếu không đủ nhãn để phủ định mọi cách gán người, không được chấm refuted.</p>
    <p>Mẫu chọn có chủ đích để bắt lỗi, không dùng tỷ lệ 27/30 làm ước lượng độ chính xác trên toàn tập.</p>
    <label>Người kiểm tra <input id="reviewer" placeholder="Họ tên"></label><button id="export">Tải JSON</button><span id="count"></span></header>'''+''.join(cards)+'''
    <script>
    const fingerprint=FINGERPRINT, ids=IDS, key='role-interval-'+fingerprint;
    function payload(){let annotations={};document.querySelectorAll('article').forEach(a=>{annotations[a.dataset.id]={decision:a.querySelector('select').value,missing_link_as_false:a.querySelector('.critical').checked,parser_uncertainty_as_fact:a.querySelector('.parser-critical').checked,notes:a.querySelector('textarea').value}});return {schema_version:1,kind:'role_interval_derivation_review',manifest_sha256:fingerprint,reviewer:document.querySelector('#reviewer').value,reviewer_type:'human_self_reported',annotations}}
    function persist(){const p=payload();localStorage.setItem(key,JSON.stringify(p));document.querySelector('#count').textContent=Object.values(p.annotations).filter(a=>a.decision).length+'/'+ids.length;}
    try{const p=JSON.parse(localStorage.getItem(key));if(p){document.querySelector('#reviewer').value=p.reviewer||'';document.querySelectorAll('article').forEach(a=>{const x=p.annotations[a.dataset.id];if(x){a.querySelector('select').value=x.decision;a.querySelector('.critical').checked=x.missing_link_as_false;a.querySelector('.parser-critical').checked=x.parser_uncertainty_as_fact;a.querySelector('textarea').value=x.notes}})}}catch(e){}
    document.addEventListener('input',persist);document.addEventListener('change',persist);persist();
    document.querySelector('#export').onclick=()=>{const p=payload();if(!p.reviewer.trim()){alert('Điền tên người kiểm tra.');return;}const u=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='role_interval_verified_review.json';a.click();URL.revokeObjectURL(u)};
    </script></html>'''
    document=document.replace('FINGERPRINT',json.dumps(fingerprint)).replace('IDS',json.dumps([c['coco_id'] for c in cases]))
    (folder/'index.html').write_text(document,encoding='utf-8')


def audit(manifest,review):
    ids={str(c['coco_id']) for c in manifest['cases']}
    if len(ids)!=30 or len(manifest['cases'])!=30:raise ValueError('Expected 30 unique audit cases.')
    if review.get('schema_version')!=1 or review.get('kind')!=manifest['kind'] or review.get('manifest_sha256')!=identity(manifest):raise ValueError('Review/manifest binding differs.')
    if review.get('reviewer_type')!='human_self_reported' or not review.get('reviewer','').strip():raise ValueError('Human reviewer provenance missing.')
    annotations=review['annotations']
    if set(annotations)!=ids:raise ValueError('Missing/extra review IDs.')
    if any(a.get('decision') not in ('correct','incorrect','unclear') or any(type(a.get(k)) is not bool for k in ('missing_link_as_false','parser_uncertainty_as_fact')) for a in annotations.values()):raise ValueError('Incomplete/invalid decisions.')
    correct=sum(a['decision']=='correct' for a in annotations.values());critical=sum(a['missing_link_as_false'] for a in annotations.values())
    parser_critical=sum(a['parser_uncertainty_as_fact'] for a in annotations.values())
    return dict(correct=correct,critical_missing_link_false=critical,critical_parser_uncertainty_fact=parser_critical,human_gate_pass=correct>=POLICY['human_correct_required'] and critical==0 and parser_critical==0,
        scope='Derivation stress-test only; numerical density gate and research decision remain separate. No automatic training.')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('review');args=ap.parse_args()
    print(json.dumps(audit(json.loads(Path(args.manifest).read_text(encoding='utf-8')),json.loads(Path(args.review).read_text(encoding='utf-8'))),indent=2))
