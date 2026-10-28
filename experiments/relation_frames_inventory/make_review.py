"""Fixed32-record visual audit; download original JPEGs only, never run models."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import textwrap
import urllib.request

from PIL import Image,ImageDraw,ImageFont
from inventory import HERE,sha
from select_v2 import stratum

ROOT=HERE/'review'
MANIFEST_SHA='6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7'


def sample(manifest):
    pairs=[]
    for split in ('train','heldout'):
        for group in ('projective_containment','contact_proximity'):
            candidates=[p for p in manifest[split] if stratum(p['relation'])==group]
            candidates.sort(key=lambda p:sha(('vsr-visual-audit-v1:'+p['pair_id']).encode()))
            if len(candidates)<4:raise ValueError('Too few pairs for fixed audit')
            pairs.extend(dict(p,split=split,stratum=group) for p in candidates[:4])
    assert len(pairs)==16 and len({p['pair_id'] for p in pairs})==16
    return pairs


def fetch(row):
    filename=row['image']['file_name'];directory=filename.split('_')[1]
    if directory not in ('train2014','val2014') or not filename.endswith(f"{row['image_id']:012d}.jpg"):
        raise ValueError('Unexpected COCO filename')
    url=f'https://s3.amazonaws.com/images.cocodataset.org/{directory}/{filename}'
    path=ROOT/'images'/filename;path.parent.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=45).read()
    with Image.open(io.BytesIO(raw)) as im:
        if im.format!='JPEG' or im.size!=(row['image']['width'],row['image']['height']):raise ValueError('JPEG identity dimensions differ')
        im.verify()
    if not path.exists():path.write_bytes(raw)
    return dict(image_id=row['image_id'],file_name=filename,url=url,bytes=len(raw),sha256=sha(raw),
                scope='Official COCO JPEG, dimensions verified; no RGB hash in pre-existing detector cache')


def font(size):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def make_card(p,role,index):
    r=p[role];im=Image.open(ROOT/'images'/r['image']['file_name']).convert('RGB')
    original=im.size;im.thumbnail((680,385));draw=ImageDraw.Draw(im)
    for name,color,letter in (('subject','#00c95e','S'),('object','#ff8b00','O')):
        box=r['predicted'][name]['bbox_xyxy'];xy=[box[0]*im.width/original[0],box[1]*im.height/original[1],box[2]*im.width/original[0],box[3]*im.height/original[1]]
        draw.rectangle(xy,outline=color,width=3)
        draw.text((xy[0]+2,max(0,xy[1]-21)),letter,font=font(20),fill=color,stroke_width=1,stroke_fill='black')
    canvas=Image.new('RGB',(720,505),'white');d=ImageDraw.Draw(canvas)
    d.text((12,7),f"#{index:02d} {p['split']} | {role} | COCO {r['image_id']}",font=font(18),fill='black')
    for k,line in enumerate(textwrap.wrap(p['statement'],65)):
        d.text((12,31+22*k),line,font=font(19),fill='black')
    canvas.paste(im,((720-im.width)//2,85))
    expected='TRUE' if r['label'] else 'FALSE'
    d.text((12,477),f"VSR label: {expected} | S={r['subject']} | O={r['object']}",font=font(17),fill='black')
    name=f'card_{index:02d}.jpg';canvas.save(ROOT/name,quality=94)
    return canvas,name


HTML='''<!doctype html><meta charset="utf-8"><title>VSR frame audit32</title>
<style>body{font:17px Arial;background:#eef2f6;max-width:1500px;margin:24px auto;padding:0 16px}header,article{background:white;border-radius:10px;padding:18px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%}label{display:block;margin:12px 0}select,textarea,input{font:inherit}select,textarea{display:block;width:100%;padding:7px;box-sizing:border-box}button{font:inherit;padding:10px}small{color:#444}.warn{color:#9d3b00}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<header><h1>Kiểm tra 32 bản ghi quan hệ VSR</h1><p>S xanh: chủ thể. O cam: đối tượng. Đánh giá <b>đúng hai instance trong khung</b> và phát biểu bên dưới. TRUE/FALSE là nhãn nguồn để kiểm tra, không phải đáp án bắt buộc.</p><p>Không suy ra quan hệ chỉ từ tâm bbox. Nếu ảnh không đủ rõ hoặc cách hiểu quan hệ mơ hồ, chọn Không rõ và ghi lý do. Không thay nhãn hoặc loại ảnh tự động.</p><p class="warn" id="notice"></p><label>Tên người kiểm tra<input id="reviewer"></label><button id="export">Xuất JSON</button><span id="progress"></span></header><main id="root"></main>
<script id="audit-data" type="application/json">PAYLOAD</script>
<script id="draft-data" type="application/json">null</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('audit-data').textContent),draft=JSON.parse(document.getElementById('draft-data').textContent);
const storageKey='vsr-audit32-'+data.audit_sha256+(draft?'-assistant':'-blank');
let state={reviewer:'',annotations:draft?structuredClone(draft.annotations):{}};
try{const saved=JSON.parse(localStorage.getItem(storageKey));if(saved&&typeof saved.annotations==='object')state=saved;}catch{}
const fields={subject_box:{clear:'Đúng instance / box rõ',incorrect:'Sai instance hoặc box',uncertain:'Không rõ'},object_box:{clear:'Đúng instance / box rõ',incorrect:'Sai instance hoặc box',uncertain:'Không rõ'},assertion:{supported:'Phát biểu đúng',contradicted:'Phát biểu sai',uncertain:'Không rõ'}};
function complete(a){return Object.entries(fields).every(([k,opts])=>Object.hasOwn(opts,a[k]))&&a.confirmed_by_human===true;}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(state));}catch{}document.getElementById('progress').textContent=' Đã tự xác nhận '+Object.values(state.annotations).filter(complete).length+'/32';}
function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;}
document.getElementById('notice').textContent=draft?'Có gợi ý sơ bộ của Codex. Chưa phải nhãn người; bạn cần tự kiểm tra rồi xác nhận từng bản ghi.':'Bản trống để tự đánh giá. Chưa có nhãn gợi ý của Codex.';
const reviewer=document.getElementById('reviewer');reviewer.value=state.reviewer;reviewer.oninput=()=>{state.reviewer=reviewer.value;save();};
for(const p of data.pairs){const article=el('article');article.append(el('h2',p.statement),el('small',p.split+' · '+p.stratum));const grid=el('div');grid.className='pair';for(const role of ['positive','negative']){const r=p[role],key=r.review_id,a=state.annotations[key]??={subject_box:'',object_box:'',assertion:'',notes:'',confirmed_by_human:false};const card=el('section');card.append(el('h3','#'+r.audit_index+' · nhãn VSR: '+(r.label?'TRUE':'FALSE')));const img=el('img');img.src=r.card;img.alt=p.statement;card.append(img);const details=el('details');details.append(el('summary','Xem ảnh gốc'));const original=el('img');original.src='images/'+r.image.file_name;original.loading='lazy';details.append(original);card.append(details);const checkbox=el('input');checkbox.type='checkbox';checkbox.checked=a.confirmed_by_human===true;
for(const [field,options] of Object.entries(fields)){const label=el('label',({subject_box:'Box S',object_box:'Box O',assertion:'Quan hệ trong phát biểu'})[field]);const select=el('select');select.append(new Option('Chưa đánh giá',''));for(const [value,text] of Object.entries(options))select.append(new Option(text,value));select.value=a[field]||'';select.onchange=()=>{a[field]=select.value;a.confirmed_by_human=false;checkbox.checked=false;save();};label.append(select);card.append(label);}
const notes=el('textarea');notes.rows=3;notes.placeholder='Ghi chú về vùng ảnh hoặc cách hiểu quan hệ';notes.value=a.notes||'';notes.oninput=()=>{a.notes=notes.value;a.confirmed_by_human=false;checkbox.checked=false;save();};card.append(notes);const confirmation=el('label');confirmation.append(checkbox,document.createTextNode(' Tôi đã tự xem ảnh và xác nhận các ô trên.'));checkbox.onchange=()=>{a.confirmed_by_human=checkbox.checked;save();};card.append(confirmation);grid.append(card);}article.append(grid);document.getElementById('root').append(article);}
document.getElementById('export').onclick=()=>{const annotations=Object.fromEntries(data.pairs.flatMap(p=>['positive','negative'].map(role=>{const r=p[role];return [r.review_id,state.annotations[r.review_id]];})));const verified=Object.values(annotations).every(complete)&&state.reviewer.trim().length>0;const payload={schema_version:1,manifest_sha256:data.manifest_sha256,audit_sha256:data.audit_sha256,reviewer:state.reviewer,reviewer_type:verified?'human_self_reported':(draft?'assistant_proposed_pending_human_verification':'human_incomplete'),training_approved:false,annotations};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='vsr_frame_audit32_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};save();
</script>'''


def main():
    raw=(HERE/'V2_MANIFEST.json.gz').read_bytes()
    if sha(raw)!=MANIFEST_SHA:raise ValueError('Wrong frozen manifest')
    manifest=json.loads(gzip.decompress(raw));pairs=sample(manifest)
    selection=sha(json.dumps([(p['split'],p['pair_id']) for p in pairs],separators=(',',':')).encode())
    ROOT.mkdir(exist_ok=True)
    rows={p[k]['image_id']:p[k] for p in pairs for k in ('positive','negative')}
    with ThreadPoolExecutor(max_workers=4) as pool:provenance=list(pool.map(fetch,rows.values()))
    cards=[]
    for pair_index,p in enumerate(pairs):
        for role_index,role in enumerate(('positive','negative')):
            index=2*pair_index+role_index+1
            r=p[role];r.update(review_id=sha((p['pair_id']+':'+role).encode())[:24],audit_index=index)
            card,name=make_card(p,role,index);r['card']=name;cards.append(card)
    for page in range(8):
        sheet=Image.new('RGB',(1440,1010),'#dddddd')
        for offset,card in enumerate(cards[4*page:4*page+4]):sheet.paste(card,((offset%2)*720,(offset//2)*505))
        sheet.save(ROOT/f'contact_{page+1:02d}.jpg',quality=95)
    data=dict(schema='vsr-frame-audit32-v1',manifest_sha256=MANIFEST_SHA,audit_sha256=selection,
              sample_policy='4 hash-first pairs per split x stratum; 16 pairs/32 records; never score-selected',pairs=pairs)
    (ROOT/'audit_manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    (ROOT/'image_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    page=HTML.replace('PAYLOAD',payload)
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    proposal=HERE/'ASSISTANT_AUDIT32.json'
    if proposal.is_file():
        draft=json.loads(proposal.read_text(encoding='utf-8'))
        ids={p[k]['review_id'] for p in pairs for k in ('positive','negative')}
        if draft['audit_sha256']!=selection or draft['manifest_sha256']!=MANIFEST_SHA or draft['reviewer_type']!='assistant_proposed_not_human' or set(draft['annotations'])!=ids or any(a['confirmed_by_human'] for a in draft['annotations'].values()):
            raise ValueError('Assistant proposal provenance differs')
        embedded=json.dumps(draft,ensure_ascii=False).replace('<','\\u003c')
        assistant=page.replace('<script id="draft-data" type="application/json">null</script>',
            '<script id="draft-data" type="application/json">'+embedded+'</script>')
        (ROOT/'assistant_review.html').write_text(assistant,encoding='utf-8')
    print(json.dumps(dict(records=32,pairs=16,images=len(rows),audit_sha256=selection,page=str(ROOT/'index.html')),indent=2))


if __name__=='__main__':main()
