"""Audit received archives and prepare a blank, blinded phrase review; no inference."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive entries'
        data = {n: json.loads(archive.read(n)) for n in (
            'protocol.json', 'summary.json', 'locked_manifest.json',
            'before_records.json', 'after_records.json', 'shared_receipt.json',
            'complete_rank0.json', 'complete_rank1.json')}
        p, s = data['protocol.json'], data['summary.json']
        assert digest(archive.read('locked_manifest.json')) == p['manifest_sha256']
        for name, sha in p['source_sha256'].items():
            assert digest(archive.read('source/' + name)) == sha, name
        for rank in range(2):
            r = data[f'complete_rank{rank}.json']
            assert r['updates'] == s['updates'] == 300
            assert r['original_parameters_unchanged'] is True
            assert r['initial_sha256'] == s['initial_sha256']
            assert r['final_sha256'] == s['final_sha256']
            assert r['protocol_sha256'] == p['protocol_sha256']
        groups = {g['group_id'] for g in data['locked_manifest.json']['groups']}
        for phase in ('before', 'after'):
            rows = data[f'{phase}_records.json']
            assert len(rows) == len(groups) == 29
            assert {r['group_id'] for r in rows} == groups
        return data


def audit(full, crop):
    a, b = full['protocol.json'], crop['protocol.json']
    assert a['variant'] == 'FULLFRAME' and b['variant'] == 'PAIRCROP'
    keys = ('manifest_sha256', 'source_sha256', 'stage_sha256', 'shared_receipt_sha256',
            'shared_receipt_file_sha256', 'checkpoint_sha256', 'dataset_sha256',
            'base_sha256', 'vocabulary_sha256', 'optimizer', 'loss', 'runtime')
    for key in keys:
        assert a[key] == b[key], key
    assert full['shared_receipt.json'] == crop['shared_receipt.json']
    initial = full['summary.json']['initial_sha256']
    assert initial == crop['summary.json']['initial_sha256'] == b['fullframe_initial_sha256']
    frozen = full['complete_rank0.json']['frozen_sha256']
    for data in (full, crop):
        for rank in range(2):
            r = data[f'complete_rank{rank}.json']
            assert r['frozen_sha256'] == frozen
            assert r['trainable_parameters'] == 848384
    f_before = {r['group_id']: r for r in full['before_records.json']}
    for r in crop['before_records.json']:
        assert r['conditions']['off'] == f_before[r['group_id']]['conditions']['off']
    return list(keys) + ['initial parameter hash', 'frozen parameter hash',
                         'trainable parameter count', 'before/off records']


def prepare(full_path, crop_path):
    full, crop = load(full_path), load(crop_path)
    checks = audit(full, crop)
    manifest = full['locked_manifest.json']
    lookups = {arm: {r['group_id']: r for r in data['after_records.json']}
               for arm, data in [('FULLFRAME', full), ('PAIRCROP', crop)]}
    result = dict(status='MATCHED_TECHNICAL_PASS_SEMANTIC_REVIEW_PENDING',
                  archive_sha256={arm: digest(path.read_bytes()) for arm, path in
                                  [('FULLFRAME', full_path), ('PAIRCROP', crop_path)]},
                  matched_checks=checks, summaries={arm: data['summary.json'] for arm, data in
                                                   [('FULLFRAME', full), ('PAIRCROP', crop)]},
                  limits=['Teacher-forced ranking is not free-generation semantic accuracy.',
                          'No new full-caption or CIDEr result.',
                          'Shared cache and adapter bytes are not included in review archives.'])
    tasks, mapping = [], {}
    for group in manifest['groups']:
        gid = group['group_id']
        if gid not in manifest['partitions']['new_probe']:
            continue
        for index, event in enumerate(group['events']):
            order = ['FULLFRAME', 'PAIRCROP']
            if int(digest((gid + str(index)).encode())[-1], 16) % 2:
                order.reverse()
            task = dict(task_id=digest((gid + ':' + str(index)).encode())[:16], event=event,
                        candidates=[])
            for arm in order:
                r = lookups[arm][gid]['conditions']['on']
                candidate_id = digest((gid + ':' + str(index) + ':' + arm).encode())[:16]
                task['candidates'].append(dict(candidate_id=candidate_id,
                    caption=r['captions'][index], truncated=r['truncated'][index]))
                mapping[candidate_id] = dict(arm=arm, group_id=gid, event_index=index,
                                            family=group['family'])
            tasks.append(task)
    tasks.sort(key=lambda t: t['task_id'])
    assert len(tasks) == 16 and len(mapping) == 32
    payload = dict(schema_version=1, tasks=tasks)
    review_sha = digest(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
    payload['review_sha256'] = review_sha
    result['review_sha256'] = review_sha
    out = HERE / 'review' / 'pair_source_blind'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    for task in tasks:
        name = task['event']['filename']
        assert Path(name).name == name
        shutil.copyfile(HERE / 'review' / 'images' / name, out / 'images' / name)
    (HERE / 'PAIR_SOURCE_COMPARISON.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Keep identities outside the page so the human reviewer sees only X/Y.
    (out.parent / 'pair_source_blind_key.json').write_text(json.dumps(dict(review_sha256=review_sha, mapping=mapping), indent=2), encoding='utf-8')
    (out / 'index.html').write_text(PAGE.replace('PAYLOAD_JSON', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')), encoding='utf-8')
    print('Matched technical audit PASS; blank review: review/pair_source_blind/index.html')


PAGE = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chấm cụm mô tả — ẩn tên nhánh</title><style>
body{font:17px/1.5 Arial;margin:24px auto;padding:0 20px;max-width:1100px;background:#eef3f7;color:#152b40}
section{background:white;border-radius:12px;padding:22px;margin:20px 0}.pic{position:relative;max-width:640px}.pic img{display:block;width:100%}.pic svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.answers{display:grid;grid-template-columns:1fr 1fr;gap:22px}select,textarea,input,button{font:inherit;max-width:100%;padding:8px}select{width:100%}textarea{width:90%}button{cursor:pointer}header{background:#eef3f7;position:sticky;top:0;z-index:1;padding:12px 0}small{display:block}@media(max-width:700px){.answers{grid-template-columns:1fr}}
</style><h1>Chấm 16 cặp người–vật, 32 cụm mô tả</h1>
<p>Chỉ xét <b>người S xanh và vật O cam</b>. Mỗi ảnh có hai câu X/Y, ẩn tên nhánh. Chấm độc lập, không bắt buộc chọn câu thắng. “Đúng, đủ cụ thể” cần mô tả đúng quan hệ của cặp đã chọn và không thêm chi tiết sai. “On a board” có thể quá chung để phân biệt đứng trên với mang ván. Nếu không thấy rõ, chọn “Chưa rõ”. Câu giống nhau nên được chấm giống nhau.</p>
<header><label>Người chấm <input id="reviewer" autocomplete="name"></label> <button id="export">Xuất JSON</button><p id="progress"></p><p id="notice" role="status"></p></header>
<main id="cases"></main><script id="data" type="application/json">PAYLOAD_JSON</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),key='pair-source-blind-'+data.review_sha256;
const allowed=['correct_specific','generic','wrong_relation_or_fact','wrong_instance','unclear'];
let state={reviewer:'',annotations:{}};
try{const saved=JSON.parse(localStorage.getItem(key));if(saved&&typeof saved.reviewer==='string'&&saved.annotations)state=saved;}catch(e){}
const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
const progress=document.getElementById('progress'),notice=document.getElementById('notice');
const ids=data.tasks.flatMap(t=>t.candidates.map(c=>c.candidate_id));
function save(){localStorage.setItem(key,JSON.stringify(state));progress.textContent='Đã chấm '+ids.filter(id=>allowed.includes(state.annotations[id]?.status)).length+'/32 câu';}
const reviewer=document.getElementById('reviewer');reviewer.value=state.reviewer;reviewer.oninput=()=>{state.reviewer=reviewer.value;save();};
data.tasks.forEach((task,i)=>{
 const section=el('section'),event=task.event;section.append(el('h2','Cặp '+(i+1)+' / 16 · COCO '+event.image_id));
 const picture=el('div');picture.className='pic';const img=el('img');img.src='images/'+event.filename;img.alt='Ảnh có cặp S/O được chọn';picture.append(img);
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 100 100');svg.setAttribute('preserveAspectRatio','none');
 [event.subject_box,event.object_box].forEach((b,j)=>{const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');Object.entries({x:b[0]*100,y:b[1]*100,width:(b[2]-b[0])*100,height:(b[3]-b[1])*100,fill:'none',stroke:j?'#ff8c00':'#00cf68','stroke-width':.65}).forEach(([k,v])=>rect.setAttribute(k,v));svg.append(rect);const label=document.createElementNS('http://www.w3.org/2000/svg','text');Object.entries({x:b[0]*100+1,y:Math.max(4,b[1]*100+4),fill:j?'#ff8c00':'#00cf68','font-size':4}).forEach(([k,v])=>label.setAttribute(k,v));label.textContent=j?'O':'S';svg.append(label);});
 picture.append(svg);section.append(picture);const link=el('a','Mở ảnh gốc');link.href=img.src;link.target='_blank';section.append(link);
 const answers=el('div');answers.className='answers';
 task.candidates.forEach((c,j)=>{const block=el('div');block.append(el('h3',j?'Câu Y':'Câu X'),el('p',c.caption));if(c.truncated)block.append(el('strong','Câu bị cắt — không tính là thành công đầy đủ.'));
 const a=state.annotations[c.candidate_id]||{status:'',notes:''};state.annotations[c.candidate_id]=a;
 const label=el('label','Đánh giá '+(j?'Y':'X')),select=el('select');[['','Chưa chấm'],['correct_specific','Đúng, đủ cụ thể'],['generic','Quá chung / thiếu quan hệ cần phân biệt'],['wrong_relation_or_fact','Sai quan hệ hoặc thêm chi tiết sai'],['wrong_instance','Mô tả nhầm người / vật'],['unclear','Chưa rõ từ ảnh']].forEach(([v,t])=>{const o=el('option',t);o.value=v;select.append(o);});select.value=a.status;select.onchange=()=>{a.status=select.value;save();};label.append(select);block.append(label);
 const note=el('textarea');note.rows=2;note.placeholder='Ghi chú (nếu cần)';note.value=a.notes||'';note.setAttribute('aria-label','Ghi chú '+(j?'Y':'X'));note.oninput=()=>{a.notes=note.value;save();};block.append(note);answers.append(block);});
 section.append(answers);document.getElementById('cases').append(section);
});
document.getElementById('export').onclick=()=>{
 if(!state.reviewer.trim()){notice.textContent='Nhập tên người chấm trước khi xuất.';return;}
 const annotations=Object.fromEntries(ids.filter(id=>allowed.includes(state.annotations[id]?.status)).map(id=>[id,state.annotations[id]]));
 if(!Object.keys(annotations).length){notice.textContent='Chưa có câu nào được chấm.';return;}
 const result={schema_version:1,review_sha256:data.review_sha256,reviewer:state.reviewer.trim(),reviewer_type:'human_self_reported',complete:Object.keys(annotations).length===32,annotations};
 const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download='pair_source_blind_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice.textContent='Đã xuất '+Object.keys(annotations).length+'/32 câu. Gửi lại JSON để tổng hợp; không đổi cấu hình mô hình.';
};save();
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fullframe', type=Path, required=True)
    parser.add_argument('--paircrop', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.fullframe, args.paircrop)
