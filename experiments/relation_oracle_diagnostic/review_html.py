"""Offline review of subject/object boxes, annotation claims and diagnostic captions."""
import html
import hashlib
import json
from pathlib import Path


def render(cases,output,records=None,image_folder='images'):
    records={r['case_id']:r for r in (records or [])};cards=[]
    for i,row in enumerate(cases,1):
        p=row['positive_index'];true=row['candidates'][p];false=row['candidates'][1-p]
        boxes=[]
        for label,b,color in zip(('S','O'),true['bounds'][:2],('#187547','#e07c00')):
            x,y,x2,y2=b
            boxes.append(f'<rect x="{x*1000}" y="{y*1000}" width="{(x2-x)*1000}" height="{(y2-y)*1000}" fill="none" stroke="{color}" stroke-width="4"/><text x="{x*1000}" y="{max(20,y*1000)}" fill="{color}" font-size="30">{label}</text>')
        predictions=''
        if row['case_id'] in records:
            predictions='<table><tr><th>Điều kiện</th><th>Caption greedy</th><th>logP(annotation+) − logP(annotation−)</th></tr>'+''.join(
                f'<tr><td>{html.escape(k)}</td><td>{html.escape(v["caption"])}</td><td>{v["positive_minus_negative_logp"]:.5f}</td></tr>' for k,v in records[row['case_id']]['conditions'].items())+'</table>'
        cards.append(f'''<article data-id="{html.escape(row['case_id'])}"><h2>{i}. {html.escape(row['filename'])}</h2>
        <div class="image"><img src="{html.escape(image_folder)}/{html.escape(row['filename'])}" alt="Ảnh chưa có local; bản review Kaggle sẽ kèm ảnh"><svg viewBox="0 0 1000 1000" preserveAspectRatio="none">{''.join(boxes)}</svg></div>
        <p><b>Annotation dương:</b> {html.escape(true['text'])}<br><b>Annotation âm:</b> {html.escape(false['text'])}</p>
        <p>F boxed: annotation+ {row['scores']['boxed'][p]:.4f}; annotation− {row['scores']['boxed'][1-p]:.4f}. Nguồn: {html.escape(row['partition'])}.</p>
        <label>Kiểm tra nhãn/box: <select><option value="unreviewed">Chưa xem</option><option value="clear_contrast">Box rõ; đúng/sai phân biệt rõ</option><option value="both_plausible">Cả hai cách nói đều có thể đúng</option><option value="unclear">Ảnh/động tác chưa rõ</option><option value="bad_box_or_label">Box hoặc nhãn đáng ngờ</option></select></label>
        <p><textarea placeholder="Ghi chú: tư thế, tay, tiếp xúc, vật bị che…"></textarea></p>{predictions}</article>''')
    document='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>20 ca sai quan hệ F</title><style>
    body{font:17px Arial;background:#edf2f6;max-width:1000px;margin:30px auto;padding:0 15px}article{background:white;padding:24px;border-radius:12px;margin:22px 0}.image{position:relative;max-width:850px}.image img{display:block;width:100%}.image svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #ccc;padding:8px;text-align:left}select,textarea{font:inherit;max-width:100%}textarea{width:90%;height:60px}button{font:inherit;padding:12px}p{line-height:1.5}</style>
    <h1>CASES_COUNT ca F không khớp annotation quan hệ</h1><p>Chọn sau khi thấy lỗi, không đại diện toàn bộ dữ liệu. S xanh: người; O cam: vật. Nhãn V-COCO không thay thế kiểm tra trực quan: nếu holding/carrying đều hợp lý, ghi “cả hai có thể đúng”. Không tự suy ra caption đúng từ việc trùng từ.</p><button onclick="save()">Tải nhận xét JSON</button>'''+''.join(cards)+'''
    <script>function save(){const annotations={};document.querySelectorAll('article').forEach(a=>annotations[a.dataset.id]={status:a.querySelector('select').value,notes:a.querySelector('textarea').value});const b=new Blob([JSON.stringify({schema_version:1,review_type:'human_self_reported',annotations},null,2)],{type:'application/json'});const u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='oracle_case_review_20.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}</script></html>'''
    cases_sha=hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    document=document.replace("schema_version:1,review_type:",f"schema_version:1,cases_sha256:'{cases_sha}',review_type:")
    Path(output).write_text(document.replace('CASES_COUNT',str(len(cases))),encoding='utf-8')
