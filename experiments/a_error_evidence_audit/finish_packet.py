"""Build a small visual evidence atlas from saved captions and explicit notes."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent;OUT=HERE/'review'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    candidates=load(OUT/'candidate30.json');notes=load(HERE/'visual_notes.json');inventory=load(OUT/'inventory.json')
    assert {str(r['coco_id']) for r in candidates}==set(notes['cases']) and not notes['human_verified'] and not notes['training_labels']
    receipts={r['coco_id']:r for r in load(OUT/'image_receipts.json')}
    cases=[]
    for r in candidates:
        p=OUT/'images'/r['filename'];assert hashlib.sha256(p.read_bytes()).hexdigest()==receipts[r['coco_id']]['sha256']
        with Image.open(p) as raw:
            size=raw.size;raw.convert('RGB').resize((224,224),Image.Resampling.BILINEAR).save(OUT/'images'/('preview224_'+r['filename']),quality=93)
        cases.append(dict(r,review=notes['cases'][str(r['coco_id'])],image=dict(receipts[r['coco_id']],original_size=list(size),preview224='PIL bilinear visual aid only; not exact cached model input or evidence the model recognizes a cue.')))
    focus=[r for r in cases if r['review']['focus']];controls=[r for r in cases if not r['review']['focus']]
    assert len(focus)==20 and len(controls)==10
    summary=dict(date='2026-10-27',reviewer_type=notes['reviewer_type'],human_verified=False,training_labels=False,
        examined_images=30,focus_cases=20,controls_or_uncertainty_cases=10,focus_status_counts=dict(Counter(r['review']['status'] for r in focus)),
        all_status_counts=dict(Counter(r['review']['status'] for r in cases)),
        source_caches=len(inventory['sources']),image_bytes=sum(r['image']['bytes'] for r in cases),
        branch_observations={name:dict(available=sum(name in r['captions'] for r in focus),unchanged_from_A=sum(r['captions'].get(name)==r['caption_A'] for r in focus)) for name in inventory['branches'] if name!='A'},
        evidence_ledger=dict(original_pixels='Available and visually inspected; original +224px preview shown.',
            references='Exact five reference strings from SHA-pinned dataset; incomplete and occasionally disagree; not negative labels.',
            clip_tokens='Saved protocol confirms visual cache source/cohort; vectors not in the review ZIP, not probed locally.',
            original_prompt='Embedding source/cohort supported by saved protocols; exact original text not locally recovered; no reconstruction from references.',
            upt='Valid-pair count available only on dev1000 overlap; count is not a relation correctness score; individual hidden vectors/role scores absent.',
            regions='Existing regional supervision/native detail runs have saved captions; auxiliary regions were not caption input at inference for VG G/T. Not an oracle per-case intervention.'),
        limitations=['Purposeful diagnostic sample, not a representative error-rate/gain estimate.','Assistant visual judgments, not independently human-verified gold.','Cannot separate encoder missing information, poor alignment and decoder non-use from saved captions alone.','No new inference, training, metric calculation or fresh test.','Raw per-image CIDEr across dev1000 vs val5000 uses different IDF and is not compared.'])
    save(HERE/'RESULT.json',summary);save(HERE/'cases20.json',focus);save(HERE/'controls10.json',controls)
    save(OUT/'provenance.json',inventory)
    status_labels=dict(clear_error='Lỗi rõ theo ảnh',unsupported_claim='Chi tiết chưa được ảnh hỗ trợ',under_specified='Đúng một phần / thiếu cụ thể',uncertain='Chưa đủ chắc',acceptable_control='Caption có cơ sở')
    def render(rows,title,other):
        cards=[]
        for n,r in enumerate(rows,1):
            rv=r['review'];e=html.escape
            body=''.join(f'<tr><td>{e(k)}</td><td>{e(v)}</td><td>{"giữ A" if v==r["caption_A"] else "đổi câu"}</td></tr>' for k,v in r['captions'].items())
            refs=''.join('<li>'+e(t)+'</li>' for t in r['references'])
            cards.append(f'''<article id="c{r['coco_id']}"><h2>{n:02d} · COCO {r['coco_id']} · {status_labels[rv['status']]}</h2>
            <div class="pixels"><figure><img src="images/{r['filename']}" alt="Ảnh gốc COCO {r['coco_id']}"><figcaption>Ảnh gốc {r['image']['original_size']}</figcaption></figure><figure><img class="small" src="images/preview224_{r['filename']}" alt="Bản xem 224px"><figcaption>224×224 để người xem; không phải tensor cache đã kiểm parity</figcaption></figure></div>
            <p class="caption"><b>A:</b> {e(r['caption_A'])}</p><p><b>Quan sát:</b> {e(rv['observation'])}</p><p><b>Giới hạn:</b> {e(rv['uncertainty'])}</p>
            <p>CHAIR: {"có cờ" if r['automatic']['chair_flagged'] else "không cờ"} · UPT valid pairs trong receipt: {r['upt_valid_pairs_recorded'] if r['upt_valid_pairs_recorded'] is not None else "không có trace cho ảnh này"}. Số cặp không xác nhận quan hệ đúng.</p>
            <details><summary>Caption của các nhánh đã có</summary><table><tr><th>Nhánh</th><th>Caption</th><th>So A</th></tr>{body}</table></details>
            <details><summary>Năm references: bằng chứng bổ sung, không phải toàn bộ sự thật</summary><ul>{refs}</ul></details></article>''')
        return f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>{title}</title><style>body{{font:17px Arial;line-height:1.5;background:#eef2f6;color:#172b3a;max-width:1150px;margin:25px auto;padding:0 15px}}header,article{{background:white;padding:24px;border-radius:12px;margin:20px 0}}h1,h2{{color:#123f60}}h2{{font-size:22px}}.pixels{{display:flex;gap:20px;align-items:center;flex-wrap:wrap}}figure{{margin:0;max-width:700px}}img{{max-width:100%;max-height:500px;object-fit:contain}}.small{{width:224px;height:224px}}figcaption{{font-size:13px;color:#536677}}.caption{{background:#eef5fb;padding:12px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:9px;border:1px solid #d6dfe6;text-align:left}}details{{margin:14px 0}}a{{color:#145b8e}}</style><header><h1>{title}</h1><p>Rà soát bởi trợ lý, <b>chưa được người dùng xác nhận</b>. Không phải20 lỗi chắc chắn: có cả lỗi rõ, claim chưa được hỗ trợ và câu đúng nhưng ít cụ thể. Không dùng làm nhãn train hoặc tỷ lệ lỗi.</p><p><a href="{other[0]}">{other[1]}</a>. Không cần chấm thêm để chạy train; trang này chỉ giúp kiểm tra lập luận và chọn câu hỏi nghiên cứu.</p><p>Không có prompt text gốc/tensor CLIP/hidden UPT cho từng ảnh ở đây. Việc nhìn thấy chi tiết trong ảnh không chứng minh model đã biểu diễn hay sử dụng được chi tiết đó.</p></header>{''.join(cards)}</html>'''
    (OUT/'index.html').write_text(render(focus,'20 trường hợp phân tích bằng chứng của A',('controls.html','10 đối chứng / trường hợp chưa chắc')),encoding='utf-8')
    (OUT/'controls.html').write_text(render(controls,'10 đối chứng và trường hợp chưa đủ chắc',('index.html','Quay lại20 trường hợp')),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
