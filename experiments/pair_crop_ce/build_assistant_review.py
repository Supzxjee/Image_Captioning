"""Separate AI-prefilled review page. Never overwrite the blank review or human state."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def build():
    directory=HERE/'review';raw=(directory/'assistant_proposed_review.json').read_bytes();proposal=json.loads(raw)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert proposal['reviewer_type']=='assistant_proposed_not_human' and proposal['groups_sha256']==manifest['groups_sha256']
    assert set(proposal['annotations'])=={g['group_id'] for g in manifest['groups']}
    assert all(not a['confirmed_by_human'] for a in proposal['annotations'].values())
    html=(directory/'index.html').read_text(encoding='utf-8')
    sha=hashlib.sha256(raw).hexdigest()
    payload=json.dumps(proposal,ensure_ascii=False).replace('<','\\u003c')
    html=html.replace('<script>','<script id="assistant-proposals" type="application/json">'+payload+'</script>\n<script>',1)
    html=html.replace("'use strict';","'use strict';\nconst proposals=JSON.parse(document.getElementById('assistant-proposals').textContent);",1)
    html=html.replace("const key='phrase-pair-audit-'+manifest.groups_sha256",f"const key='assistant-review-{sha}-'+manifest.groups_sha256")
    html=html.replace("let state={reviewer:'',annotations:{}},position=0;","let state={reviewer:'',annotations:JSON.parse(JSON.stringify(proposals.annotations))},position=0;")
    html=html.replace("typeof a.notes==='string';","typeof a.notes==='string'&&typeof a.confirmed_by_human==='boolean';")
    html=html.replace('const count=Object.values(state.annotations).filter(completed).length;',
        'const count=Object.values(state.annotations).filter(a=>completed(a)&&a.confirmed_by_human===true).length;')
    html=html.replace('Đã chấm đủ trường:', 'Bạn đã kiểm tra và xác nhận:')
    html=html.replace("change(e.value);save();", "change(e.value);answer(groups[position]).confirmed_by_human=false;render();")
    html=html.replace("notes.oninput=()=>{a.notes=notes.value;save();};", "notes.oninput=()=>{a.notes=notes.value;a.confirmed_by_human=false;confirm.checked=false;save();};")
    needle='response.append(label);root.append(response);save();}'
    replacement="""response.append(label);
 const warning=element('p','Nhãn gợi ý do Codex chấm, CHƯA là nhãn người. Đọc cả ảnh và ghi chú; sửa ô sai rồi đánh dấu xác nhận.','warn');response.append(warning);
 const confirm=element('input');confirm.type='checkbox';confirm.checked=a.confirmed_by_human===true;
 const confirmation=element('label');confirmation.append(confirm,element('span',' Tôi đã tự kiểm tra cả hai cặp và xác nhận các ô hiện tại (kể cả Không rõ).'));
 confirm.onchange=()=>{a.confirmed_by_human=confirm.checked;save();};response.append(confirmation);
 root.append(response);save();}"""
    assert needle in html;html=html.replace(needle,replacement)
    html=html.replace("reviewer_type:'human_self_reported',annotations,exported_at:",
        "reviewer_type:Object.values(annotations).every(a=>a.confirmed_by_human===true&&completed(a))?'human_self_reported':'assistant_proposed_pending_human_verification',assistant_proposal_sha256:'"+sha+"',label_origin:'assistant_draft_then_explicit_human_confirmation',training_approved:false,annotations,exported_at:")
    html=html.replace('pair_crop_candidate_review.json','pair_crop_assisted_verified_review.json')
    html=html.replace('<b>Chưa huấn luyện. Không có nhãn đúng/sai điền sẵn.</b>',
        '<b>Bản nhãn sơ bộ của Codex để bạn kiểm tra. Chưa phải nhãn người xác nhận; chưa dùng huấn luyện.</b>')
    html=html.replace('Chưa có nhãn đúng/sai điền sẵn.','Các ô đang chứa nhãn gợi ý của Codex.')
    html=html.replace('Đã xuất JSON. Gửi lại tệp này; chưa dùng làm nhãn huấn luyện tự động.',
        'Đã xuất JSON kèm trạng thái xác nhận. Nếu chưa xác nhận đủ, tệp vẫn được đánh dấu có nhãn AI chưa kiểm tra. Gửi lại để kiểm tra, chưa tự dùng huấn luyện.')
    (directory/'assistant_review.html').write_text(html,encoding='utf-8')
    print('Separate assistant_review.html built; original index.html and human localStorage untouched')


if __name__=='__main__':build()
