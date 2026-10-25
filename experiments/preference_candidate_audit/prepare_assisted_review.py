"""Package visual AI preannotations separately from later human verification."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent

# Choices recorded from viewing all30 original COCO images before consulting the hidden key.
JUDGMENTS=[
 ('7592','tie','medium','Cả hai đều mô tả được các biển xanh/trắng gắn trên cột. Chưa có khác biệt nội dung đủ rõ để chọn một câu.'),
 ('241935','Y','high','Xe đạp nằm trước cửa hàng có hàng hóa và biển hiệu; không có bằng chứng đây là cửa nhà ở như X.'),
 ('442811','X','medium','Người trên ván trượt tuyết và hệ thống cáp treo hiện rõ. X mô tả thêm chi tiết này; Y cũng hợp lý nhưng hướng chuyển động khó xác nhận từ ảnh.'),
 ('393125','X','high','Đèn giao thông treo trên đường phố với bầu trời hoàng hôn. Y gọi là ban đêm và bỏ mất bối cảnh rõ trong ảnh.'),
 ('102858','Y','high','Ghế trên vỉa hè trước các cửa hàng. Chưa thấy bằng chứng đó là quán cà phê như X.'),
 ('57940','Y','medium','Người trượt trên đường dốc tuyết được dựng cho sự kiện, có rào và băng quảng cáo. Y phù hợp hơn mô tả ngọn đồi chung chung của X; việc gọi competition vẫn có yếu tố suy luận.'),
 ('256875','unclear','low','Ảnh là đường tuyết giữa cây rừng, không rõ có thể gọi field không. Từ ảnh tĩnh khó phân biệt đang di chuyển với đứng trên skis; cần bạn kiểm tra lại.'),
 ('170221','X','medium','Có người đang làm động tác với ván trên vỉa hè. X dùng person an toàn hơn khẳng định kid của Y; tuổi không xác định rõ trong ảnh.'),
 ('16335','X','medium','Hai con ngựa trên nền cát; X nói đứng là phù hợp với tư thế nhìn thấy. Chưa đủ bằng chứng cho hành động đi bộ ở Y.'),
 ('429025','Y','high','Có người trên phố vào ban đêm. X thêm nhận định small town không xác minh được từ ảnh.'),
 ('78410','Y','medium','Xe buýt màu trắng ở bên đường được thấy rõ. X cũng khá đúng nhưng Y mô tả màu sắc và vị trí cụ thể, có bằng chứng hơn.'),
 ('101172','unclear','high','Gấu bông nằm phía trước người lái, không phải trên lưng như X. Y có thể bị hiểu là gấu đội mũ, trong khi người lái mới đội mũ. Cả hai câu đều có vấn đề về quan hệ hoặc cách gắn cụm từ.'),
 ('509589','Y','medium','Người đeo kính đứng trên ván; phía sau có người đứng quan sát. X nói người phía sau ngồi chưa được ảnh hỗ trợ rõ. Boy/man và eye glasses/sunglasses cần bạn cân nhắc lại.'),
 ('302713','Y','medium','Nhiều xe đang dừng/chờ đèn giao thông. Y sát cảnh hơn X nói xe đang chạy, nhưng parked là từ chưa thật chuẩn cho dừng chờ đèn.'),
 ('419644','X','medium','Người cầm dao và nĩa để cắt/ăn pizza. Không có khay rau xanh riêng như Y. X nói slice hơi cụ thể vì pizza còn nguyên trên bàn.'),
 ('229188','tie','high','Hai câu gần như cùng nội dung: người với ván trượt ở khu dốc bê tông. Không có khác biệt đủ rõ về tính đúng của nội dung.'),
 ('260951','Y','high','Hai nài ngựa đang đua trên đường đua. Cả hai khá đúng; Y mô tả thêm người cưỡi là thành phần nổi bật trong ảnh.'),
 ('450543','tie','high','Cả hai đều nói người đang nhảy/làm động tác với ván trên không. Khác biệt chủ yếu ở cách diễn đạt.'),
 ('176778','unclear','high','Có một bồn cầu, một bồn tắm và mèo đứng cạnh. X lặp toilet dễ thành hai bồn cầu và bỏ mèo; Y nhận đúng mèo nhưng nói sitting trong khi mèo có vẻ đứng.'),
 ('167913','X','high','Người lái đang điều khiển mô tô màu vàng trên đường. Không phải mô tô đỗ bên đường như Y.'),
 ('40844','unclear','high','Trẻ đang mở/kéo ngăn tủ trong bếp. Không thấy trẻ chuẩn bị thức ăn như X; making a mess ở Y cũng là suy diễn chưa có bằng chứng rõ.'),
 ('46720','Y','medium','Nhiều con cừu cúi đầu ăn cỏ trên sườn đồi. Grazing của Y mô tả đúng hành động hơn standing chung chung ở X.'),
 ('523795','X','medium','Hai hươu cao cổ trên cỏ cạnh/dưới cây. X bổ sung cây và cỏ nhìn thấy rõ; Y cũng đúng nhưng ít thông tin hơn.'),
 ('395717','tie','high','Người ngồi trên thuyền, đội mũ và mặc áo khoác, phía sau có nước. Hai câu đều được ảnh hỗ trợ; chưa có câu nào hơn rõ ràng.'),
 ('326731','Y','medium','Có bốn hươu cao cổ, tường đá phía sau và cỏ dưới chân. X đếm đúng bốn nhưng gọi nền đất đá chưa chuẩn; Y mô tả nền cỏ an toàn hơn.'),
 ('9018','unclear','medium','Có người cưỡi ngựa đốm sáng màu trên đường. X gọi brown horse chưa chính xác rõ; Y gọi biển MOTEL là street sign cũng cần cân nhắc. Không đủ chắc để chọn một câu hoàn toàn tốt hơn.'),
 ('271696','Y','high','Nhìn rõ ba người ngồi trên ghế dưới ô. X nói couple không khớp số người thấy được.'),
 ('289410','X','high','Mèo nằm trên mặt chăn/giường. Y nói mèo nằm dưới chăn là sai quan hệ.'),
 ('551575','X','high','Nhóm người với ván lướt ở cạnh biển, nhiều người đang cầm ván. X mô tả được quan hệ cầm ván và bối cảnh biển; có một người ngồi xổm nên không hiểu standing là tất cả mọi người đều đứng.'),
 ('216333','Y','medium','Có nhiều vali xếp chồng cao. Phần trên của đống vali bị cắt khỏi khung hình nên chưa xác nhận được khẳng định lên tới trần của X.'),
]


def build():
    root=HERE/'review';loaded=json.loads((root/'loaded.json').read_text(encoding='utf-8'));files=loaded['files']
    original=files['review_30.html'];fingerprint=hashlib.sha256(files['eligible_pairs.json'].encode()).hexdigest()
    ids=re.findall(r'<h2>COCO (\d+)</h2>',original)
    assert ids==[j[0] for j in JUDGMENTS] and len(set(ids))==30
    data=dict(schema=1,manifest_sha256=fingerprint,reviewer='AI assistant',reviewer_type='ai_preannotation',
        human_verified=False,annotations={i:choice for i,choice,_,_ in JUDGMENTS},
        notes={i:note for i,_,_,note in JUDGMENTS},confidence={i:confidence for i,_,confidence,_ in JUDGMENTS},
        evidence='All30 original COCO images viewed in browser screenshots using HTTPS s3.amazonaws.com/images.cocodataset.org; no model inference.',
        caution='AI preannotations are not human annotations and cannot satisfy the locked human gate.')
    # Seal choices before comparing to the model preference key.
    (root/'assistant_preannotations.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    key=json.loads(files['review_key.json']);agreed=[i for i,choice,_,_ in JUDGMENTS if choice==key[i]]
    summary=dict(status='AI_PREANNOTATION_COMPLETE_HUMAN_PENDING',records=30,choices=dict(Counter(j[1] for j in JUDGMENTS)),
        agrees_with_selected_preference=len(agreed),opposes_selected_preference=sum(choice in ('X','Y') and choice!=key[i] for i,choice,_,_ in JUDGMENTS),
        ties_or_unclear=sum(choice not in ('X','Y') for _,choice,_,_ in JUDGMENTS),
        human_gate_pass=None,manifest_sha256=fingerprint,
        interpretation='AI-only screen finds uncertain/mistaken/near-equivalent pairs. Human verification still required; do not alter rules or expand/train automatically.')
    (HERE/'AI_REVIEW_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Keep X/Y, ordering and original captions. Only the assistant copy is prefilled.
    page=original.replace('https://images.cocodataset.org/','https://s3.amazonaws.com/images.cocodataset.org/')
    page=page.replace('<h1>Kiểm tra cặp caption TRAIN</h1>','<h1>Kiểm tra lại phần chấm trước của trợ lý</h1><p><strong>Đây là nhãn AI, chưa phải đánh giá của bạn.</strong> Hãy xem ảnh, sửa X/Y nếu cần và đánh dấu đã kiểm tra từng cặp. Không cần đồng ý với gợi ý của trợ lý.</p><p>Người kiểm tra: <input id="reviewer" placeholder="Họ tên"></p>')
    for i,choice,confidence,note in JUDGMENTS:
        pattern=r'(<select data-id="'+i+r'">)(.*?)(</select>)'
        def fill(match):
            options=match[2]
            if choice in ('X','Y'):options=options.replace('<option>'+choice+'</option>','<option selected>'+choice+'</option>')
            else:options=options.replace('<option value="'+choice+'">','<option selected value="'+choice+'">')
            return match[1]+options+match[3]
        page,n=re.subn(pattern,fill,page);assert n==1
        marker='</select></label>'
        start=page.index('<h2>COCO '+i+'</h2>');end=page.index(marker,start)+len(marker)
        extra='<p><strong>Ghi chú AI:</strong> '+html.escape(note)+'</p><p>Ghi chú của bạn: <textarea data-note="'+i+'" rows="2" style="width:95%"></textarea></p><p><label><input type="checkbox" data-verified="'+i+'"> Tôi đã xem ảnh và kiểm tra lựa chọn của cặp này</label></p>'
        page=page[:end]+extra+page[end:]
    page=page[:page.index('<button onclick="download()">')]+'''<p id="progress"></p><button onclick="download()">Tải phần đánh giá đã kiểm tra</button><script>
const fingerprint=FINGERPRINT;
function download(){const reviewer=document.querySelector('#reviewer').value.trim();const annotations={},notes={},verified={};document.querySelectorAll('select[data-id]').forEach(s=>annotations[s.dataset.id]=s.value);document.querySelectorAll('[data-note]').forEach(s=>notes[s.dataset.note]=s.value);document.querySelectorAll('[data-verified]').forEach(s=>verified[s.dataset.verified]=s.checked);const count=Object.values(verified).filter(Boolean).length;if(!reviewer){alert('Vui lòng ghi tên người kiểm tra.');return}if(count!==30){alert('Mới xác nhận '+count+'/30 cặp. Hãy kiểm tra đủ các cặp trước khi xuất nhãn người.');return}const result={schema:1,manifest_sha256:fingerprint,reviewer,reviewer_type:'human_verified_ai_assisted',human_verified:true,annotations,notes,verified};const b=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='candidate_pair_assisted_verified_review.json';a.click();URL.revokeObjectURL(a.href)}
function progress(){document.querySelector('#progress').textContent='Đã xác nhận '+document.querySelectorAll('[data-verified]:checked').length+'/30 cặp.'}document.addEventListener('change',progress);progress();
</script>'''.replace('FINGERPRINT',json.dumps(fingerprint))
    assert page.count('data-verified=')==30 and page.count('<article>')==30
    assert 'review_key' not in page and 'human_verified_ai_assisted' in page
    (root/'assistant_review.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
