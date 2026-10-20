"""Collect official JPEGs and reuse the existing blank human review interface."""
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/phrase_pair_feasibility'))
from data import identity
from render import TEMPLATE


def dhash(image):
    image=image.convert('L').resize((9,8));pixels=list(image.getdata())
    return sum((pixels[y*9+x]>pixels[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))


def main():
    out=HERE/'review';m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));directory=out/'images';directory.mkdir(exist_ok=True)
    images={e['filename']:e for g in m['groups'] for e in g['events']};receipts=[]
    for filename,event in images.items():
        target=directory/filename;url=f'https://images.cocodataset.org/{event["filepath"]}/{filename}'
        if target.exists():raw=target.read_bytes()
        else:
            try:
                with urlopen(url,timeout=40) as r:raw=r.read()
            except Exception:
                url=url.replace('https://','http://')
                with urlopen(url,timeout=40) as r:raw=r.read()
            target.write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as im:
            assert im.format=='JPEG' and im.getexif().get(274,1)==1
            im.load();width,height=im.size;fingerprint=dhash(im)
        for g in m['groups']:
            for e in g['events']:
                if e['filename']!=filename:continue
                assert e['width'] in (None,width) and e['height'] in (None,height)
                e.update(width=width,height=height)
        receipts.append(dict(filename=filename,url=url,sha256=hashlib.sha256(raw).hexdigest(),width=width,height=height,dhash=f'{fingerprint:016x}'))
        print('JPEG checked',len(receipts),'/',len(images),filename,flush=True)
    prior=[]
    for path in (ROOT/'experiments/phrase_pair_feasibility/review/images').glob('*.jpg'):
        with Image.open(path) as im:prior.append((path.name,dhash(im)))
    flags=[]
    for i,a in enumerate(receipts):
        comparisons=prior+[(b['filename'],int(b['dhash'],16)) for b in receipts[:i]]
        for name,value in comparisons:
            if name==a['filename']:continue
            distance=(int(a['dhash'],16)^value).bit_count()
            if distance<=6:flags.append(dict(candidate=a['filename'],other=name,distance=distance))
    m['groups_sha256']=identity(m['groups']);m['near_duplicate_flags']=flags
    m['near_duplicate_scope']='64bit dHash distance<=6; new candidates vs each other and53oldfeasibilityJPEGs; flags require inspection, not certified exhaustive'
    (out/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'image_receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    n=len(m['groups'])
    template=TEMPLATE.replace('Audit 30 nhóm mô tả–bằng chứng','Crop/fullframe: kiểm tra ứng viên')
    template=template.replace('Kiểm tra tính khả thi: 30 nhóm mô tả ↔ bằng chứng ảnh',f'Crop/fullframe: {n-1} ứng viên mới + 1 nhóm fit cần xác nhận lại')
    template=template.replace('Làm <b>10 nhóm đầu trước</b>, xuất JSON gửi lại để kiểm tra tính khả thi. Chưa cần hoàn tất cả 30.',
        'Tối đa 10 ứng viên mỗi loại ván. Nhóm cuối fit_reconfirm_235799 là xác nhận nhãn cũ, không phải probe mới. Không xem output model khi chấm.')
    template=template.replace('Đây là đánh giá pixel, chưa phải thông tin CLIP thực sự giữ được.',
        'Trường 224 chỉ ghi nhận: có thể chọn Không/Không rõ mà nhóm vẫn hợp lệ nếu ảnh gốc phân biệt được. Không ép hai câu thành đối lập khi cả hai cùng đúng.')
    template=template.replace('/30',f'/{n}').replace('phrase_pair_feasibility_review.json','pair_crop_candidate_review.json')
    embedded=json.dumps(m,ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(template.replace('__MANIFEST__',embedded),encoding='utf-8')
    summary=json.loads((HERE/'PREPARATION.json').read_text(encoding='utf-8'))
    summary.update(groups_sha256=m['groups_sha256'],images_verified=len(receipts),near_duplicate_flags=flags,
        annotation_boxes_only=False,scope=m['near_duplicate_scope'],selection='Take first4 human-valid groups in each family in existing hash order; no model-output selection; no replacing failures beyond10/family')
    (HERE/'PREPARATION.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Review ready, blank labels. Duplicate flags:',flags)


if __name__=='__main__':main()
