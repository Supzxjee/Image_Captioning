"""Build a self-contained manual notebook. Never executes its cells."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/semantic_evidence_pilot'
PROBE=ROOT/'experiments/semantic_evidence_probe'
COMMIT='c09823178a9b28186e71554d368e2716c22af176'


def sources():
    result={name:(HERE/name).read_text(encoding='utf-8') for name in (
        'pilot_model.py','run_verified.py','verified_data.py','test_verified_data.py','test_verified_model.py','test_verified_ddp.py')}
    for name in ('pilot_data.py','test_pilot_data.py','pilot_metric_smoke.py'):
        result[name]=(OLD/name).read_text(encoding='utf-8')
    result['evidence_core.py']=(OLD/'pilot_model.py').read_text(encoding='utf-8')
    result['base_runner.py']=(OLD/'run_pilot.py').read_text(encoding='utf-8')
    result['probe_data.py']=(PROBE/'probe_data.py').read_text(encoding='utf-8')
    result['synthetic_fixtures.py']=(OLD/'test_pilot_model.py').read_text(encoding='utf-8').split('class EvidenceChecks')[0]
    for source in result.values():ast.parse(source)
    return result


def main():
    nb=json.loads((OLD/'SEMANTIC_EVIDENCE_E_2GPU.ipynb').read_text(encoding='utf-8'))
    embedded=sources();labels=(HERE/'verified_manifest.json').read_text(encoding='utf-8')
    hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in embedded.items()}
    label_hash=hashlib.sha256(labels.encode()).hexdigest()
    texts=[''.join(c['source']) for c in nb['cells']]
    texts[0]='''# Verified same-image alignment — F, hai GPU

Một nhánh trên **GPU T4 ×2**, Internet ON, **Save & Run All**. Bạn tự chạy notebook.
Giữ kiến trúc E và nền caption SCST+CE+thưởng đối tượng; thay loss giả nhãn bằng nhãn V-COCO xác nhận trên cùng ảnh.
Không cần thêm dataset: nhãn và mã đã nhúng sẵn. Giữ sáu Input giống A/E.

503 cặp fit / 310 ảnh; 140 cặp calibration / 79 ảnh; probe cũ 62 cặp / 59 ảnh.
Không dùng nhãn calibration/probe để huấn luyện hay chọn checkpoint. Không dùng test caption.
Tập fit nhỏ và được lặp lại; chưa có bằng chứng phương pháp này cải thiện caption.
'''
    texts[1]=texts[1].replace('SEMANTIC EVIDENCE E','VERIFIED SAME IMAGE F').replace("ARM = 'E'","ARM = 'F'").replace('semantic_evidence_pilot_E','semantic_evidence_verified_pairs_F')
    texts[1]+="\n# Khởi tạo từ checkpoint nền gốc (SHA 5f8...), KHÔNG dùng checkpoint E/bin.\n"
    setup=texts[3].split('\nSOURCES = ')[0]
    setup=setup.replace("PROJECT_COMMIT = '1dd4dc15f42eded978945a3da66c65ec6a725b59'",f"PROJECT_COMMIT = '{COMMIT}'")
    setup=setup.replace('/tmp/Image_Captioning-semantic-evidence-pilot','/tmp/Image_Captioning-verified-same-image')
    # Keep source below Kaggle's 1 MB kernel limit without requiring another Input.
    bundle=json.dumps(dict(sources=embedded,labels=labels),ensure_ascii=False,separators=(',',':')).encode()
    setup+='\nimport base64, hashlib, zlib\n'
    setup+='BUNDLE_B64 = '+repr(base64.b64encode(zlib.compress(bundle,9)).decode())+'\n'
    setup+='BUNDLE_SHA = '+repr(hashlib.sha256(bundle).hexdigest())+'\n'
    setup+="bundle_bytes = zlib.decompress(base64.b64decode(BUNDLE_B64,validate=True))\n"
    setup+="assert hashlib.sha256(bundle_bytes).hexdigest()==BUNDLE_SHA,'Embedded bundle changed'\n"
    setup+="bundle = json.loads(bundle_bytes)\nSOURCES = bundle['sources']\nVERIFIED_LABELS = bundle['labels']\n"
    setup+='SOURCE_HASHES = '+repr(hashes)+'\n'
    setup+='VERIFIED_MANIFEST_SHA = '+repr(label_hash)+'\n'
    setup+="for name, source in SOURCES.items():\n    assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_HASHES[name],name\n    (REPO/name).write_text(source,encoding='utf-8')\n"
    setup+="assert hashlib.sha256(VERIFIED_LABELS.encode()).hexdigest()==VERIFIED_MANIFEST_SHA\n"
    setup+="VERIFIED_MANIFEST = REPO/'verified_manifest.json'\nVERIFIED_MANIFEST.write_text(VERIFIED_LABELS,encoding='utf-8')\n"
    setup+="print('Verified F source and locked labels installed',flush=True)\n"
    texts[3]=setup
    texts[4]='## 2. Preflight\nKiểm tra dữ liệu, loss và gradient bằng dữ liệu giả, sau đó đồng bộ hai GPU và metric. Notebook dừng nếu kiểm tra lỗi.\n'
    texts[5]=texts[5].replace("('test_pilot_data.py','test_pilot_model.py')","('test_pilot_data.py','test_verified_data.py','test_verified_model.py')").replace('test_pilot_ddp.py','test_verified_ddp.py')
    texts[6]='''## 3. Chuẩn bị dữ liệu
Stage cùng 11.000 ảnh caption như A/E và các ảnh trong bộ nhãn xác nhận.
Embedding CLIP đóng băng; riêng tập fit được đưa vào loss. Hai tập còn lại chỉ dùng chẩn đoán.
'''
    texts[7]=texts[7].replace("embedded_source_sha256=SOURCE_HASHES)","embedded_source_sha256=SOURCE_HASHES,verified_manifest=str(VERIFIED_MANIFEST),\n           verified_manifest_sha256=VERIFIED_MANIFEST_SHA,aux_stage='/tmp/verified_same_image_aux_F.h5')")
    texts[7]=texts[7].replace("REPO/'run_pilot.py'","REPO/'run_verified.py'")
    texts[7]+="\nprint('LOCKED VERIFIED COUNTS:',locked['verified_counts'],flush=True)\n"
    texts[8]='''## 4. Huấn luyện F trên hai GPU
1.250 cập nhật; batch toàn cục 8, mỗi GPU 4 ảnh caption + 4 cặp xác nhận.
200 cập nhật đầu: CE + 0,1 loss căn chỉnh. Sau đó thêm 0,05 SCST với reward CIDEr-D + 0,5 thưởng đối tượng.
Loss căn chỉnh: max(0; 0,2 − điểm đúng + điểm sai), dùng cặp đã xác nhận trên cùng ảnh.
Lịch cố định 5.000 lượt sai quan hệ và 5.000 lượt sai người; không dùng loss giả nhãn E cũ.
Giữ checkpoint cuối; không chọn bằng calibration/probe/test. Đây là nhiều giám sát và tính toán hơn A/E.
'''
    texts[9]=texts[9].replace("REPO/'run_pilot.py'","REPO/'run_verified.py'")
    texts[10]='''## 5. Đánh giá và lưu
Caption greedy val1000, cùng A/E; thêm tắt nhánh, tráo text, tráo vùng.
Matcher trước/sau học trên calibration và probe cố định; có toàn ảnh và xoá đặc trưng làm kiểm tra đối chứng.
Calibration/probe thuộc caption train/restval và đã từng dùng trong nghiên cứu trước, không phải test hoàn toàn mới.
Sai vật thể chỉ kiểm tra chuyển giao, không có cặp âm vật thể được tự suy diễn để học.
Gửi ZIP review; giữ riêng checkpoint.pth trong Saved Output. ZIP không chứa checkpoint/HDF5.
'''
    texts[11]=texts[11].replace("REPO/'run_pilot.py'","REPO/'run_verified.py'")
    for i,(cell,text) in enumerate(zip(nb['cells'],texts)):
        cell['source']=text.splitlines(True);cell['id']=f'f-{i:02}'
        if cell['cell_type']=='code':
            ast.parse(text);cell['outputs']=[];cell['execution_count']=None
    output=HERE/'VERIFIED_SAME_IMAGE_F_2GPU.ipynb'
    serialized=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    assert len(serialized)<900000,'Notebook must stay below Kaggle 1 MB limit'
    assert sum(len(text.encode()) for text in texts)<900000,'Kernel source must stay below 1 MB'
    output.write_bytes(serialized)
    print(output.name,f'built: {len(serialized):,} bytes; cells NOT executed')


if __name__=='__main__':main()
