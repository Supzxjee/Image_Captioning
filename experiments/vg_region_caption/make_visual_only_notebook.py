"""Create GV without regenerating or changing the historical G/T notebooks."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
from make_notebooks import old,replace_once

HERE=Path(__file__).resolve().parent


def embedded_sources(notebook):
    tree=ast.parse(''.join(notebook['cells'][2]['source']))
    packed=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    sources=json.loads(zlib.decompress(base64.b64decode(packed)))
    hashes=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCE_HASHES' for t in n.targets))
    for name,source in sources.items():
        if hashlib.sha256(source.encode()).hexdigest()!=hashes[name]:raise ValueError('Original G source mismatch')
    return sources


def main():
    original=HERE/'VG_REGION_CAPTION_G_2GPU.ipynb'
    template=json.loads(original.read_text(encoding='utf-8'))
    bundle=embedded_sources(template)
    for name in ('visual_only_model.py','run_visual_only.py','test_visual_only_model.py','test_visual_only_ddp.py'):
        bundle[name]=(HERE/name).read_text(encoding='utf-8')
    cells=template['cells']
    cells[0]=old.md('''
        # GV — VG phrase gradient chỉ cập nhật visual projection + Q-Former
        **Một notebook, T4×2, Internet ON, Save & Run All.** Không cần chạy lại A/G/T.
        Giữ nguyên checkpoint đầu của G (không load checkpoint G đã train), caption train10k/dev1000,
       10000phrase/3307ảnh, cùng mask/lịch,1250updates,globalbatch8,seed42,LR5e-6.
        Caption chính: CE200 rồi CE+.05SCST(CIDErD+.5object_count_v2), cập nhật visual và decoder như A/G.
        Phrase phụ: .1CE như G, nhưng decoder dùng tham số hiện tại detached; gradient vẫn đi qua decoder về visual memory.
        Không thêm tham số, không no_grad toàn decoder, không đóng băng decoder ở nhiệm vụ caption.
        Dataset đầu vào giống G; VG payload nhúng sẵn. Kiểm tra gradient và DDP phải PASS trước train.
        Bước staging cache vẫn giống G/T để giữ đối chứng; thời gian đọc trên Kaggle có thể lâu.
        ''' )
    source=''.join(cells[1]['source'])
    source=replace_once(source,"ARM='G'","ARM='GV'")
    source=replace_once(source,'VG REGION G: BOOTSTRAP START','VG VISUAL ONLY GV: BOOTSTRAP START')
    source=replace_once(source,'/kaggle/working/vg_region_caption_G','/kaggle/working/vg_region_caption_GV')
    cells[1]=old.code(source)
    cells[2]=old.bootstrap(bundle,'region_gv')
    source=''.join(cells[3]['source'])
    source=replace_once(source,"'test_region_model.py')","'test_region_model.py','test_visual_only_model.py')")
    source=replace_once(source,'test_region_ddp.py','test_visual_only_ddp.py')
    cells[3]=old.code(source)
    source=''.join(cells[4]['source']).replace('run_region.py','run_visual_only.py')
    source=replace_once(source,'Matched G/T and historical A; numeric gates plus blind factual review; dev reused, no automatic promotion.',
        'GV must beat A with locked numeric gates; mechanism attribution also requires GV-G evidence and blind factual review. No automatic tuning/full/seed rescue.')
    cells[4]=old.code(source)
    cells[5]=old.md('''
        Gửi `vg_region_caption_GV_review.zip`; giữ riêng `checkpoint.bin`.
        So với A và G đã có. GV−A CIDEr≥.01, paired95%lower>0, CHAIRs/i không tăng, recall giảm≤.01;
        muốn quy lợi ích cho thay đường gradient cần GV−G paired95%lower>0, rồi review mù nếu vượt gate.
        Nếu không vượt A hoặc tăng factual errors: dừng VG region-CE ở cache/quy mô này;
        không tự động thử freeze khác, trọng số khác hoặc thêm seed.
        Chỉ ngắt gradient trực tiếp vào tham số decoder ở nhiệm vụ phụ; decoder vẫn có thể đổi gián tiếp do visual memory đổi.
        Weak VG/CLIPcontext/dev tái sử dụng vẫn là giới hạn. Chưa có gain hoặc novelty được chứng minh.
        Nếu lỗi trước output: `print(LOG.read_text(encoding='utf-8')[-12000:])`.
        ''')
    for i,cell in enumerate(cells):
        cell['id']=f'gv-{i:02d}'
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']));cell.update(execution_count=None,outputs=[])
    raw=json.dumps(template,ensure_ascii=False,indent=1).encode('utf-8')
    if len(raw)>=1_000_000:raise ValueError('Kaggle notebook source must be below1MB')
    (HERE/'VG_REGION_CAPTION_GV_2GPU.ipynb').write_bytes(raw)
    print('GV unexecuted notebook:',len(raw),'bytes')


if __name__=='__main__':main()
