"""Generate portable G/T notebooks by reusing the successful A bootstrap."""
import ast
import importlib.util
import json
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('hidden_notebooks',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
# All old architecture/reward code stays pinned; new experiment sources are hashed overlays.
old.COMMIT='dfa1d4e0c52bea8672c6b770127034c2ec377b58'


def replace_once(source, before, after):
    if source.count(before)!=1: raise ValueError('Template changed: '+before)
    return source.replace(before,after)


def main():
    bundle=old.sources()
    for name in ('region_data.py','region_model.py','run_region.py','test_region_data.py',
                 'test_region_model.py','test_region_ddp.py','AUXILIARY_MANIFEST.json'):
        bundle[name]=(HERE/name).read_text(encoding='utf-8')
    bundle['auxiliary_payload.json']=zlib.decompress((HERE/'auxiliary_payload.bin').read_bytes()).decode('utf-8')
    template=json.loads((ROOT/'experiments/upt_hidden_pair/UPT_HIDDEN_CAPTION_A_2GPU.ipynb').read_text(encoding='utf-8'))
    for arm in ('G','T'):
        cells=json.loads(json.dumps(template['cells']))
        cells[0]=old.md(f'''
            # VG region-caption pilot {arm} — {'Q-Former đọc patch trong vùng' if arm=='G' else 'Q-Former đọc patch toàn ảnh, đối chứng cùng phrase'}
            Một notebook dùng **T4×2**; Internet ON; **Save & Run All**. G/T có thể chạy song song trên hai tài khoản.
            Caption: baseline A giữ nguyên legacy, train10k/dev1000,1250updates,globalbatch8,
            CE200 rồi CE+.05SCST(CIDErD+.5object_count_v2).
            Thêm .1 phrase-CE mỗi update: cùng10000phrase/3307ảnh, cùng thứ tự, không thêm tham số.
            Auxiliary: cả hai bỏ CLS và legacy; G mask bbox, T dùng196patch. Suy luận caption chỉ dùng ảnh+legacy như A.
            Annotation VG là giám sát yếu, chưa phải GT sạch. Không sửa cấu hình nhánh sau khi xem điểm nhánh trước.
            Dữ liệu VG nhỏ đã nhúng và hash khóa trong notebook, **không cần thêm dataset VG hay tải ảnh VG**.
            ''' )
        source=''.join(cells[1]['source'])
        source=replace_once(source,"ARM='A'",f"ARM='{arm}'")
        source=replace_once(source,'UPT HIDDEN A: BOOTSTRAP START',f'VG REGION {arm}: BOOTSTRAP START')
        source=replace_once(source,'/upt-hidden-pairs-caption11k/upt_hidden_pairs_11k','/upt-hidden-pairs-caption11k')
        source=replace_once(source,'/kaggle/working/upt_hidden_caption_A',f'/kaggle/working/vg_region_caption_{arm}')
        cells[1]=old.code(source)
        cells[2]=old.bootstrap(bundle,'region_'+arm.lower())
        source=''.join(cells[3]['source'])
        source=replace_once(source,"('test_hidden_data.py','test_hidden_hdf5.py','test_hidden_model.py')",
            "('test_hidden_data.py','test_hidden_hdf5.py','test_region_data.py','test_region_model.py')")
        source=replace_once(source,"'test_hidden_ddp.py'","'test_region_ddp.py'")
        cells[3]=old.code(source)
        source=''.join(cells[4]['source']).replace('run_hidden.py','run_region.py').replace('/tmp/upt_hidden_','/tmp/vg_region_')
        start=source.index('locked_numeric_gate=dict(')
        end=source.index('promotion_requires=',start)
        source=source[:start]+source[end:]
        source=replace_once(source,'Matched valid A/H, numeric gates and blind human review of supported relations/instances/extra facts; no automatic promotion.',
            'Matched G/T and historical A; numeric gates plus blind factual review; dev reused, no automatic promotion.')
        cells[4]=old.code(source)
        cells[5]=old.md('''
            Gửi `vg_region_caption_G_review.zip` và `vg_region_caption_T_review.zip`; giữ riêng checkpoint.bin mỗi nhánh.
            Chỉ so sánh sau khi cả hai hoàn tất. Không rerank, không chọn checkpoint theo dev, không dùng test.
            Gate: G−A CIDEr≥.01 và paired95% lower>0; CHAIRs/i không tăng, recall giảm tối đa.01;
            muốn quy lợi ích cho vùng cần G−T paired95% lower>0 và review mù nội dung được ảnh hỗ trợ.
            G ngang T chưa chứng minh căn chỉnh. Patch CLIP đã contextualized, mask không cô lập ngữ cảnh tuyệt đối.
            Nếu lỗi: gửi lỗi đầu tiên hoặc `print(LOG.read_text(encoding='utf-8')[-12000:])`.
            ''' )
        for i,cell in enumerate(cells):
            cell['id']=f'region-{i:02d}'
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']));cell.update(execution_count=None,outputs=[])
        payload=json.dumps(dict(template,cells=cells),ensure_ascii=False,indent=1).encode('utf-8')
        if len(payload)>=1_000_000:raise ValueError('Kaggle source must be under1MB')
        destination=HERE/f'VG_REGION_CAPTION_{arm}_2GPU.ipynb'
        destination.write_bytes(payload);print(destination.name,len(payload),'bytes; unexecuted')


if __name__=='__main__':main()
