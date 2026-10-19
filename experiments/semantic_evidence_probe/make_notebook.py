"""Package a manual inference-only notebook; never execute its cells."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).parent
BASE = '8242105fdb0099700e9e8d82a6e098c5ce3ed700'


def code(source):
    source=textwrap.dedent(source).strip()+'\n'
    ast.parse(source)
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))


def markdown(source):
    return dict(cell_type='markdown',metadata={},source=textwrap.dedent(source).strip().splitlines(True))


def build():
    sources={n:(HERE/n).read_text(encoding='utf-8') for n in ('run_probe.py','probe_data.py','test_probe_data.py','test_probe_model.py')}
    sources['pilot_model.py']=(HERE.parent/'semantic_evidence_pilot/pilot_model.py').read_text(encoding='utf-8')
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    manifest=(HERE/'probe_manifest.json').read_text(encoding='utf-8')
    cells=[markdown('''
    # Kiểm tra bộ căn chỉnh E — không huấn luyện

    62 cặp / 59 ảnh: đổi quan hệ12, đổi vật thể12, đổi người38.
    Cả hai GPU cùng suy luận một checkpoint E; không chạy hai nhánh huấn luyện.
    Loại toàn bộ ảnh trong10k ảnh train pilot E. Các ảnh vẫn thuộc COCO train/restval và đã xuất hiện trong tiền huấn luyện caption; đây là chẩn đoán, không phải test độc lập.

    Chạy ngay sau notebook E trên cùng session hoặc thêm saved output/checkpoint E vào notebook mới.
    **Chỉnh CHECKPOINT bên dưới đúng đường dẫn checkpoint.pth của E**. Không dùng checkpoint SCST gốc/A.
    Giữ đúng checkpoint SHA651f411f...; notebook kiểm tra hash trước khi tải mô hình.
    Chọn T4x2, bật Internet, bạn tự Save & Run All. Không cần raw ảnh, prompt cache, V-COCO hay legacy prompt dataset.
    ''')]
    cells.append(code('''
    print('FROZEN E MATCHER PROBE: START — NO TRAINING', flush=True)
    import os
    os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='42',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    from pathlib import Path
    CHECKPOINT = Path('/kaggle/working/semantic_evidence_pilot_E/checkpoint.pth')
    VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
    OUTPUT = Path('/kaggle/working/semantic_evidence_matcher_probe')
    # Nếu mở notebook mới: thay CHECKPOINT bằng đường dẫn chính xác từ Input của output E đã lưu.
    assert CHECKPOINT.is_file(), f'Missing E checkpoint: {CHECKPOINT}. Copy its actual Input path; do not use original/A.'
    assert VISUAL_CACHE.is_dir(), VISUAL_CACHE
    assert not OUTPUT.exists(), f'Use a fresh OUTPUT: {OUTPUT}'
    '''))
    setup='''
    import json, subprocess, sys, hashlib
    REPO = Path('/tmp/Image_Captioning-semantic-evidence-probe')
    def run(args, cwd=None):
        print('Running:', ' '.join(map(str,args)), flush=True)
        subprocess.run(list(map(str,args)), cwd=cwd, check=True)
    if not REPO.exists(): run(['git','clone','--no-checkout','https://github.com/Supzxjee/Image_Captioning.git',REPO])
    assert (REPO/'.git').is_dir(), REPO
    run(['git','fetch','--depth','1','origin','__BASE__'],cwd=REPO)
    run(['git','checkout','--detach','__BASE__'],cwd=REPO)
    run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy'])
    PROBE = REPO/'frozen_e_probe'; PROBE.mkdir(exist_ok=True)
    os.environ['PYTHONPATH'] = str(REPO) + os.pathsep + str(PROBE)
    '''
    setup=textwrap.dedent(setup).replace('__BASE__',BASE)
    setup+='\nSOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\nMANIFEST_TEXT = '+repr(manifest)+'\n'
    setup+="for name, source in SOURCES.items(): (PROBE/name).write_text(source,encoding='utf-8')\n"
    setup+="MANIFEST = PROBE/'probe_manifest.json'; MANIFEST.write_text(MANIFEST_TEXT,encoding='utf-8')\n"
    setup+="for name, expected in SOURCE_HASHES.items(): assert hashlib.sha256((PROBE/name).read_bytes()).hexdigest()==expected\n"
    cells.append(code(setup))
    cells.append(code('''
    import torch
    assert torch.cuda.device_count()==2, 'Select GPU T4 x2.'
    for rank in (0,1): print('GPU',rank,torch.cuda.get_device_name(rank),flush=True)
    run([sys.executable,'-m','unittest','test_probe_data','test_probe_model','-v'],cwd=PROBE)
    run([sys.executable,'-u','-c',"from transformers import CLIPTextModel,AutoTokenizer; r='57c216476eefef5ab752ec549e440a49ae4ae5f3'; CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16',revision=r); AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16',revision=r); print('Pinned text encoder ready',flush=True)"])
    def sha(path):
        value=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
        return value.hexdigest()
    expected=json.loads(MANIFEST_TEXT)['checkpoint_sha256']
    assert sha(CHECKPOINT)==expected, 'Checkpoint does not match completed E. Do not substitute original/A.'
    cfg=dict(checkpoint=str(CHECKPOINT),visual_cache=str(VISUAL_CACHE),output=str(OUTPUT),
             manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),source_sha256=SOURCE_HASHES)
    CONFIG=PROBE/'config.json'; CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    run([sys.executable,'-u',PROBE/'run_probe.py','--config',CONFIG],cwd=REPO)
    '''))
    cells.append(markdown('''
    ## Đọc kết quả
    Tải `semantic_evidence_matcher_probe_review.zip` gửi lại. ZIP không chứa checkpoint.
    `summary.json` báo tỷ lệ chọn cặp đúng, hòa tính0.5; bootstrap theo ảnh. Điểm cosine không phải xác suất.
    So `boxed` với `global_view` (bỏ ràng buộc box) và `zero_visual` (xóa đặc trưng ảnh).
    Nhãn âm là người đã chấm hoặc V-COCO ghi rõ actor/action=0; không suy từ thiếu nhãn.
    Cặp đổi động từ cân bằng holding/riding trong từng lớp vật thể; vẫn chỉ12 cặp nên không kết luận rộng.
    Điểm tốt trên bộ này cũng chưa chứng minh tăng CIDEr hoặc tính đúng của caption.
    '''))
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=4)
    (HERE/'FROZEN_E_MATCHER_PROBE_2GPU.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
    print('Notebook packaged; no cells executed.')


if __name__=='__main__': build()
