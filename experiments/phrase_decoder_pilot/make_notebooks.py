"""Build two small, unexecuted notebooks with identical code/data; no notebook launch."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

HERE=Path(__file__).resolve().parent
BASE='bb2039b8a9ba9d4606047ae8041ee1d6d79ac840'
NAMES=('phrase_data.py','phrase_model.py','run_phrase.py','phrase_review.py','test_phrase_data.py','test_phrase_model.py')


def sources():return {n:(HERE/n).read_text(encoding='utf-8') for n in NAMES}


def cell(kind,text):
    text=textwrap.dedent(text).strip()+'\n'
    if kind=='code':ast.parse(text)
    result=dict(cell_type=kind,metadata={},source=text.splitlines(True))
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def build(arm):
    src=sources();manifest=(HERE/'locked_manifest.json').read_text(encoding='utf-8')
    for code in src.values():ast.parse(code)
    payload=json.dumps(dict(sources=src,manifest=manifest),ensure_ascii=False,separators=(',',':')).encode()
    hashes={n:hashlib.sha256(code.encode()).hexdigest() for n,code in src.items()}
    cells=[cell('markdown',f'''
    # Phrase decoder technical pilot — {arm}, T4 ×2
    **Một notebook/một nhánh dùng cả hai GPU bằng DDP. Người dùng tự chạy.**
    CE và CROSS cùng kiến trúc/data/schedule: 14 nhóm fit, 7 nhóm adaptation holdout, 300 updates.
    CROSS thêm0.25 loss đối chứng từ bước61; CE giữ0. Hai nhánh đều tính cùng logits/loss phụ để giữ compute gần nhau.
    Đây là phép thử sinh cụm nhỏ trên ảnh đã review/train trước đây, **không phải caption benchmark hoặc chứng minh novelty**.
    Không SCST, không CHAIR/CIDEr, không selector/nối câu. Legacy prompt/UPT text không vào model.
    Bật Internet, chọn T4 ×2, Save & Run All. Chỉ cần ba Input: checkpoint nền, visual-cache, MSCOCO2014(JSON+ảnh).
    '''),cell('code',f'''
    print('PHRASE DECODER {arm}: BOOTSTRAP START',flush=True)
    from pathlib import Path
    ARM = '{arm}'
    # Dùng checkpoint NỀN đã dùng cho A–F, SHA5f8eead3..., KHÔNG dùng F checkpoint_F.bin SHA4938...
    CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
    VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
    COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
    COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
    OUTPUT = Path('/kaggle/working/phrase_decoder_' + ARM.lower() + '_21')
    # Nếu mount của tài khoản bạn khác, chỉ sửa đường dẫn; không đổi nội dung/hash.
    '''),cell('markdown','''
    ## Setup và preflight
    Không cài lại PyTorch; không tải pretrained model. Chỉ clone code nền đã pin rồi áp source overlay có hash.
    Tests model/DDP dùng dữ liệu tổng hợp. Chỉ sau các tests mới chuẩn bị features và huấn luyện thật.
    '''),cell('code',f'''
    import base64, hashlib, json, os, shutil, subprocess, sys, zlib
    os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      PYTHONHASHSEED='42', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    REPO = Path('/tmp/Image_Captioning-phrase-' + ARM.lower())
    LOG = Path('/tmp/phrase_boot_' + ARM.lower() + '.log')
    STAGE = Path('/tmp/phrase_stage_' + ARM.lower() + '.npz')
    CONFIG = REPO / 'phrase_config.json'
    def run(args,cwd=None):
        args=list(map(str,args));line='Running: '+' '.join(args)
        print(line,flush=True)
        with LOG.open('a',encoding='utf-8') as log:
            log.write(line+'\\n');log.flush()
            process=subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        if code:raise subprocess.CalledProcessError(code,args)
    for p in (CHECKPOINT,COCO_JSON):
        if not p.is_file():raise FileNotFoundError(p)
    for p in (VISUAL_CACHE,COCO_IMAGES):
        if not p.is_dir():raise FileNotFoundError(p)
    if OUTPUT.exists() or STAGE.exists():raise FileExistsError('Use a fresh OUTPUT/STAGE or fresh session; do not overwrite a completed run.')
    import torch
    assert torch.cuda.device_count()==2,'Select GPU T4 x2'
    assert all(torch.cuda.get_device_name(i)=='Tesla T4' for i in (0,1)),'Locked pilot uses two T4 GPUs'
    if not REPO.exists():run(['git','clone','--no-checkout','https://github.com/Supzxjee/Image_Captioning.git',REPO])
    run(['git','fetch','--depth','1','origin','{BASE}'],cwd=REPO)
    run(['git','checkout','--detach','{BASE}'],cwd=REPO)
    run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow'])
    PAYLOAD_B64 = {base64.b64encode(zlib.compress(payload,9)).decode()!r}
    PAYLOAD_SHA = {hashlib.sha256(payload).hexdigest()!r}
    SOURCE_HASHES = {hashes!r}
    raw=zlib.decompress(base64.b64decode(PAYLOAD_B64,validate=True));assert hashlib.sha256(raw).hexdigest()==PAYLOAD_SHA
    packet=json.loads(raw)
    for name,source in packet['sources'].items():
        assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_HASHES[name]
        (REPO/name).write_text(source,encoding='utf-8')
    MANIFEST=REPO/'locked_manifest.json';MANIFEST.write_text(packet['manifest'],encoding='utf-8')
    cfg=dict(arm=ARM,checkpoint=str(CHECKPOINT),visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),
             coco_images=str(COCO_IMAGES),output=str(OUTPUT),stage=str(STAGE),manifest=str(MANIFEST),
             manifest_sha256=hashlib.sha256(packet['manifest'].encode()).hexdigest(),source_sha256=SOURCE_HASHES,
             project_commit='{BASE}',payload_sha256=PAYLOAD_SHA)
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    run([sys.executable,'-u','test_phrase_data.py'],cwd=REPO)
    run([sys.executable,'-u','test_phrase_model.py'],cwd=REPO)
    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','run_phrase.py','smoke','--config',CONFIG],cwd=REPO)
    '''),cell('markdown','''
    ## Chuẩn bị38 ảnh và huấn luyện một nhánh
    Không copy toàn bộ visual-cache. Chỉ stage38 ảnh; global memory dùng đường visual frozen không prompt.
    Kiểm tra checkpoint/vocab/hash/ảnh gốc trước training. Lưu hash model/optimizer sau60 bước để đối chiếu hai nhánh.
    Không chọn checkpoint theo holdout. End checkpoint duy nhất, 300 updates.
    '''),cell('code','''
    run([sys.executable,'-u','run_phrase.py','prepare','--config',CONFIG],cwd=REPO)
    shutil.copy2(LOG,OUTPUT/'setup.log')
    (OUTPUT/'source').mkdir()
    for name in SOURCE_HASHES:shutil.copy2(REPO/name,OUTPUT/'source'/name)
    try:
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','run_phrase.py','worker','--config',CONFIG],cwd=REPO)
        run([sys.executable,'-u','run_phrase.py','finish','--config',CONFIG],cwd=REPO)
    finally:
        shutil.copy2(LOG,OUTPUT/'run.log')
        run([sys.executable,'-u','run_phrase.py','archive','--config',CONFIG],cwd=REPO)
    print('SEND:',str(OUTPUT)+'_review.zip',flush=True)
    print('KEEP adaptation separately:',OUTPUT/'adaptation.bin',flush=True)
    print('No semantic correctness claim until free captions are reviewed.',flush=True)
    '''),cell('markdown','''
    ## Gửi kết quả
    Gửi review ZIP của CE và CROSS. ZIP có scores, caption trước/sau on/off, ảnh+box và log; không chứa weights/features.
    `adaptation.bin` chỉ chứa tham số học mới, cần checkpoint nền và loader riêng khi tái dùng; không dùng thay checkpoint caption thông thường.
    Điểm agreement không phải CIDEr hoặc độ đúng caption. Bộ holdout chỉ7 nhóm đã review, không phải test độc lập.
    ''')]
    for i,c in enumerate(cells):c['id']=f'phrase-{arm.lower()}-{i:02}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),cells=cells)
    raw=json.dumps(nb,ensure_ascii=False,indent=1).encode();assert len(raw)<900000
    path=HERE/f'PHRASE_DECODER_{arm}_2GPU.ipynb';path.write_bytes(raw);print(path.name,len(raw),'bytes; NOT executed')


if __name__=='__main__':
    for arm in ('CE','CROSS'):build(arm)
