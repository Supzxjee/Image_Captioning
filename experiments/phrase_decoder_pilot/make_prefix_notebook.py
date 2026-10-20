"""Package a manual evaluation-only notebook; never execute a model/notebook."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from make_notebooks import HERE,sources,cell
from compare_reviews import read_review,compare

BASE='d772e6854b55f88abd9d7fc45465181fc7d1726d'


def build():
    paths={arm:Path('C:/Users/leduc/Downloads')/f'phrase_decoder_{arm.lower()}_21_review.zip' for arm in ('CE','CROSS')}
    loaded={arm:read_review(path) for arm,path in paths.items()};compare(loaded['CE'],loaded['CROSS'])
    src=sources();src['run_prefix_diagnostic.py']=(HERE/'run_prefix_diagnostic.py').read_text(encoding='utf-8')
    src['test_prefix_data.py']=(HERE/'test_prefix_data.py').read_text(encoding='utf-8')
    expected={}
    for arm,r in loaded.items():
        expected[arm]={k:r['cfg'][k] for k in ('protocol_sha256','manifest_sha256','stage_tensor_sha256')}
        expected[arm].update(arm=arm,initial_sha256=r['summary']['initial_sha256'],final_sha256=r['summary']['final_sha256'],
            records={v['group_id']:v for v in r['phases']['after'] if next(g for g in r['manifest']['groups'] if g['group_id']==v['group_id'])['same_image']})
    packet=dict(sources=src,manifest=(HERE/'locked_manifest.json').read_text(encoding='utf-8'),expected=expected)
    raw=json.dumps(packet,ensure_ascii=False,separators=(',',':')).encode();hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in src.items()}
    cells=[cell('markdown','''
    # Pair-prefix diagnostic — evaluation only, CE + CROSS, T4 ×2
    Không training, optimizer, SCST hay cập nhật weights. GPU0 đọc CE; GPU1 đọc CROSS độc lập, không DDP collectives.
    Chỉ kiểmtra 4 nhóm cùng ảnh: 2holdout đã xem +2fit controls. Không phải test độc lập.
    Bắt buộc khôi phục đúng full-model hash và replay câu/score trước khi diễn giải can thiệp.
    **Cần hai file adaptation.bin từ OUTPUT lần chạy CE và CROSS, không nằm trong review ZIP.**
    Có thể đổi tên thành adaptation_ce.bin và adaptation_cross.bin khi tạo dataset. Đổi đuôi/đổi tên không đổi bytes.
    Inputs: checkpoint nền gốc SHA5f8..., visual-cache, MSCOCOJSON+ảnh, hai adapter. Không dùng F.bin.
    Chạy Save & Run All, InternetON, T4×2. Notebook này chưa được tác giả chạy model.
    '''),cell('code','''
    print('PAIR PREFIX DIAGNOSTIC: START / NO TRAINING',flush=True)
    from pathlib import Path
    CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
    VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
    COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
    COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
    # Sửa hai path nếu đã upload adapter thành dataset. KHÔNG trỏ vào checkpoint nền/F.
    ADAPTER_CE=Path('/kaggle/working/phrase_decoder_ce_21/adaptation.bin')
    ADAPTER_CROSS=Path('/kaggle/working/phrase_decoder_cross_21/adaptation.bin')
    OUTPUT=Path('/kaggle/working/pair_prefix_diagnostic_4')
    '''),cell('code',f'''
    import base64,hashlib,json,os,shutil,subprocess,sys,zlib
    os.environ.update(PYTHONUNBUFFERED='1',HDF5_USE_FILE_LOCKING='FALSE',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    REPO=Path('/tmp/Image_Captioning-pair-prefix');LOG=Path('/tmp/pair_prefix_boot.log');CONFIG=REPO/'prefix_config.json'
    def run(args,cwd=None):
        args=list(map(str,args));print('Running:',' '.join(args),flush=True)
        with LOG.open('a',encoding='utf-8') as log:
            process=subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        if code:raise subprocess.CalledProcessError(code,args)
    for p in (CHECKPOINT,COCO_JSON,ADAPTER_CE,ADAPTER_CROSS):
        if not p.is_file():raise FileNotFoundError(f'Missing {{p}}. Set adapter paths to the two saved adaptation.bin files; review ZIPs do not contain weights.')
    assert ADAPTER_CE.resolve()!=ADAPTER_CROSS.resolve(),'Need two different adapters; metadata/full hashes are checked below'
    for p in (VISUAL_CACHE,COCO_IMAGES):assert p.is_dir(),p
    assert not OUTPUT.exists(),'Use a fresh OUTPUT/session; no overwrite'
    import torch
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_name(i)=='Tesla T4' for i in (0,1)),'Select T4 x2'
    if not REPO.exists():run(['git','clone','--no-checkout','https://github.com/Supzxjee/Image_Captioning.git',REPO])
    run(['git','fetch','--depth','1','origin','{BASE}'],REPO)
    run(['git','checkout','--detach','{BASE}'],REPO)
    run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow'])
    PAYLOAD={base64.b64encode(zlib.compress(raw,9)).decode()!r}
    data=zlib.decompress(base64.b64decode(PAYLOAD,validate=True));assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
    packet=json.loads(data);hashes={hashes!r}
    for name,source in packet['sources'].items():
        assert hashlib.sha256(source.encode()).hexdigest()==hashes[name]
        (REPO/name).write_text(source,encoding='utf-8')
    manifest=REPO/'locked_manifest.json';manifest.write_text(packet['manifest'],encoding='utf-8')
    cfg=dict(arm='CE',checkpoint=str(CHECKPOINT),visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),coco_images=str(COCO_IMAGES),
        output=str(OUTPUT),stage='/tmp/pair_prefix_stage.npz',manifest=str(manifest),manifest_sha256=hashlib.sha256(packet['manifest'].encode()).hexdigest(),
        source_sha256=hashes,project_commit='{BASE}',adapters=dict(CE=str(ADAPTER_CE),CROSS=str(ADAPTER_CROSS)),expected=packet['expected'])
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    run([sys.executable,'-u','test_prefix_data.py'],REPO)
    run([sys.executable,'-u','test_phrase_model.py'],REPO)
    '''),cell('code','''
    # Frozen visual staging only: reuse the exact original preparation and tokenizer.
    run([sys.executable,'-u','run_phrase.py','prepare','--config',CONFIG],REPO)
    (OUTPUT/'source').mkdir()
    for name in hashes:shutil.copy2(REPO/name,OUTPUT/'source'/name)
    try:
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','run_prefix_diagnostic.py','worker','--config',CONFIG],REPO)
        run([sys.executable,'-u','run_prefix_diagnostic.py','finish','--config',CONFIG],REPO)
    finally:
        shutil.copy2(LOG,OUTPUT/'run.log')
        run([sys.executable,'-u','run_phrase.py','archive','--config',CONFIG],REPO)
    print('SEND:',str(OUTPUT)+'_review.zip',flush=True)
    '''),cell('markdown','''
    ## Diễn giải và gửi kết quả
    Gửi pair_prefix_diagnostic_4_review.zip. Có top10 token, branch ranks/logP, logP từng token teacher-forced,
    sinh từBOS / sau a person / sau prefixchung của hai mô tả; ảnh+box và log.
    Natural: đổi sang cặp thật trong cùngảnh, G/P giốngnhau. Evidenceoff: chỉlà thamchiếu nhạy với nhánh.
    Geometryswap: chỉtráo vector8D, giữ masks/membership; là canthiệp khôngnhấtquán/OOD, không chứngminh loại bỏ hếtgeometry.
    Sau a person, branch_tokens của phone là holding/looking, nhưng prefix chung thực của hai câu phone chỉlà a person.
    Nếu replay score/câu/hash thấtbại, dừng diễn giải và gửilog; không giảm tolerance hoặc bỏassert.
    Không có semanticaccuracy tựđộng; không huấn luyện hoặc đề xuất routing từ riêng kếtquả này.
    ''')]
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),cells=cells)
    for i,c in enumerate(cells):c['id']=f'pair-prefix-{i}'
    data=json.dumps(nb,ensure_ascii=False,indent=1).encode();assert len(data)<900000
    path=HERE/'PAIR_PREFIX_DIAGNOSTIC_2GPU.ipynb';path.write_bytes(data)
    print(path.name,len(data),'bytes; NOT executed')


if __name__=='__main__':build()
