"""Build two manual Kaggle notebooks. No notebook, encoder or training execution."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'phrase_decoder_pilot'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(HERE))
from make_notebooks import cell
from pair_data import experiment_manifest

BASE='6e2f9ef8553c7a9637916a7cf0ec3a90697b4c33'
OLD_NAMES=('phrase_data.py','phrase_model.py','run_phrase.py','test_phrase_model.py')
NEW_NAMES=('pair_data.py','pair_model.py','run_pair.py','test_pair_model.py','crop_geometry.py','test_preparation.py','test_pair_data.py','inspect_visual_metadata.py','audit_cache_metadata.py')


def packet():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    m=experiment_manifest(read(HERE/'LOCKED_DATA.json'),read(OLD/'locked_manifest.json'))
    manifest=json.dumps(m,indent=2,ensure_ascii=False)
    (HERE/'PAIR_MANIFEST.json').write_text(manifest,encoding='utf-8')
    src={n:(OLD/n).read_text(encoding='utf-8') for n in OLD_NAMES}
    src.update({n:(HERE/n).read_text(encoding='utf-8') for n in NEW_NAMES})
    for s in src.values():ast.parse(s)
    raw=json.dumps(dict(sources=src,manifest=manifest),ensure_ascii=False,separators=(',',':')).encode()
    return raw,{n:hashlib.sha256(s.encode()).hexdigest() for n,s in src.items()}


def build(variant,raw,hashes):
    initial='''
    CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
    COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
    '''
    if variant=='FULLFRAME':initial+='''
    VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
    COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
    SHARED=Path('/kaggle/working/pair_crop_shared')
    '''
    else:initial+='''
    # SỬA thành folder chứa shared_receipt.json và feature_cache.bin từ notebook FULLFRAME đã chạy xong.
    SHARED=Path('/kaggle/input/datasets/ducanh2403/pair-crop-shared/pair_crop_shared')
    COCO_IMAGES=SHARED/'images'
    VISUAL_CACHE=Path('/tmp/not_used_in_paircrop')
    '''
    cells=[cell('markdown',f'''
    # {variant}–CE — one arm, two T4 GPUs, 300 updates
    Người dùng tự chạy. Notebook chưa được chạy bởi tác giả.
    Hai nhánh mới cùng baseline/CE/schedule/initialization; khác nguồn patch và tọa độ patch tương ứng.
    14fit,7exposeddevelopment,8new-to-adaptation probe(4skate+4surf), không phải test độc lập toàn hệ thống.
    FULLFRAME chạy trước: tạo cachefull/crop/G chung và chặn train nếu featurecompatibility thấtbại.
    PAIRCROP chạy sau: mount nguyên folder shared, không tải CLIP/trích feature lại.
    KhôngCROSS/SCST/routing. Không dùng adapterCE/CROSS cũ hay checkpointF. BậtInternet, chọnT4×2,Save&RunAll.
    '''),cell('code',f'''
    print('PAIR SOURCE {variant}: START',flush=True)
    from pathlib import Path
    VARIANT={variant!r}
    {initial}
    OUTPUT=Path('/kaggle/working/pair_source_'+VARIANT.lower()+'_ce')
    '''),cell('code',f'''
    import base64,hashlib,json,os,shutil,subprocess,sys,zlib
    os.environ.update(PYTHONUNBUFFERED='1',HDF5_USE_FILE_LOCKING='FALSE',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    REPO=Path('/tmp/Image_Captioning-pair-source-'+VARIANT.lower());LOG=Path('/tmp/pair_source_'+VARIANT.lower()+'.log');CONFIG=REPO/'pair_config.json'
    def run(args,cwd=None):
        args=list(map(str,args));print('Running:',' '.join(args),flush=True)
        with LOG.open('a',encoding='utf-8') as log:
            process=subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        if code:raise subprocess.CalledProcessError(code,args)
    for p in (CHECKPOINT,COCO_JSON):
        if not p.is_file():raise FileNotFoundError(p)
    if VARIANT=='FULLFRAME':
        assert VISUAL_CACHE.is_dir() and COCO_IMAGES.is_dir()
        assert not SHARED.exists(),'Use fresh shared folder/session'
    else:
        for name in ('feature_cache.bin','shared_receipt.json','fullframe_complete_rank0.json','fullframe_protocol.json'):
            assert (SHARED/name).is_file(),f'Wrong shared path or unfinished FULLFRAME: {{SHARED/name}}'
    assert not OUTPUT.exists(),'Use a fresh OUTPUT; no overwrite'
    import torch
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_name(i)=='Tesla T4' for i in (0,1)),'Select T4 x2'
    if not REPO.exists():run(['git','clone','--no-checkout','https://github.com/Supzxjee/Image_Captioning.git',REPO])
    run(['git','fetch','--depth','1','origin','{BASE}'],REPO);run(['git','checkout','--detach','{BASE}'],REPO)
    run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow'])
    PAYLOAD={base64.b64encode(zlib.compress(raw,9)).decode()!r}
    raw=zlib.decompress(base64.b64decode(PAYLOAD,validate=True));assert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
    packet=json.loads(raw);hashes={hashes!r}
    for name,source in packet['sources'].items():
        assert hashlib.sha256(source.encode()).hexdigest()==hashes[name]
        (REPO/name).write_text(source,encoding='utf-8')
    manifest=REPO/'locked_manifest.json';manifest.write_text(packet['manifest'],encoding='utf-8')
    cfg=dict(arm='CE',variant=VARIANT,checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),coco_images=str(COCO_IMAGES),
        shared=str(SHARED),output=str(OUTPUT),manifest=str(manifest),manifest_sha256=hashlib.sha256(packet['manifest'].encode()).hexdigest(),source_sha256=hashes,project_commit='{BASE}')
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    run([sys.executable,'-u','test_preparation.py'],REPO)
    run([sys.executable,'-u','test_pair_data.py'],REPO)
    run([sys.executable,'-u','test_pair_model.py'],REPO)
    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','run_pair.py','smoke','--config',CONFIG],REPO)
    '''),cell('code','''
    try:
        run([sys.executable,'-u','run_pair.py','prepare','--config',CONFIG],REPO)
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','run_pair.py','worker','--config',CONFIG],REPO)
        run([sys.executable,'-u','run_pair.py','finish','--config',CONFIG],REPO)
    finally:
        if OUTPUT.is_dir():
            shutil.copy2(LOG,OUTPUT/'run.log')
            for name in ('compatibility.json','original_metadata.json'):
                if (SHARED/name).is_file():shutil.copy2(SHARED/name,OUTPUT/name)
            run([sys.executable,'-u','run_pair.py','archive','--config',CONFIG],REPO)
    print('SEND AFTER BOTH ARMS:',str(OUTPUT)+'_review.zip',flush=True)
    print('KEEP adapter separately:',OUTPUT/'adaptation.bin',flush=True)
    if VARIANT=='FULLFRAME':print('NEXT: preserve/upload WHOLE shared folder including images:',SHARED,flush=True)
    '''),cell('markdown','''
    ## Kết quả và bước tiếp
    FULLFRAME: lưu nguyên /kaggle/working/pair_crop_shared thành dataset (feature_cache.bin,receipt,images và fullframe_* references).
    Dữ liệu nhỏ, chỉ46ảnh/58cropviews; không copy hay mã hóa lại123287ảnh.
    Sửa SHARED trong PAIRCROP thành mountthực rồi chạy; đừng thay treatment sau khi xem outputprobe của control.
    Gửi cả pair_source_fullframe_ce_review.zip và pair_source_paircrop_ce_review.zip. ZIP review không chứa weights/features/images;
    shared folder giữ ảnh để tạo review mù sau khi cảhai hoàn tất. Giữ adaptation.bin riêng, không dùng như checkpoint caption thường.
    Primary metric cần chấm mù ngữ nghĩa sinh tựdo của đúng cặp. Ranking/xácsuất/exactmatch không thay semanticaccuracy.
    Nếu featuregate fail, gửi FULLFRAME reviewZIP + pair_crop_shared/compatibility.json; không bỏassert/nới tolerance.
    ''')]
    for i,c in enumerate(cells):c['id']=f'pair-{variant.lower()}-{i}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),cells=cells)
    data=json.dumps(nb,indent=1,ensure_ascii=False).encode();assert len(data)<900000
    path=HERE/f'PAIR_SOURCE_{variant}_CE_2GPU.ipynb';path.write_bytes(data);print(path.name,len(data),'bytes; NOT executed')


if __name__=='__main__':
    raw,hashes=packet()
    for variant in ('FULLFRAME','PAIRCROP'):build(variant,raw,hashes)
