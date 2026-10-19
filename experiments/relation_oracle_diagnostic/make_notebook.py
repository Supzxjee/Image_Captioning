"""Small self-contained manual inference notebook, never executes cells."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import textwrap
import zlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE='fa0087fe5b94b9b9db5bb0ba865683286e7e6e97'


def sources():
    spec=importlib.util.spec_from_file_location('verified_builder',ROOT/'experiments/verified_same_image/make_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.sources()
    result={k:old[k] for k in ('base_runner.py','pilot_model.py','evidence_core.py','pilot_data.py','probe_data.py','synthetic_fixtures.py')}
    result.update({name:(HERE/name).read_text(encoding='utf-8') for name in ('oracle_data.py','review_html.py','run_diagnostic.py','test_oracle_data.py','test_oracle_model.py')})
    return result


def code(text):
    text=textwrap.dedent(text).strip()+'\n';ast.parse(text)
    return dict(cell_type='code',source=text.splitlines(True),metadata={},outputs=[],execution_count=None)


def markdown(text):return dict(cell_type='markdown',source=textwrap.dedent(text).strip().splitlines(True),metadata={})


def build():
    files=sources();manifest=(HERE/'cases_manifest.json').read_text(encoding='utf-8')
    payload=json.dumps(dict(sources=files,manifest=manifest),ensure_ascii=False,separators=(',',':')).encode()
    hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in files.items()}
    cells=[markdown('''
    # Chẩn đoán đường quan hệ → bằng chứng → decoder: F, 20 ảnh
    **Không huấn luyện, không chọn checkpoint, không đánh giá hiệu năng benchmark.** Hai T4 chia nhau suy luận cùng F.
    Bốn điều kiện: tắt nhánh / UPT / annotation dương / annotation âm. Hai điều kiện annotation giữ cùng box GT và ba slot S/O/R, chỉ đổi text quan hệ.
    Nhãn âm V-COCO có thể vẫn mô tả hành động hợp lý; cần xem ảnh, không tự coi đây là đúng/sai tuyệt đối.
    Bạn tự chạy Save & Run All, Internet ON, GPU T4 ×2. Không cần CHAIR, object detections hay huấn luyện lại.
    ''')]
    cells.append(code('''
    print('F RELATION CONDITIONING DIAGNOSTIC: NO TRAINING',flush=True)
    from pathlib import Path
    # Nếu chạy ngay sau F trong cùng session, đường dẫn này có sẵn.
    # Nếu notebook mới: sửa thành đường dẫn checkpoint F trong Input, giữ đúng SHA4938bfe8...
    CHECKPOINT=Path('/kaggle/working/semantic_evidence_verified_pairs_F/checkpoint.pth')
    VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
    LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
    UPT_PROMPT=Path('/kaggle/input/datasets/ducanh2403/upt-prompts-full-all-actions-v2/prompt_cache.json')
    COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
    COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
    OUTPUT=Path('/kaggle/working/relation_conditioning_diagnostic_20')
    '''))
    setup=textwrap.dedent('''
    import json,os,subprocess,sys,hashlib,base64,zlib,shutil
    os.environ.update(PYTHONUNBUFFERED='1',HDF5_USE_FILE_LOCKING='FALSE',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_DISABLE_XET='1')
    REPO=Path('/tmp/Image_Captioning-relation-conditioning')
    LOG=Path('/tmp/relation_conditioning_setup.log')
    def run(args,cwd=None):
        print('Running:',' '.join(map(str,args)),flush=True)
        with LOG.open('a',encoding='utf-8') as log:
            with subprocess.Popen(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as process:
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                status=process.wait()
        if status:raise RuntimeError(f'Command failed ({status}); log: {LOG}')
    for path in (CHECKPOINT,LEGACY_PROMPT,UPT_PROMPT,COCO_JSON):assert path.is_file(),f'Copy actual mounted path: {path}'
    assert COCO_IMAGES.is_dir() and VISUAL_CACHE.is_dir()
    assert not OUTPUT.exists(),'Choose a fresh OUTPUT/restart session'
    import torch
    assert torch.cuda.device_count()==2,'Select T4 x2; one frozen checkpoint across both GPUs'
    if not REPO.exists():run(['git','clone','--no-checkout','https://github.com/Supzxjee/Image_Captioning.git',REPO])
    run(['git','fetch','--depth','1','origin','__BASE__'],cwd=REPO)
    run(['git','checkout','--detach','__BASE__'],cwd=REPO)
    run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow','tqdm'])
    ''').replace('__BASE__',BASE)
    setup+='\nPAYLOAD_B64 = '+repr(base64.b64encode(zlib.compress(payload,9)).decode())+'\n'
    setup+='PAYLOAD_SHA = '+repr(hashlib.sha256(payload).hexdigest())+'\n'
    setup+="raw=zlib.decompress(base64.b64decode(PAYLOAD_B64,validate=True))\nassert hashlib.sha256(raw).hexdigest()==PAYLOAD_SHA\nbundle=json.loads(raw)\nSOURCES=bundle['sources']\n"
    setup+='SOURCE_HASHES = '+repr(hashes)+'\n'
    setup+="for name,source in SOURCES.items():\n    assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_HASHES[name]\n    (REPO/name).write_text(source,encoding='utf-8')\n"
    setup+="MANIFEST=REPO/'cases_manifest.json'\nMANIFEST.write_text(bundle['manifest'],encoding='utf-8')\n"
    setup+="for row in json.loads(bundle['manifest'])['cases']:\n    assert any(p.is_file() for p in (COCO_IMAGES/row['filepath']/row['filename'],COCO_IMAGES/row['filename'])),f'Missing raw review image: {row[\"filename\"]}; check COCO_IMAGES before GPU inference'\n"
    setup+="OUTPUT.mkdir(parents=True)\n(OUTPUT/'source').mkdir()\nfor name,source in SOURCES.items():(OUTPUT/'source'/name).write_text(source,encoding='utf-8')\n"
    setup+="shutil.copy2(MANIFEST,OUTPUT/'cases_manifest.json')\n"
    setup+="cfg=dict(checkpoint=str(CHECKPOINT),visual_cache=str(VISUAL_CACHE),legacy_prompt=str(LEGACY_PROMPT),upt_prompt=str(UPT_PROMPT),dataset_json=str(COCO_JSON),coco_images=str(COCO_IMAGES),output=str(OUTPUT),manifest=str(MANIFEST),manifest_sha256=hashlib.sha256(bundle['manifest'].encode()).hexdigest(),source_sha256=SOURCE_HASHES,project_commit='"+BASE+"')\n"
    setup+="CONFIG=OUTPUT/'config.json'\nCONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')\n"
    cells.append(code(setup))
    cells.append(markdown('''
    ## Kiểm tra trước khi suy luận
    Kiểm tra cách chọn ca, giữ box/slot cố định và module frozen bằng dữ liệu giả. Chưa thực hiện huấn luyện.
    '''))
    cells.append(code('''
    for name in ('test_oracle_data.py','test_oracle_model.py'):run([sys.executable,'-u',REPO/name],cwd=REPO)
    print('SYNTHETIC PREFLIGHT PASS',flush=True)
    shutil.copy2(LOG,OUTPUT/'setup.log')
    LOG=OUTPUT/'run.log'
    '''))
    cells.append(markdown('''
    ## Suy luận bốn điều kiện và xuất review
    Hai worker độc lập, mỗi GPU 10 ảnh; không optimizer/backward/DDP collectives.
    Kiểm tra checkpoint/nhãn/cache bằng hash, replay điểm matcher F, xác nhận trọng số không đổi.
    ZIP gồm 20 ảnh, box, caption, likelihood và HTML để kiểm tra nhãn. Không tự chấm caption đúng chỉ bằng từ khóa.
    '''))
    cells.append(code('''
    try:
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',REPO/'run_diagnostic.py','worker','--config',CONFIG],cwd=REPO)
        run([sys.executable,'-u',REPO/'run_diagnostic.py','finish','--config',CONFIG],cwd=REPO)
    finally:
        run([sys.executable,'-u',REPO/'run_diagnostic.py','archive','--config',CONFIG],cwd=REPO)
    print('SEND:',str(OUTPUT)+'_review.zip',flush=True)
    print('REVIEW:',OUTPUT/'case_review_20.html',flush=True)
    '''))
    for i,cell in enumerate(cells):cell['id']=f'oracle-{i:02}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    raw=json.dumps(nb,ensure_ascii=False,indent=1).encode();assert len(raw)<900000
    path=HERE/'RELATION_CONDITIONING_DIAGNOSTIC_20_2GPU.ipynb';path.write_bytes(raw)
    print(path.name,len(raw),'bytes; cells not executed')


if __name__=='__main__':build()
