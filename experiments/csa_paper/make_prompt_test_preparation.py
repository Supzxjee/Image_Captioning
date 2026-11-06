"""Author CPU preparation notebook only, no user notebook execution."""
import argparse
import json
import subprocess
from pathlib import Path
import make_prompt_scst_notebooks as bundle
from make_smoke_notebook import cell,RUNNER
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    bundle.ENTRY=['prepare_prompt_test.py','test_prompt_test_preparation.py','make_prompt_test_preparation.py']
    bundle.ASSETS=['PROMPT_TEST_PROTOCOL.md','FULL_PROMPT_FRESH_V175_RESULT.json','RERANK_TEST_INPUTS.json','SAVE_FULL_ROI_V134_PACKAGE.json',
        'best9_saved_text.json.gz','BEST9_SAVED_TEXT_PACKAGE.json','SAVE_XE139_PACKAGE.json','full_roi_review_text/input_receipt.json']
    mapping,hashes=bundle.source_bundle(revision)
    setup='''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
BASE=Path('/kaggle/input/datasets')
COCO=BASE/'vuthetam/mscoco-2014'
VISUAL=BASE/'ducanh2403/visual-cache'
PROMPTS=BASE/'ducanh2403/upt-prompts-full-all-actions-v2/prompt_cache.json'
DETECTIONS=BASE/'ducanh2403/objectdetectionecache/objectdetectioncache.json'
PROMPT175=BASE/'ducanh2403/csa-prompt-full-karpathy-v175'
OUTPUT=Path('/kaggle/working/csa_prompt_test_preparation_v1')
for p in (COCO/'dataset_coco.json',VISUAL,PROMPTS,DETECTIONS,PROMPT175):
    if not p.exists():raise FileNotFoundError('Attach input/correct path: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve prior output; use fresh runtime')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-test-prepare-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
'''
    setup+='REVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
    setup+='''
for source,dest in MAPPING.items():
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+source,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=HASHES[source]:raise ValueError('Changed pinned file:'+source)
    p=SOURCE/dest;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
DEPS=SOURCE/'deps';os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
os.environ['OMP_NUM_THREADS']='2'
CONFIG=SOURCE/'config.json'
CONFIG.write_text(json.dumps(dict(output=str(OUTPUT),dataset_json=str(COCO/'dataset_coco.json'),detections=str(DETECTIONS),
    visual_cache=str(VISUAL),image_root=str(COCO/'images' if (COCO/'images').is_dir() else COCO),
    prompts=str(PROMPTS),prompt_root=str(PROMPT175),hf_cache=str(SOURCE/'hf-cache'))),encoding='utf-8')
'''
    run='''complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_test_preparation.py'],timeout=120)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'transformers==4.57.1','tokenizers==0.22.1','huggingface-hub==0.36.2','safetensors==0.6.2'],timeout=300)
    run([sys.executable,'-u',SOURCE/'prepare_prompt_test.py','--config',CONFIG],timeout=7200)
    complete=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))['status']=='PROMPT_TEST_PREPARATION_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,caption_generation=False,caption_training=False,caption_scoring=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt') and p.name!='test_metadata.json':z.write(p,p.name)
    print('SEND REVIEW:',REVIEW,'\\nKEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('PREPARATION COMPLETE; send review before caption evaluation.' if complete else 'INCOMPLETE; send log/review, no substitute prompt sources.',flush=True)
'''
    intro='''# TEST5000 — chuẩn bị ROI và prompt, chưa đánh giá caption
**Accelerator None**, Internet ON. Không cần Secret/checkpoint. CPU pooling từ visual-cache, không chạy lại CLIP ảnh/UPT; chỉ encode CPU các text chưa có trong175.
Gắn5Dataset: COCO2014(cảJPEG+JSON), visual-cache, upt-prompts-full-all-actions-v2, objectdetectionecache, csa-prompt-full-karpathy-v175. Chỉnh path ở cell đầu nếu mount khác.
Giữ nguyên KarpathyTEST5000, ROI/profile/vocab và ngưỡng cũ. Không dùng caption tham chiếu để tạo prompt; prompt thiếu ảnh sẽ dừng. Metadata tham chiếu chỉ giữ riêng cho evaluator sau này.
Giữ cả output và gửi `csa_prompt_test_preparation_v1_review.zip`. Bước này cần trước vì cache175 không chứaTEST. Sau khi xác minh receipt/coverage mới chuẩn bị lượt caption TEST C180/P60 và normal/zero/shuffle theo protocol đã khóa.
Tác giả chỉ kiểm metadata/static/fixture; chưa chạy notebook hoặc model. Không tiêu GPU nếu chọn Accelerator None.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',intro),cell('code',setup),cell('code',RUNNER),cell('code',run)])
    p=HERE/'CSA_PROMPT_TEST_PREPARATION_CPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(p.name,len(mapping),'pins',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
