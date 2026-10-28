"""Package a small CPU cache-copy notebook. Never submit or execute Kaggle jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CORE='dd49cbb'


def cell(kind,text):
    result=dict(cell_type=kind,metadata={},source=textwrap.dedent(text).strip()+'\n')
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def main():
    commit=subprocess.check_output(['git','rev-parse',CORE],cwd=ROOT,text=True).strip()
    paths={'prepare_frames_cpu.py':'experiments/relation_frames_inventory/prepare_frames_cpu.py',
           'visual_cache.py':'captioning/visual_cache.py','semantic_prompts.py':'captioning/semantic_prompts.py',
           'relation_frames_manifest.bin':'experiments/relation_frames_inventory/V2_MANIFEST.json.gz'}
    assets={name:dict(path=path,sha256=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)).hexdigest()) for name,path in paths.items()}
    cells=[cell('markdown','''
        # Chuẩn bị cache quan hệ VSR v2 — CPU
        **Accelerator: None · Internet ON · Save & Run All. Không GPU, không train.**

        Chỉ sao chép feature CLIP và embedding prompt **gốc** đã có cho612ảnh.
        Không tạo detector, CLIP hoặc captioner; không đổi nhãn VSR. Audit nhãn vẫn chờ người xác nhận.

        Add Input ba dataset: `visual-cache`, `prompt-cache`, `objectdetectionecache`.
        Sửa ba đường dẫn ở cell cấu hình nếu Kaggle mount khác; không đổi nguồn prompt sang UPT/VSR.
        Dùng khoảng1GiB output trống; RAM cần đủ để mở cache prompt đầy đủ, tương tự preparation trước.
        Không cần checkpointA, ảnhRGB, API token hoặc dataset VSR riêng cho bước này.
        '''),cell('code','''
        from pathlib import Path
        VISUAL=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        LEGACY=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
        OUTPUT=Path('/kaggle/working/relation_frames_v2_cache')
        REVIEW=Path('/kaggle/working/relation_frames_v2_cache_review.zip')
        '''),cell('code',f'''
        import hashlib,json,os,subprocess,sys,tempfile,urllib.request,zipfile
        os.environ['CUDA_VISIBLE_DEVICES']=''
        SOURCE_COMMIT={commit!r}
        ASSETS={assets!r}
        SOURCE=Path(tempfile.mkdtemp(prefix='relation-frames-v2-cpu-'))
        LOG=Path('/kaggle/working/relation_frames_v2_cache.log')
        if not VISUAL.is_dir():raise FileNotFoundError(VISUAL)
        for p in (LEGACY,DETECTIONS):
            if not p.is_file():raise FileNotFoundError(p)
        if OUTPUT.exists():raise FileExistsError('Giữ output cũ; đổi sang runtime/output mới trước khi chạy lại.')
        for name,entry in ASSETS.items():
            url='https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+SOURCE_COMMIT+'/'+entry['path']
            print('Fetching pinned source:',name,flush=True)
            raw=urllib.request.urlopen(url,timeout=60).read()
            if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Source hash differs: '+name)
            (SOURCE/name).write_bytes(raw)
        print('Pinned source verified. This job copies caches on CPU; no model forward.',flush=True)
        '''),cell('code','''
        args=[sys.executable,'-u',str(SOURCE/'prepare_frames_cpu.py'),
              '--manifest',str(SOURCE/'relation_frames_manifest.bin'),'--visual',str(VISUAL),
              '--legacy',str(LEGACY),'--detections',str(DETECTIONS),'--output',str(OUTPUT)]
        returncode=None
        try:
            with LOG.open('w',encoding='utf-8') as log:
                process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                returncode=process.wait()
        finally:
            with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
                if LOG.is_file():z.write(LOG,'notebook.log')
                if OUTPUT.is_dir():
                    for p in OUTPUT.iterdir():
                        if p.is_file() and p.suffix in ('.json','.bin'):z.write(p,p.name)
                z.writestr('execution_status.json',json.dumps(dict(returncode=returncode,source_commit=SOURCE_COMMIT,
                    neural_forward=False,caption_training=False,training_approved=False),indent=2))
            print('SEND SMALL REVIEW ZIP:',REVIEW,flush=True)
        if returncode!=0:raise RuntimeError('CPU cache preparation failed. Send review ZIP/log; do not substitute prompts or relabel data.')
        receipt=json.loads((OUTPUT/'receipt.json').read_text())
        if receipt['status']!='CACHE_COPY_COMPLETE_PENDING_LABEL_AUDIT':raise RuntimeError('Missing completion receipt')
        print('KEEP ENTIRE FOLDER:',OUTPUT,flush=True)
        print('Prepared original caches only. Label audit and model/training preflight remain pending.',flush=True)
        '''),cell('markdown','''
        Giữ toàn bộ `relation_frames_v2_cache` để dùng lại; gửi `relation_frames_v2_cache_review.zip`.
        `relation_frames_manifest.bin` là metadata gzip giữ nguyên byte; dùng đuôi.bin để tránh Kaggle tự giải nén.
        Thành công của notebook chỉ xác nhận cache đã sao chép, chưa xác nhận chất lượng nhãn hoặc mô hình.
        ''')]
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(isGpuEnabled=False,isInternetEnabled=True)),cells=cells)
    target=HERE/'PREPARE_RELATION_FRAMES_V2_CPU.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],str(target),'exec')
    if target.stat().st_size>=1000000:raise ValueError('Notebook too large')
    package=dict(source_commit=commit,assets=assets,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 notebook_bytes=target.stat().st_size,static_code_compile=True,executed=False,neural_execution=False)
    (HERE/'PREPARATION_NOTEBOOK_PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes; static cells compile; NOT executed')


if __name__=='__main__':main()
