"""Generate small, pinned, user-run notebooks. Never submit a Kaggle job."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def cell(kind,text):
    c=dict(cell_type=kind,metadata={},source=textwrap.dedent(text).strip()+'\n')
    if kind=='code':c.update(execution_count=None,outputs=[])
    return c


def main(core):
    commit=subprocess.check_output(['git','rev-parse',core],cwd=ROOT,text=True).strip()
    names=['frame_data.py','frame_model.py','run_frames.py','compare_frames.py','test_frame_data.py']
    paths={n:'experiments/relation_frame_pretrain/'+n for n in names}
    paths.update({'scst_runtime.py':'captioning/scst_runtime.py','gain_metadata.bin':'experiments/semantic_grounding_method/data/gain_metadata.json.gz'})
    assets={n:dict(path=p,sha256=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)).hexdigest()) for n,p in paths.items()}
    packages=[]
    for arm in ('RelPos','RelContrast'):
        cells=[cell('markdown',f'''
            # Relation frames — {arm}, 200 updates, T4 ×2
            **Internet ON · GPU T4 ×2 · Save & Run All. Một notebook cho mỗi tài khoản.**

            Add Input: `ducanh2403/relation-frames-v2-cache-v105` và dataset chứa checkpoint A
            `ducanh2403/pair-source-fullframe-ce`. Không cần tải lại ảnh, CLIP, UPT hoặc prompt gốc.

            Hai nhánh dùng cùng kiến trúc, initialization, 256 train pairs, 64 held-out pairs và thứ tự batch.
            RelPos học mô tả đúng; RelContrast thêm truth/paired loss cho đúng predicate được gắn nhãn.
            Đây là frame pretraining, chưa train caption bằng SCST và chưa chứng minh cải thiện caption.

            Trước train: full input SHA, token/box checks, model/gradient/frozen-state và two-rank optimizer checks.
            Chỉ chấm checkpoint200; không tự chọn epoch. Trần mỗi nhánh **.5 device-GPU-hour**
            (2 GPU tương ứng tối đa15phút ở pha GPU, gồm preflight/train/eval; đây là cap, không phải ETA).
            Nếu chạm cap hoặc test fail, dừng và gửi review; không tự nới cap/test.
            '''),cell('code',f'''
            from pathlib import Path
            ARM={arm!r}
            CACHE=Path('/kaggle/input/datasets/ducanh2403/relation-frames-v2-cache-v105')
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
            OUTPUT=Path('/kaggle/working/relation_frames_'+ARM)
            DEVICE_GPU_HOURS_CAP=.5
            '''),cell('code',f'''
            import hashlib,json,os,subprocess,sys,tempfile,threading,time,urllib.request,zipfile
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            os.environ['OMP_NUM_THREADS']='1'
            SOURCE_COMMIT={commit!r}
            ASSETS={assets!r}
            SOURCE=Path(tempfile.mkdtemp(prefix='relation-frames-'+ARM+'-'))
            LOG=Path('/kaggle/working/relation_frames_'+ARM+'.log')
            REVIEW=Path('/kaggle/working/relation_frames_'+ARM+'_review.zip')
            if not CACHE.is_dir():raise FileNotFoundError(CACHE)
            if not CHECKPOINT.is_file():raise FileNotFoundError(CHECKPOINT)
            if OUTPUT.exists():raise FileExistsError('Output cũ được giữ nguyên; dùng runtime/output mới.')
            for name,entry in ASSETS.items():
                print('Fetching pinned source:',name,flush=True)
                url='https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+SOURCE_COMMIT+'/'+entry['path']
                raw=urllib.request.urlopen(url,timeout=60).read()
                if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Source hash differs: '+name)
                (SOURCE/name).write_bytes(raw)
            sys.path.insert(0,str(SOURCE))
            from frame_data import FILES,CHECKPOINT_SHA,META_SHA
            CONFIG=SOURCE/'config.json'
            cfg=dict(arm=ARM,cache=str(CACHE),checkpoint=str(CHECKPOINT),metadata=str(SOURCE/'gain_metadata.bin'),
                output=str(OUTPUT),updates=200,seed=42,device_gpu_hours_cap=DEVICE_GPU_HOURS_CAP,
                source_commit=SOURCE_COMMIT,source_assets=ASSETS,cache_sha256=FILES,
                checkpoint_sha256=CHECKPOINT_SHA,metadata_sha256=META_SHA,
                global_batch_pairs=16,learning_rate=.0001,weight_decay=.01,gradient_clip=1.,
                frame_dim=512,entity_dim=256,gru_hidden=256,output_head='trainable Linear256->10000',
                word_embedding='frozen original A',loss_weights=dict(positive=1.,truth=.5,pair=.5),
                model_preflight_atol=.00001,model_preflight_rtol=.00001)
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            print('Source and protocol ready; no job submitted by assistant.',flush=True)
            '''),cell('code','''
            def run(args):
                print('Running:', ' '.join(map(str,args)),flush=True)
                with LOG.open('a',encoding='utf-8') as log:
                    process=subprocess.Popen(list(map(str,args)),cwd=SOURCE,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,bufsize=1)
                    stop=threading.Event()
                    def heartbeat():
                        while not stop.wait(30):
                            print('Worker active:',ARM,'| watch step/preflight lines for actual progress.',flush=True)
                    beat=threading.Thread(target=heartbeat,daemon=True);beat.start()
                    try:
                        for line in process.stdout:
                            print(line,end='',flush=True);log.write(line);log.flush()
                        code=process.wait()
                    finally:stop.set();beat.join(timeout=1)
                if code:raise subprocess.CalledProcessError(code,args)

            success=False
            try:
                run([sys.executable,'-u','test_frame_data.py'])
                run([sys.executable,'-u','run_frames.py','prepare','--config',CONFIG])
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                    '--nproc_per_node=2','--max_restarts=0','run_frames.py','run','--config',CONFIG])
                result=json.loads((OUTPUT/'result.json').read_text())
                if result['status']!='ENDPOINT_200_COMPLETE':raise RuntimeError('Endpoint incomplete')
                success=True
            finally:
                with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
                    if LOG.is_file():z.write(LOG,'notebook.log')
                    z.write(CONFIG,'notebook_config.json')
                    for name in ASSETS:
                        if name.endswith('.py'):z.write(SOURCE/name,'source/'+name)
                    if OUTPUT.is_dir():
                        for path in OUTPUT.iterdir():
                            if path.suffix in ('.json','.jsonl'):z.write(path,path.name)
                    z.writestr('execution_status.json',json.dumps(dict(success=success,arm=ARM,
                        source_commit=SOURCE_COMMIT,assistant_launched_job=False),indent=2))
                print('SEND SMALL REVIEW ZIP:',REVIEW,flush=True)
                if (OUTPUT/'checkpoint.bin').is_file():print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
                if not success:print('INCOMPLETE. Send ZIP/log; no result accepted and no caption training.',flush=True)
            if success:print('DONE',ARM,'| send both arms for paired comparison before further training.',flush=True)
            '''),cell('markdown','''
            Gửi `relation_frames_RelPos_review.zip` và `relation_frames_RelContrast_review.zip` khi hai lượt xong.
            Giữ `checkpoint.bin` của cả hai. Notebook không tạo Dataset hoặc chạy caption stage tự động.
            Cap là ngân sách device-hours; giới hạn Kaggle thực tế có thể tính khác, nên kiểm tra quota tài khoản trước khi chạy.
            ''')]
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),cells=cells)
        path=HERE/('PRETRAIN_'+arm.upper()+'_2GPU.ipynb');path.write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        for c in cells:
            if c['cell_type']=='code':compile(c['source'],str(path),'exec')
        assert path.stat().st_size<1000000
        packages.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE/'NOTEBOOK_PACKAGE.json').write_text(json.dumps(dict(source_commit=commit,assets=assets,notebooks=packages,
        static_code_compile=True,local_neural_execution=False,kaggle_launched=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(packages,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',required=True);main(p.parse_args().core)
