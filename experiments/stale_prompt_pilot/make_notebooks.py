"""Package one C or T arm per two-GPU notebook. Never launches it."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
s=importlib.util.spec_from_file_location('nb',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(s);s.loader.exec_module(nb);nb.HERE=HERE
nb.COMMIT=subprocess.check_output(['git','rev-parse','335b136'],cwd=ROOT,text=True).strip()
assert len(nb.COMMIT)==40


def sources():
    files={n:(HERE/n).read_text(encoding='utf-8') for n in ('run_stale.py','pilot_contract.py','test_contract.py','test_pilot_ddp.py')}
    for folder,names in [('stale_prompt_preparation',['profile_stale.py','stale_data.py','test_stale_data.py','test_stale_runtime.py','run_prepare.py']),
        ('attention_geometry_diagnostic',['run_diagnostic.py','diagnostic_data.py']),
        ('upt_hidden_pair',['hidden_data.py','hidden_metric_smoke.py']),('upt_alignment_pilot',['pilot_data.py'])]:
        files.update({n:(ROOT/'experiments'/folder/n).read_text(encoding='utf-8') for n in names})
    for n,source in files.items():ast.parse(source,filename=n)
    return files


def build():
    bundle=sources();records={}
    for arm in ('C','T'):
        cells=[nb.md(f'''
            # Stale semantic prompt — nhánh {arm}, hai GPU
            **T4×2 · Internet ON · Save & Run All.** Mỗi notebook chỉ chạy một nhánh.
            C và T cùng seed/order,TRAIN10k,2.000aux,1.250updates,batch tổng8,LR5e-6.
            C:main CE +.1KL neo A +.1neutral CE +.1visual contrast; T thêm .1stale loss.
            Dùng nguyên cache đã chuẩn bị; không mã hóa lại ảnh hay prompt, không chọn ca active.
            Raw greedy A và end trên dev1000; checkpoint cuối duy nhất. Không SCST/sweep/test.
            **Argon chưa phản biện độc lập**; user đã đồng ý thử C/T có giới hạn sau quyết định Astra.
            Gắn output `ducanh2403/stale`, checkpoint A, prompt-cache .pt, visual-cache, MSCOCO JSON.
            Không cần ảnh gốc, prompt JSON hoặc UPT cache. Tự tìm receipt đúng trong thư mục con.
            Stage main inputs có thể tốn I/O đáng kể, cần khoảng8GiB trống; không nằm trong profile6phút.
            Ngân sách: tối đa2giờ staging RIÊNG +2giờ train/end greedy RIÊNG.
            Cài đặt, kiểm hash/preflight, sinh A ban đầu và CPU scoring nằm ngoài hai timer;
            toàn notebook có thể vượt4giờ. Đây không phải trần tổng2giờ hoặc ETA.
            ''')]
        cells.append(nb.code(f'''
            from pathlib import Path
            ARM='{arm}'
            CACHE_ROOT=Path('/kaggle/input/notebooks/ducanh2403/stale')
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            OUTPUT=Path('/kaggle/working/stale_prompt_CT_{arm}')
            STAGE=Path('/tmp/stale_prompt_CT_{arm}_main.h5')
            for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON):assert p.is_file(),p
            for p in (CACHE_ROOT,VISUAL_CACHE):assert p.is_dir(),p
            assert not OUTPUT.exists() and not STAGE.exists(),'Preserve previous output; use fresh paths/session.'
            '''))
        cells.append(nb.bootstrap(bundle,'stale-ct-'+arm.lower()))
        cells.append(nb.code('''
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2','h5py','numpy','Pillow','tqdm','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version'])
            CHAIR=Path('/tmp/stale-ct-chair')
            checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
            import torch
            assert torch.cuda.device_count()==2,'Select T4x2 for each one-arm notebook.'
            run([sys.executable,'-u','test_contract.py'],cwd=REPO)
            run([sys.executable,'-u','test_stale_data.py'],cwd=REPO)
            run([sys.executable,'-u','test_stale_runtime.py'],cwd=REPO)
            run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','test_pilot_ddp.py'],cwd=REPO)
            '''))
        cells.append(nb.code('''
            cfg=dict(arm=ARM,cache_root=str(CACHE_ROOT),checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
                dataset_json=str(COCO_JSON),chair=str(CHAIR),output=str(OUTPUT),stage=str(STAGE),implementation_sha256=SOURCE_HASHES,project_commit=PROJECT_COMMIT)
            CONFIG=Path('/tmp/stale_ct_'+ARM+'.json');CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            PROTOCOL=OUTPUT/'protocol.json'
            try:
                run([sys.executable,'-u','run_stale.py','prepare','--config',CONFIG],cwd=REPO)
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0','run_stale.py','worker','--config',PROTOCOL],cwd=REPO)
                run([sys.executable,'-u','run_stale.py','evaluate','--config',PROTOCOL],cwd=REPO)
            finally:
                if OUTPUT.is_dir():
                    if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                    run([sys.executable,'-u','run_stale.py','archive','--config',PROTOCOL if PROTOCOL.is_file() else CONFIG],cwd=REPO)
            print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
            print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
            '''))
        cells.append(nb.md('''
            Gửi ZIP review của cả C và T; giữ riêng checkpoint.bin và cache gốc.
            ZIP có log/protocol/predictions/per-imageCIDEr/CHAIR, không có HDF5 hoặc checkpoint lớn.
            Một nhánh hoàn tất chưa đủ tuyên bố thắng: cần so T−A và T−C bằng pairedCI,
            trước đó kiểm A predictions của C/T trùng nhau cùng cohort/cache/optimizer protocol.
            Nếu A lệch, giải thích runtime/baseline trước khi quy khác biệt cho stale loss.
            CHAIR/recall và kiểm prompt abandonment/artifact trước claim cơ chế.
            Dừng nếu hash/binding/frozen state lỗi,nonfinite/OOM hoặc projection thực vượt2h/arm.
            Không dùng interim dev để sửa loss/chọn checkpoint/đổi seed.
            '''))
        name=f'STALE_PROMPT_{arm}_2GPU.ipynb';nb.write(name,cells);path=HERE/name
        records[arm]=dict(file=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (HERE/'PACKAGE.json').write_text(json.dumps(dict(status='PREPARED_UNEXECUTED',notebooks=records,project_commit=nb.COMMIT,
        sources_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()},local_model_execution=False,independent_Argon_review='pending'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':build()
