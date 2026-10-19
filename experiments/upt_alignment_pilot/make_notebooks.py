"""Build self-contained notebooks only. Does not execute notebook cells."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

HERE = Path(__file__).parent
COMMIT = '1dd4dc15f42eded978945a3da66c65ec6a725b59'
FILES = ['pilot_data.py','pilot_model.py','run_pilot.py','test_pilot_data.py','test_pilot_model.py',
         'test_pilot_ddp.py','pilot_metric_smoke.py']
DESCRIPTIONS = {'A':'Đối chứng: huấn luyện tiếp baseline', 'B':'Thêm vùng ảnh, không thêm văn bản UPT',
                'C':'Vùng ảnh + văn bản UPT, cộng đặc trưng trực tiếp',
                'D':'Cổng thị giác được điều kiện hóa bởi văn bản UPT'}


def code(source):
    source = textwrap.dedent(source).strip()+'\n'
    ast.parse(source)
    return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=source.splitlines(True))


def markdown(source):
    return dict(cell_type='markdown',metadata={},source=textwrap.dedent(source).strip().splitlines(True))


def main():
    sources = {name:(HERE/name).read_text(encoding='utf-8') for name in FILES}
    hashes = {name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    for arm, description in DESCRIPTIONS.items():
        cells = [markdown(f'''
        # UPT alignment pilot — nhánh {arm}: {description}

        **Notebook này chạy MỘT nhánh trên HAI GPU bằng DDP.** Chọn **GPU T4 ×2**, bật Internet rồi **Save & Run All**.
        Mã nguồn mới đã nhúng sẵn bên trong; không cần tải các tệp `.py` khác lên Kaggle.

        10.000 ảnh train + 1.000 ảnh val, cùng cohort ở cả bốn nhánh; không dùng test.
        1.250 cập nhật, batch toàn cục 8 (4/GPU); 200 cập nhật đầu CE, sau đó CE + SCST + thưởng đối tượng.
        Đây là phép thử sàng lọc, chưa chứng minh cải thiện caption hoặc tính mới của phương pháp.
        ''')]
        cells.append(code(f'''
        print('UPT ALIGNMENT {arm}: BOOTSTRAP START', flush=True)
        from pathlib import Path

        ARM = '{arm}'
        CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
        LEGACY_PROMPT = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        UPT_PROMPT = Path('/kaggle/input/datasets/ducanh2403/upt-prompts-full-all-actions-v2/prompt_cache.json')
        VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
        OUTPUT = Path('/kaggle/working/upt_alignment_pilot_{arm}')

        # Bạn của bạn sửa các mount ở trên nếu khác; giữ nguyên nội dung đầu vào.
        # Không cần COCO_IMAGES hay raw UPT cache: dùng visual HDF5 + prompt JSON đã có.
        '''))
        cells.append(markdown('## 1. Chuẩn bị môi trường và mã nguồn\nKho mã đặt trong `/tmp` để không chiếm danh sách Output. Không cài lại PyTorch của Kaggle.'))
        setup = r'''
        import json, os, shutil, subprocess, sys
        os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                          HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='120', TOKENIZERS_PARALLELISM='false',
                          PYTHONHASHSEED='42', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        REPO = Path('/tmp/Image_Captioning-upt-pilot')
        CHAIR = Path('/tmp/upt-pilot-chair')
        SETUP_LOG = Path('/tmp/upt_pilot_setup_' + ARM + '.log')
        PROJECT_COMMIT = '__COMMIT__'

        def run(args, cwd=None, log=None):
            args = list(map(str,args))
            print('\nRunning:', ' '.join(args), flush=True)
            with Path(log or SETUP_LOG).open('a',encoding='utf-8') as output:
                output.write('\nRunning: ' + ' '.join(args) + '\n'); output.flush()
                with subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,errors='replace',bufsize=1) as process:
                    for line in process.stdout:
                        print(line,end='',flush=True); output.write(line); output.flush()
                    status = process.wait()
            if status: raise RuntimeError(f'Command failed ({status}); full log: {log or SETUP_LOG}')

        def checkout(url,path,revision):
            if not path.exists(): run(['git','clone','--no-checkout',url,path])
            assert (path/'.git').is_dir(),path
            run(['git','fetch','--depth','1','origin',revision],cwd=path)
            run(['git','checkout','--detach',revision],cwd=path)

        for file in (CHECKPOINT,LEGACY_PROMPT,UPT_PROMPT,COCO_JSON,DETECTIONS):
            assert file.is_file(),f'Missing exact input: {file}. Copy mounted path from Input.'
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists(),f'Choose a fresh OUTPUT or restart session: {OUTPUT}'
        import torch
        assert torch.cuda.device_count()==2,'Select GPU T4 x2: ONE notebook, ONE arm, TWO GPUs.'
        for i in range(2): print('GPU',i,torch.cuda.get_device_name(i),flush=True)
        checkout('https://github.com/Supzxjee/Image_Captioning.git',REPO,PROJECT_COMMIT)
        checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2',
             'h5py','pandas','Pillow','tqdm','scipy','nltk==3.9.2','future'])
        run([sys.executable,'-m','pip','install','-q','--no-deps',
             'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
        run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
        run(['java','-version'])
        '''
        setup = textwrap.dedent(setup).replace('__COMMIT__',COMMIT)
        setup += '\nSOURCES = '+repr(sources)+'\n'
        setup += 'SOURCE_HASHES = '+repr(hashes)+'\n'
        setup += "for name, source in SOURCES.items(): (REPO/name).write_text(source,encoding='utf-8')\n"
        setup += "print('Embedded pilot source installed',flush=True)\n"
        cells.append(code(setup))
        cells.append(markdown('## 2. Kiểm tra trước khi chạy\nKiểm tra pooling, mask, gradient CE/SCST, đồng bộ hai GPU và bộ tính điểm. Chưa huấn luyện caption tại bước này.'))
        cells.append(code('''
        for name in ('test_pilot_data.py','test_pilot_model.py'):
            run([sys.executable,'-u',REPO/name],cwd=REPO)
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
             '--max_restarts=0',REPO/'test_pilot_ddp.py'],cwd=REPO)
        run([sys.executable,'-u',REPO/'pilot_metric_smoke.py',CHAIR],cwd=REPO)
        print('PREFLIGHT PASS',flush=True)
        '''))
        cells.append(markdown('## 3. Khóa tập ảnh và chuẩn bị đặc trưng\nChỉ stage 11.000 ảnh từ visual cache. Tạo embedding CLIP đóng băng cho cụm chủ thể/đối tượng/quan hệ; giữ liên kết instance và box. Bước này chủ yếu I/O, không cần cả hai GPU.'))
        cells.append(code('''
        CONFIG = Path('/tmp/upt_pilot_config_' + ARM + '.json')
        expected = {
            'checkpoint':'5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75',
            'legacy_prompt':'d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d',
            'upt_prompt':'24e7861a4aa0e7e731d67689dce195688096232bcbef1b4264034cf520d6957d',
            'dataset_json':'2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d',
            'detections':'498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c',
        }
        cfg = dict(arm=ARM,checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),upt_prompt=str(UPT_PROMPT),
                   visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),detections=str(DETECTIONS),
                   output=str(OUTPUT),stage='/tmp/upt_alignment_stage_'+ARM+'.h5',chair=str(CHAIR),
                   global_batch_size=8,eval_batch_size=16,warmup_updates=200,base_lr=5e-6,branch_lr=1e-4,
                   scst_weight=.05,expected_hashes=expected,seed=42,project_commit=PROJECT_COMMIT,
                   embedded_source_sha256=SOURCE_HASHES)
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        run([sys.executable,'-u',REPO/'run_pilot.py','prepare','--config',CONFIG],cwd=REPO)
        PROTOCOL = OUTPUT/'protocol.json'
        (OUTPUT/'source').mkdir()
        for name,source in SOURCES.items(): (OUTPUT/'source'/name).write_text(source,encoding='utf-8')
        shutil.copy2(SETUP_LOG,OUTPUT/'setup.log')
        locked = json.loads(PROTOCOL.read_text())
        print('LOCKED COHORT:',locked['cohort_sha256'],locked['counts_by_mode'],flush=True)
        '''))
        cells.append(markdown(f'''## 4. Huấn luyện nhánh {arm} bằng hai GPU
        Hai worker thuộc cùng nhánh, đồng bộ gradient DDP; GPU 0 không chạy nhánh khác GPU 1.
        Đánh giá checkpoint ban đầu trước, sau đó 1.250 cập nhật và đánh giá checkpoint cuối.
        Không chọn checkpoint bằng test; checkpoint cuối được lưu riêng ở Output.
        '''))
        cells.append(code('''
        TRAIN_LOG = OUTPUT/'run.log'
        try:
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                 '--max_restarts=0',REPO/'run_pilot.py','worker','--config',PROTOCOL],cwd=REPO,log=TRAIN_LOG)
        except Exception:
            run([sys.executable,'-u',REPO/'run_pilot.py','archive','--config',PROTOCOL],cwd=REPO,log=TRAIN_LOG)
            raise
        '''))
        cells.append(markdown('''## 5. Tính điểm và đóng gói kết quả
        CIDEr/BLEU/METEOR/ROUGE và CHAIR trên cùng 1.000 val, giải mã greedy cho mọi nhánh.
        B/C/D có thêm tắt nhánh, tráo text, tráo vùng; đây là kiểm tra mức sử dụng bằng chứng, chưa phải kiểm định quan hệ đúng/sai.
        Gửi lại ZIP review; giữ `checkpoint.pth` trong Saved Output. ZIP review không chứa checkpoint hay HDF5.
        '''))
        cells.append(code('''
        try:
            run([sys.executable,'-u',REPO/'run_pilot.py','evaluate','--config',PROTOCOL],cwd=REPO,log=TRAIN_LOG)
        finally:
            run([sys.executable,'-u',REPO/'run_pilot.py','archive','--config',PROTOCOL],cwd=REPO,log=TRAIN_LOG)
        result = json.loads((OUTPUT/'run_summary.json').read_text())
        print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
        print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
        print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.pth',flush=True)
        '''))
        payload = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                       language_info=dict(name='python',version='3.12')),cells=cells)
        for i,cell in enumerate(cells): cell['id'] = f'{arm.lower()}-{i:02}'
        name = f'UPT_ALIGNMENT_{arm}_2GPU.ipynb'
        (HERE/name).write_text(json.dumps(payload,indent=1,ensure_ascii=False),encoding='utf-8')
        print(name,flush=True)
    with zipfile.ZipFile(HERE/'UPT_ALIGNMENT_4_NOTEBOOKS.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(HERE.glob('UPT_ALIGNMENT_?_2GPU.ipynb')): z.write(path,path.name)
        if (HERE/'README.md').is_file(): z.write(HERE/'README.md','README.md')


if __name__ == '__main__': main()
