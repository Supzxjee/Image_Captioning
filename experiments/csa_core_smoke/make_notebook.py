"""Build a portable, unexecuted technical notebook; never run models locally."""
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('hidden_notebooks',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.COMMIT = 'dfa1d4e0c52bea8672c6b770127034c2ec377b58'


def main():
    original = old.sources()
    bundle = {n:original[n] for n in ('pilot_data.py','pilot_model.py','run_pilot.py','hidden_data.py','captioning/scst_runtime.py')}
    for name in ('csa_core.py','csa_model.py','test_csa_core.py','run_csa_smoke.py'):
        bundle[name] = (HERE/name).read_text(encoding='utf-8')
    for name,source in bundle.items():
        ast.parse(source,filename=name)
    cells = [old.md('''
        # CSA core — kiểm chứng và đo chi phí trên T4×2
        **Internet ON, T4×2, Save & Run All.** Chỉ kiểm chứng, chưa phải pilot cải thiện caption.
        Giữ nguyên core CSA 2 object + 1 relation, hai bank gốc đóng băng;
        đầu vào 32 Q-Former queries, decoder đọc [memory baseline; Vo; Vr].
        Mã/banks CSA tải từ commit đã khóa, kiểm SHA256 trước khi sử dụng (~23,5MB banks).
        CPU: đối chiếu output/gradient N=32 và150, buffer và memory/mask.
        GPU: checkpoint thật, kiểm bypass, **8 ảnh TRAIN**, batch toàn cục8,
        **2 bước CE + 2 bước CE+.05SCST(CIDErD+.5object_count_v2)**; không đọc ảnh dev/test.
        Lặp lại cùng8ảnh để đo kỹ thuật; LR chỉ phục vụ probe, không chốt recipe huấn luyện.
        Không cần UPT cache, prompt UPT, ảnh gốc hay dataset VG. Không lưu checkpoint đã cập nhật.
        ''')]
    cells.append(old.code('''
        print('CSA CORE TECHNICAL SMOKE: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
        OUTPUT=Path('/kaggle/working/csa_core_smoke')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,DETECTIONS):
            assert p.is_file(),p
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists(),'Preserve old output; use fresh OUTPUT/session.'
        OUTPUT.mkdir(parents=True)
        '''))
    cells.append(old.bootstrap(bundle,'csa_core'))
    cells.append(old.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow','tqdm','nltk==3.9.2','future'])
        run([sys.executable,'-m','pip','install','-q','--no-deps',
             'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
        run([sys.executable,'-u','-c',"from pattern.en import singularize;assert singularize('umbrellas')=='umbrella'"])
        import torch
        assert torch.cuda.device_count()==2,'Select GPU T4x2.'
        CHAIR=Path('/tmp/csa-core-chair')
        checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
        '''))
    cells.append(old.code('''
        import urllib.request, zipfile
        REFERENCE=OUTPUT/'reference'
        REFERENCE.mkdir()
        sys.path.insert(0,str(REPO))
        from csa_core import BANKS, REFERENCE_SHA, sha
        REF_COMMIT='0cc570ba7c50ef08c46a21439fe73ef0a5df97ab'
        ASSETS={'models/caption/csa_net.py':REFERENCE_SHA}
        ASSETS.update({'coco_caption/'+name:value[0] for name,value in BANKS.items()})
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
                 dataset_json=str(COCO_JSON),detections=str(DETECTIONS),chair=str(CHAIR),output=str(OUTPUT),
                 reference=str(REFERENCE),project_commit=PROJECT_COMMIT,csa_commit=REF_COMMIT,
                 implementation_sha256=SOURCE_HASHES,train_images=8,global_batch=8,steps_CE=2,steps_SCST=2,
                 baseline_lr=5e-6,new_core_lr=1e-4,probe_only=True,dev_used=False,test_used=False)
        CONFIG=OUTPUT/'protocol.json'
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        for name,source in SOURCES.items():
            target=OUTPUT/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source,encoding='utf-8')
        try:
            for name,expected in ASSETS.items():
                destination=REFERENCE/Path(name).name
                url=f'https://raw.githubusercontent.com/CrossmodalGroup/CSA-Net/{REF_COMMIT}/{name}'
                print('Downloading pinned author asset:',name,flush=True)
                request=urllib.request.Request(url,headers={'User-Agent':'csa-core-technical-smoke'})
                with urllib.request.urlopen(request,timeout=120) as response:
                    destination.write_bytes(response.read())
                assert sha(destination)==expected, f'Reference hash differs: {name}'
            run([sys.executable,'-u','test_csa_core.py',REFERENCE],cwd=REPO)
            run([sys.executable,'-u','run_csa_smoke.py','prepare','--config',CONFIG],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                 '--nproc_per_node=2','--max_restarts=0','run_csa_smoke.py','worker','--config',CONFIG],cwd=REPO)
            for rank in range(2):
                receipt=json.loads((OUTPUT/f'complete_rank{rank}.json').read_text())
                assert receipt['status']=='TECHNICAL_SMOKE_PASS' and receipt['completed_updates']==4
            print('TECHNICAL SMOKE COMPLETE. Caption gain NOT evaluated.',flush=True)
        finally:
            if LOG.is_file():
                shutil.copyfile(LOG,OUTPUT/'notebook.log')
            review=Path(str(OUTPUT)+'_review.zip')
            with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
                for p in sorted(OUTPUT.rglob('*')):
                    if p.is_file() and p.suffix in ('.json','.log','.py'):
                        archive.write(p,p.relative_to(OUTPUT))
            print('SEND REVIEW ZIP:',review,flush=True)
        '''))
    cells.append(old.md('''
        Gửi `csa_core_smoke_review.zip`, kể cả nếu lỗi giữa chừng.
        Nếu preflight/gradient/resource không đạt thì dừng, chưa chạy pilot dài.
        Nếu đạt, dùng kết quả bộ nhớ/thời gian để chốt recipe cho23,16triệu tham số mới.
        Không diễn giải loss trên8ảnh hoặc4updates thành bằng chứng cải thiện/căn chỉnh đúng.
        Lõi dùng latent queries thay detector regions; readout ghép token khác decoder CSA gốc.
        Nếu không thấy traceback gốc: `print(LOG.read_text(encoding='utf-8')[-16000:])`.
        '''))
    for i,cell in enumerate(cells):
        cell['id']=f'csa-smoke-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    raw=json.dumps(nb,ensure_ascii=False,indent=1).encode('utf-8')
    assert len(raw)<1_000_000
    (HERE/'CSA_CORE_SMOKE_2GPU.ipynb').write_bytes(raw)
    print('Unexecuted notebook:',len(raw),'bytes')


if __name__=='__main__':
    main()
