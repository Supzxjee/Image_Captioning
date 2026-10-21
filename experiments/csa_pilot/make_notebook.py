"""Portable user-run notebook. Generation never executes training or inference."""
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
    bundle = {n:original[n] for n in ('pilot_data.py','pilot_model.py','run_pilot.py','hidden_data.py',
                                      'captioning/scst_runtime.py','hidden_metric_smoke.py')}
    for name in ('csa_core.py','csa_model.py'):
        bundle[name] = (HERE.parent/'csa_core_smoke'/name).read_text(encoding='utf-8')
    for name in ('phase_policy.py','run_csa_pilot.py','test_policy_static.py','test_phase_model.py'):
        bundle[name] = (HERE/name).read_text(encoding='utf-8')
    for name,source in bundle.items():
        ast.parse(source,filename=name)
    cells = [old.md('''
        # CSA trên Q-Former — pilot hai pha, T4×2
        **Internet ON, GPU T4×2, Save & Run All. Một notebook dùng cả hai GPU.**
        Pha1:1250updates chỉ học core CSA bằng CE, LRwarmup100steps lên1e-4, baseline frozen.
        Pha2:1250updates học chung, LRbase5e-6/core2e-5, CE200steps rồi CE+.05SCST;
        rewardCIDErD+.5object_count_v2. Reset optimizer/RNG/sampler giữa hai pha, bank luôn frozen.
        Giữ core2object+1relation và memory[baseline;Vo;Vr] đã smoke. Không dùng checkpoint sau smoke.
        Train10k/dev1000 như A, seed42, globalbatch8. Chỉ cuối pha2 đánh giá original/on/off rawgreedy.
        Không test/rerank/best-checkpoint selection; CSA có thêm1250core-onlysteps soA.
        Dataset giống CSA smoke; không cần cache UPT hoặc VG. Hai bank/sourceCSA tải có hash kiểm.
        ''')]
    cells.append(old.code('''
        print('CSA TWO-PHASE PILOT: BOOTSTRAP START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
        OUTPUT=Path('/kaggle/working/csa_two_phase_pilot')
        PHASE1_CHECKPOINT=None  # Chỉ khi tiếp tục từ phase1_checkpoint.bin của đúng notebook này.
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,DETECTIONS):
            assert p.is_file(),p
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists(),'Preserve existing output; use fresh OUTPUT/session.'
        if PHASE1_CHECKPOINT is not None:
            PHASE1_CHECKPOINT=Path(PHASE1_CHECKPOINT)
            assert PHASE1_CHECKPOINT.is_file(),PHASE1_CHECKPOINT
        OUTPUT.mkdir(parents=True)
        '''))
    cells.append(old.bootstrap(bundle,'csa_pilot'))
    cells.append(old.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2',
             'h5py','numpy','Pillow','tqdm','nltk==3.9.2','future'])
        run([sys.executable,'-m','pip','install','-q','--no-deps',
             'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
        run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
        run(['java','-version'])
        import torch
        assert torch.cuda.device_count()==2,'Select T4x2.'
        CHAIR=Path('/tmp/csa-pilot-chair')
        checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
        '''))
    cells.append(old.code('''
        import urllib.request,zipfile
        sys.path.insert(0,str(REPO))
        from csa_core import BANKS,REFERENCE_SHA,sha
        REFERENCE=OUTPUT/'reference'
        REFERENCE.mkdir()
        REF_COMMIT='0cc570ba7c50ef08c46a21439fe73ef0a5df97ab'
        assets={'models/caption/csa_net.py':REFERENCE_SHA}
        assets.update({'coco_caption/'+name:value[0] for name,value in BANKS.items()})
        for name,source in SOURCES.items():
            p=OUTPUT/'source'/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(source,encoding='utf-8')
        CONFIG=OUTPUT/'requested_config.json'
        cfg=dict(arm='CSA',checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),
                 visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),detections=str(DETECTIONS),
                 chair=str(CHAIR),reference=str(REFERENCE),output=str(OUTPUT),stage='/tmp/csa_two_phase_stage.h5',
                 phase1_checkpoint=str(PHASE1_CHECKPOINT or OUTPUT/'phase1_checkpoint.bin'),
                 resume_phase1=PHASE1_CHECKPOINT is not None,project_commit=PROJECT_COMMIT,csa_commit=REF_COMMIT,
                 implementation_sha256=SOURCE_HASHES,global_batch_size=8,updates_per_phase=1250,
                 evaluation_policy='Fixed final phase2 checkpoint; original/on/off greedy; reused dev1000, no test.',
                 gate=dict(CIDEr_delta_vs_A_min=.01,paired95_lower_gt=0,CHAIRs_increase_max=0,
                           CHAIRi_increase_max=0,Recall_drop_max=.01,beat_original=True),
                 comparison_limits='1250 extra core-only updates vs A; not matched compute/capacity.')
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        PROTOCOL=OUTPUT/'protocol.json'
        try:
            for name,expected in assets.items():
                print('Downloading pinned author asset:',name,flush=True)
                url=f'https://raw.githubusercontent.com/CrossmodalGroup/CSA-Net/{REF_COMMIT}/{name}'
                request=urllib.request.Request(url,headers={'User-Agent':'csa-two-phase-pilot'})
                with urllib.request.urlopen(request,timeout=120) as response:
                    (REFERENCE/Path(name).name).write_bytes(response.read())
                assert sha(REFERENCE/Path(name).name)==expected,name
            run([sys.executable,'-u','test_policy_static.py'],cwd=REPO)
            run([sys.executable,'-u','test_phase_model.py'],cwd=REPO)
            for phase in (1,2):
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                     '--nproc_per_node=2','--max_restarts=0','test_phase_model.py','--phase',phase],cwd=REPO)
            run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
            run([sys.executable,'-u','run_csa_pilot.py','prepare','--config',CONFIG],cwd=REPO)
            phases=(2,) if PHASE1_CHECKPOINT is not None else (1,2)
            for phase in phases:
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                     '--nproc_per_node=2','--max_restarts=0','run_csa_pilot.py','worker',
                     '--phase',phase,'--config',PROTOCOL],cwd=REPO)
            run([sys.executable,'-u','run_csa_pilot.py','evaluate','--config',PROTOCOL],cwd=REPO)
        finally:
            if LOG.is_file():
                shutil.copyfile(LOG,OUTPUT/'notebook.log')
            if PROTOCOL.is_file():
                run([sys.executable,'-u','run_csa_pilot.py','archive','--config',PROTOCOL],cwd=REPO)
            else:
                with zipfile.ZipFile(str(OUTPUT)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
                    for p in sorted(OUTPUT.rglob('*')):
                        if p.is_file() and p.suffix in ('.json','.py','.log'):
                            z.write(p,p.relative_to(OUTPUT))
        print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
        print('KEEP FINAL CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
        print('KEEP PHASE1 CHECKPOINT:',cfg['phase1_checkpoint'],flush=True)
        '''))
    cells.append(old.md('''
        Gửi `csa_two_phase_pilot_review.zip`; giữ riêng `checkpoint.bin` và `phase1_checkpoint.bin`.
        Nếu hết phiên sau pha1, mount checkpoint đó vào notebook mới, sửa PHASE1_CHECKPOINT ở cellđầu.
        Pha2 vẫn tạo optimizer mới và reset RNG/sampler; dữ liệu/source/recipe khác sẽ bị từ chối.
        Không dùng checkpoint.bin cuối pha2 hoặc checkpoint smoke làm checkpoint pha1.
        Đánh giá/soA sau khi kiểm matched protocol/runtime; chưa tự động tuyên bố gain/grounding.
        Chỉ khi ΔCIDEr≥.01 và paired95%lower>0, CHAIRs/i không tăng, recalldrop≤.01, vượtoriginal
        mới review mù factuality; không đạt thì dừng recipe, không tự fulltrain/tune/extendsteps.
        Nếu lỗi: `print(LOG.read_text(encoding='utf-8')[-16000:])`.
        '''))
    for i,c in enumerate(cells):
        c['id']=f'csa-pilot-{i:02d}'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),cells=cells)
    raw=json.dumps(notebook,ensure_ascii=False,indent=1).encode('utf-8')
    assert len(raw)<1_000_000
    (HERE/'CSA_TWO_PHASE_PILOT_2GPU.ipynb').write_bytes(raw)
    print('Unexecuted notebook:',len(raw),'bytes')


if __name__=='__main__':
    main()
