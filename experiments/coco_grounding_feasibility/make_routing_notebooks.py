"""Package two unexecuted user-run pilots; does not import torch or run a model."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.COMMIT='2bf1e041744f00667fde287baacaeac745462835'
FILES=['route_probe.py','run_route_smoke.py','routing_data.py','routing_model.py','routing_stage.py','routing_preflight.py','run_routing.py']


def main():
    bundle={n:(HERE/n).read_text(encoding='utf-8') for n in FILES}
    for folder,names in [('upt_alignment_pilot',['pilot_data.py','pilot_model.py','run_pilot.py']),
                         ('upt_hidden_pair',['hidden_data.py','run_hidden.py','hidden_metric_smoke.py'])]:
        bundle.update({n:(ROOT/'experiments'/folder/n).read_text(encoding='utf-8') for n in names})
    for n in ['captioning/scst_runtime.py','captioning/upt_reference.py']:
        bundle[n]=(ROOT/n).read_text(encoding='utf-8')
    for name,source in bundle.items():ast.parse(source,filename=name)
    ready=json.loads((HERE/'FULL_DATA_READY.json').read_text(encoding='utf-8'))
    receipt=ready['data_manifest_receipt_sha256']
    model_sha=hashlib.sha256((ROOT/'captioning/models.py').read_text(encoding='utf-8').encode()).hexdigest()
    for arm in ('B','R'):
        cells=[nb.md(f'''
            # COCO noun–patch routing {arm} — {'spatial-path control' if arm=='B' else 'bbox-conditioned routing'}
            **Internet ON · T4×2 · Save & Run All.** Một notebook dùng cả hai GPU cho một nhánh.
            B/R có thể chạy song song trên hai tài khoản; mỗi tài khoản cần quyền đọc private datasets.
            Giữ baseline Q-Former + legacy prompt, không thêm module/parameter inference.
            Cùng train10k/dev1000, seed42,1250updates,batch8, CE200 rồi CE+.05SCST(CIDEr-D+.5object_count_v2).
            Thêm .05 L_aux từ step1: B dùng khối lượng đường truyền tới mọi patch; R tới patch trong bbox.
            Không có kết quả mới trước khi chạy. Nhãn noun–box là giám sát yếu, không phải grounding gold.
            ''')]
        cells.append(nb.code(f'''
            print('COCO ROUTING {arm}: START',flush=True)
            from pathlib import Path
            ARM='{arm}'
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            NOUN_MANIFEST=Path('/kaggle/input/datasets/ducanh2403/coco-noun-box-train10k-v11')
            OUTPUT=Path('/kaggle/working/coco_routing_{arm}')
            for path in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,DETECTIONS,NOUN_MANIFEST/'manifest_receipt.json'):
                assert path.is_file(),f'Missing exact input: {{path}}; sửa path nếu dataset được mount ở tài khoản bạn.'
            assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
            assert not OUTPUT.exists(),'Preserve old output; use fresh session/output.'
            '''))
        cells.append(nb.bootstrap(bundle,'coco-routing-'+arm.lower()))
        cells.append(nb.code('''
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2',
                'h5py','numpy','Pillow','tqdm','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps',
                'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version'])
            CHAIR=Path('/tmp/coco-routing-chair')
            checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
            import torch
            assert torch.cuda.device_count()==2,'Select GPU T4x2 for this one arm.'
            run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
            '''))
        cells.append(nb.code(f'''
            CONFIG=Path('/tmp/coco_routing_'+ARM+'.json')
            cfg=dict(arm=ARM,checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
                dataset_json=str(COCO_JSON),detections=str(DETECTIONS),noun_manifest=str(NOUN_MANIFEST),output=str(OUTPUT),
                stage='/tmp/coco_routing_'+ARM+'.h5',chair=str(CHAIR),project_commit=PROJECT_COMMIT,
                implementation_sha256=SOURCE_HASHES,noun_receipt_sha256='{receipt}',models_source_sha256='{model_sha}',
                global_batch_size=8,eval_batch_size=16,warmup_updates=200,base_lr=5e-6,scst_weight=.05,seed=42,
                aux_weight=.05,route_epsilon=1e-8,updates=1250,
                adamw_options=dict(weight_decay=.01,betas=[.9,.999],eps=1e-8,amsgrad=False,maximize=False,
                    foreach=None,capturable=False,differentiable=False,fused=None),
                record_optimizer_manifest=True,max_gradient_norm=1.,dropout_policy='model.eval() with gradients enabled',
                locked_numeric_gate=dict(R_minus_A_CIDEr_min=.01,R_minus_A_paired95_lower_gt=0.,
                    R_CIDEr_gt_original=True,CHAIRs_increase_max=0.,CHAIRi_increase_max=0.,Recall_drop_max=.01,
                    position_attribution_requires_R_minus_B_paired95_lower_gt=0.),
                promotion_requires='Matched protocol, numeric gates, then small blind factual review. No test/rerank/sweep.')
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            PROTOCOL=OUTPUT/'protocol.json'
            try:
                run([sys.executable,'-u','run_routing.py','prepare','--config',CONFIG],cwd=REPO)
                for mode in ('preflight','worker'):
                    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                        '--nproc_per_node=2','--max_restarts=0','run_routing.py',mode,'--config',PROTOCOL],cwd=REPO)
                run([sys.executable,'-u','run_routing.py','evaluate','--config',PROTOCOL],cwd=REPO)
            finally:
                import zipfile
                if OUTPUT.is_dir():
                    if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                    for name,source in SOURCES.items():
                        target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_text(source,encoding='utf-8')
                    if PROTOCOL.is_file():
                        run([sys.executable,'-u','run_routing.py','archive','--config',PROTOCOL],cwd=REPO)
                    else:
                        with zipfile.ZipFile(str(OUTPUT)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
                            for path in sorted(OUTPUT.rglob('*')):
                                if path.is_file() and path.suffix in ('.json','.jsonl','.log','.py'):
                                    z.write(path,path.relative_to(OUTPUT))
            print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
            print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
            '''))
        cells.append(nb.md('''
            Giữ checkpoint.bin riêng và gửi ZIP review. Stage HDF5 trong /tmp không phải output cần lưu.
            Không cần UPT/VG cache, ảnh gốc hay full COCO annotation ZIP. Cần dataset 7 JSON noun-box đã khóa.
            Preflight nằm trong notebook, không cập nhật trọng số. FAIL thì gửi log; không bỏ qua/đổi tolerance.
            Nếu không thấy lỗi gốc: `print(LOG.read_text(encoding='utf-8')[-16000:])`.
            Chờ đủ B/R mới so sánh với A đã có. Dev1000 đã dùng trước; chưa phải bằng chứng test độc lập.
            '''))
        nb.HERE=HERE
        nb.write(f'COCO_ROUTING_{arm}_2GPU.ipynb',cells)
    with zipfile.ZipFile(HERE/'review/COCO_ROUTING_B_R_NOTEBOOKS.zip','w',zipfile.ZIP_DEFLATED) as z:
        for arm in ('B','R'):
            path=HERE/f'COCO_ROUTING_{arm}_2GPU.ipynb';z.write(path,path.name)


if __name__=='__main__':main()
