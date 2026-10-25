"""Generate the small user-run inference notebook; does not execute it."""
import importlib.util
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)


def build():
    nb.HERE=HERE;nb.COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    bundle={n:(HERE/n).read_text(encoding='utf-8') for n in ['candidate_data.py','run_candidates.py','test_candidates.py']}
    bundle['evidence_data.py']=(ROOT/'experiments/noun_evidence_proposal/evidence_data.py').read_text(encoding='utf-8')
    bundle['lexical_policy.json']=(ROOT/'experiments/coco_grounding_feasibility/MAPPING_POLICY_V11.json').read_text(encoding='utf-8')
    for name in ['captioning/models.py','captioning/tokenizer.py','captioning/config.py','captioning/semantic_prompts.py','captioning/visual_cache.py','captioning/scst_runtime.py','captioning/scst.py','captioning/object_consistency.py']:
        bundle[name]=(ROOT/name).read_text(encoding='utf-8')
    cells=[nb.md('''
        # TRAIN candidate headroom — 2.000 ảnh, T4 ×2, không huấn luyện
        Kiểm tra exact A có sinh được caption tốt hơn greedy hay không trước khi xây preference bank.
        Một notebook dùng hai GPU inference độc lập. 500 ảnh đầu đo tốc độ; nếu dự toán vượt ngân sách thì dừng,
        nếu đạt chạy đủ2.000ảnh TRAIN. Mỗiảnh1greedy+4sample(temperature.8/top-p.95),max30token gồm BOS.
        Không train, không optimizer, không đánh giá val/test và không tự mở rộng10k.

        **T4×2 · Internet ON · Save & Run All.** Thêm5dataset: pair-source-fullframe-ce,prompt-cache,
        visual-cache,mscoco-2014,objectdetectionecache. Không cần H/L hoặc UPT hidden cache.
        Gửi `preference_candidate_train2000_review.zip`; giữ toàn bộ folder để dùng lại nếu phương pháp được chốt.
        Oracle headroom TRAIN không phải kết quả cải thiện mô hình. Novelty vẫn chưa được xác nhận.
        ''')]
    cells.append(nb.code('''
        print('TRAIN CANDIDATE AUDIT START — NO TRAINING',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
        OUTPUT=Path('/kaggle/working/preference_candidate_train2000')
        STAGE=Path('/tmp/preference_candidate_train2000.h5')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,DETECTIONS):assert p.is_file(),p
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists() and not STAGE.exists(),'Preserve old results; use a fresh session/path.'
        '''))
    cells.append(nb.bootstrap(bundle,'preference-candidate-2000'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow','pycocoevalcap==1.2','nltk==3.9.2'])
        run(['java','-version'])
        run([sys.executable,'-u','test_candidates.py'],cwd=REPO)
        import torch
        assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'Select T4x2.'
        CONFIG=Path('/tmp/preference_candidate_config.json')
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
            dataset_json=str(COCO_JSON),detections=str(DETECTIONS),output=str(OUTPUT),stage=str(STAGE),
            labels_archive=str(REPO/'experiments/noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'),
            project_commit=PROJECT_COMMIT,implementation_sha256=SOURCE_HASHES,reported_remaining_gpu_hours=50,
            worker_budget_seconds=3*3600,scope='TRAIN2000 candidate feasibility only; no training/no val/test',
            generation=dict(greedy=1,samples=4,temperature=.8,top_p=.95,max_length_including_BOS=30))
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8');PROTOCOL=OUTPUT/'protocol.json'
        try:
            run([sys.executable,'-u','run_candidates.py','prepare','--config',CONFIG],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                '--max_restarts=0','run_candidates.py','worker','--config',PROTOCOL],cwd=REPO)
            run([sys.executable,'-u','run_candidates.py','evaluate','--config',PROTOCOL],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                if not PROTOCOL.is_file():shutil.copyfile(CONFIG,OUTPUT/'requested_config.json')
                for name,source in SOURCES.items():
                    path=OUTPUT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source,encoding='utf-8')
                import zipfile
                with zipfile.ZipFile(str(OUTPUT)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
                    for p in sorted(OUTPUT.rglob('*')):
                        if p.is_file():z.write(p,p.relative_to(OUTPUT))
        result=json.loads((OUTPUT/'numeric_summary.json').read_text(encoding='utf-8'))
        assert result['status']=='CANDIDATE_AUDIT_COMPLETE','ZIP creation alone is not success.'
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        print('SEND:',str(OUTPUT)+'_review.zip',flush=True)
        print('KEEP:',OUTPUT,flush=True)
        '''))
    nb.write('TRAIN_CANDIDATE_HEADROOM_2000_2GPU.ipynb',cells)


if __name__=='__main__':build()
