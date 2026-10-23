"""Build portable, unexecuted G/S notebooks; no torch/model imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)


def main():
    nb.COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();nb.HERE=HERE
    files=['evidence_data.py','evidence_model.py','evidence_stage.py','evidence_metrics.py','run_evidence.py','A_secondary_reference.json']
    bundle={name:(HERE/name).read_text(encoding='utf-8') for name in files}
    for name in ('pilot_data.py','pilot_model.py','run_pilot.py'):bundle[name]=(ROOT/'experiments/upt_alignment_pilot'/name).read_text(encoding='utf-8')
    bundle['hidden_metric_smoke.py']=(ROOT/'experiments/upt_hidden_pair/hidden_metric_smoke.py').read_text(encoding='utf-8')
    for name in ('captioning/scst_runtime.py','captioning/models.py'):bundle[name]=(ROOT/name).read_text(encoding='utf-8')
    for name,source in bundle.items():
        if name.endswith('.py'):ast.parse(source,filename=name)
    data=json.loads((HERE/'DATA_PACKAGE.json').read_text())
    for arm in ('G','S'):
        cells=[nb.md(f'''
            # Noun evidence {arm} — {'global supervision control' if arm=='G' else 'bbox-supervised patch evidence'}
            **Internet ON · GPU T4×2 · Save & Run All.** Một notebook, một nhánh, cả hai GPU.
            Đóng băng toàn bộ checkpoint A; chỉ học head108.752tham số từ196patchCLIP→80lớp.
            Cộng bằng chứng ảnh vào logits của từ đối tượng, có chặn hot+dog. Không ép chèn từ.
            CE + .05SCST(CIDEr-D+.5object_count_v2) sau200updates +.1presence +.1region.
            2.500updates, hai vòng train10k, seed42, AdamW1e-4. G/S chỉ khác pooling trong region loss.
            Đánh giá A và endpoint rawgreedy trên val5000 đã dùng trước; **không dùng test**.
            Gửi ZIP review, giữ head.bin và checkpoint A riêng. Đây là thực nghiệm, chưa có bằng chứng cải thiện.
            ''')]
        cells.append(nb.code(f'''
            print('NOUN EVIDENCE {arm}: START',flush=True)
            from pathlib import Path
            ARM='{arm}'
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            OUTPUT=Path('/kaggle/working/noun_evidence_{arm}')
            for path in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,DETECTIONS):
                assert path.is_file(),f'Missing exact input: {{path}}. Sửa path nếu tài khoản bạn mount khác.'
            assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
            assert not OUTPUT.exists(),'Preserve old output; use a fresh session/output.'
            '''))
        cells.append(nb.bootstrap(bundle,'noun-evidence-'+arm.lower()))
        cells.append(nb.code(f'''
            # Five small label JSON files are bundled with the pinned source commit (~0.96MB ZIP).
            # No extra Kaggle dataset/upload is needed; optional dataset copy must have identical receipt.
            import zipfile
            LABEL_ARCHIVE=REPO/'experiments/noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'
            assert hashlib.sha256(LABEL_ARCHIVE.read_bytes()).hexdigest()=='{data['archive_sha256']}'
            LABELS=REPO/'noun-evidence-labels'
            LABELS.mkdir()
            with zipfile.ZipFile(LABEL_ARCHIVE) as archive:
                assert set(archive.namelist())=={{'groups.json','targets.json','lexicon.json','prior.json','receipt.json'}}
                archive.extractall(LABELS)
            assert hashlib.sha256((LABELS/'receipt.json').read_bytes()).hexdigest()=='{data['receipt_sha256']}'
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2',
                'h5py','numpy','Pillow','tqdm','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps',
                'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version'])
            CHAIR=Path('/tmp/noun-evidence-chair')
            checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
            import torch
            assert torch.cuda.device_count()==2,'Select GPU T4x2 for this one arm.'
            run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
            '''))
        cells.append(nb.code(f'''
            import numpy as np
            groups=json.loads((LABELS/'groups.json').read_text())
            ids=sorted(r['image_id'] for r in groups['val'])
            rng=np.random.default_rng(20261023);shuffled=np.asarray(ids)[rng.permutation(len(ids))]
            # Prelock a derangement independently of model outputs; fixed-point-free cyclic shift in random order.
            donors={{str(int(shuffled[i])):int(shuffled[(i+1)%len(ids)]) for i in range(len(ids))}}
            assert all(int(i)!=d for i,d in donors.items())
            CONFIG=Path('/tmp/noun_evidence_'+ARM+'.json')
            cfg=dict(arm=ARM,checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
                dataset_json=str(COCO_JSON),detections=str(DETECTIONS),labels=str(LABELS),label_receipt_sha256='{data['receipt_sha256']}',
                output=str(OUTPUT),stage='/tmp/noun_evidence_'+ARM+'.h5',chair=str(CHAIR),project_commit=PROJECT_COMMIT,
                implementation_sha256=SOURCE_HASHES,global_batch_size=8,eval_batch_size=16,seed=42,updates=2500,
                head_lr=1e-4,weight_decay=.01,max_gradient_norm=1.,warmup_updates=200,scst_weight=.05,
                object_reward_weight=.5,presence_weight=.1,region_weight=.1,logit_weight=.5,correction_clip=2.,
                shuffle_donors=donors,shuffle_seed=20261023,head_parameters=108752,
                gate=dict(S_minus_A_CIDEr_min=.01,S_minus_A_paired95_lower_gt=0.,S_minus_G_paired95_lower_gt=0.,
                    CHAIRs_increase_max=0.,CHAIRi_increase_max=0.,Recall_drop_max=.005),
                selection='Endpoint only; no test, rerank, sweep or automatic promotion',
                control_policy='Evaluate S controls provisionally after S/A numeric gate; interpret only after S/G gate too.')
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8');PROTOCOL=OUTPUT/'protocol.json'
            def distributed(mode):
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                    '--max_restarts=0','run_evidence.py',mode,'--config',PROTOCOL],cwd=REPO)
            try:
                run([sys.executable,'-u','run_evidence.py','prepare','--config',CONFIG],cwd=REPO)
                distributed('preflight')
                distributed('worker')
                run([sys.executable,'-u','run_evidence.py','evaluate','--config',PROTOCOL],cwd=REPO)
                if ARM=='S' and json.loads((OUTPUT/'run_summary.json').read_text())['S_A_gate']:
                    distributed('controls')
                    run([sys.executable,'-u','run_evidence.py','evaluate','--config',PROTOCOL],cwd=REPO)
            finally:
                if OUTPUT.is_dir():
                    if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                    for name,source in SOURCES.items():
                        target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source,encoding='utf-8')
                    if PROTOCOL.is_file():run([sys.executable,'-u','run_evidence.py','archive','--config',PROTOCOL],cwd=REPO)
                    else:
                        shutil.copyfile(CONFIG,OUTPUT/'requested_config.json')
                        with zipfile.ZipFile(str(OUTPUT)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
                            for path in sorted(OUTPUT.rglob('*')):
                                if path.is_file():z.write(path,path.relative_to(OUTPUT))
            print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
            print('KEEP HEAD:',OUTPUT/'head.bin',flush=True)
            '''))
        cells.append(nb.md('''
            **Đầu vào:** checkpoint A, legacy prompt cache, visual cache, MSCOCO dataset JSON, detection cache.
            Không cần UPT/VG/noun-box dataset hay ảnh gốc. Nhãn TRAIN đóng gói cùng source và kiểm SHA.
            Preflight FAIL thì dừng trước training; ZIP có thể tồn tại dù thất bại. Chỉ COMPLETE mới là kết quả.
            G/S cần so sánh khớp order, runtime, base hash và paired CI; chưa thể kết luận từ mỗi nhánh riêng.
            head.bin chỉ chứa head mới; cần giữ A checkpoint để dùng lại mô hình.
            '''))
        nb.write(f'NOUN_EVIDENCE_{arm}_2GPU.ipynb',cells)


if __name__=='__main__':main()
