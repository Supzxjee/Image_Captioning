"""Portable one-model/two-GPU H/L pilot notebook; never executes locally."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb);nb.HERE=HERE;nb.COMMIT='8b8516959dba8e8519ba6536c9169b2469e3506b'


def build():
    bundle={n:(HERE/n).read_text(encoding='utf-8') for n in ('detail_data.py','detail_model.py','detail_stage.py','run_detail.py','detail_metrics.py','test_detail.py')}
    bundle['tile_data.py']=(ROOT/'experiments/native_detail_benchmark/tile_data.py').read_text(encoding='utf-8')
    bundle['hl_upload_inventory.json']=(ROOT/'experiments/native_detail_cache/upload_inventory.json').read_text(encoding='utf-8')
    bundle['hidden_metric_smoke.py']=(ROOT/'experiments/upt_hidden_pair/hidden_metric_smoke.py').read_text(encoding='utf-8')
    for n in ('captioning/models.py','captioning/scst_runtime.py','captioning/scst.py','captioning/tokenizer.py','captioning/config.py','captioning/semantic_prompts.py','captioning/visual_cache.py'):
        bundle[n]=(ROOT/n).read_text(encoding='utf-8')
    cells=[nb.md('''
        # Native-detail H/L — một adapter chung, T4 ×2
        **Một notebook, hai GPU**, không chạy hai mô hình H/L độc lập.
        Exact A frozen;262.656 tham số adapter;H/L cùng trọng số. H/L→Q' rồi tính lại legacy grounding→decoder A.
        **CE-only**, đủ5caption TRAIN có EOS,4epoch/5.000update,globalbatch8,seed42.
        Không thêm SCST/object reward, không chọn checkpoint theo val. Kiến thức SCST trước đây vẫn nằm trong A.

        Chọn **T4 ×2**, Internet ON, **Save & Run All**. Root không tự chạy notebook.
        Thêm5dataset: **native-detail-hl-15k-datase**, **pair-source-fullframe-ce**, **prompt-cache**, **visual-cache**, **mscoco-2014**.
        Không cần ảnh gốc/detectioncache/UPT/VG cache mới.

        Notebook kiểm hash cache thật, stage nguyên dữ liệu A (không lượng tử hóa thêm), kiểm zero-init/gradient,
        đo25bước CE và tốc độ sinh trên mẫu TRAIN rồi khôi phục head+optimizer+seed ban đầu.
        Nếu ước lượng có hệ số dự phòng1,5 trong ngân sách **8giờ wall**, notebook tiếp tục5.000update và đánh giá.
        Nếu không đủ ngân sách hoặc preflight lỗi, dừng với review ZIP; không gọi lượt thiếu là kết quả âm.
        Bạn báo còn30giờGPU;ước lượng wall không được coi là số giờ quota thực tế Kaggle sẽ tính.

        Đánh giá rawgreedy A/L/H trên **reusedval5000**, không test. Gate đầu tư:H−A≥.01,H−L≥.005,cả haiCIcận dưới>0,
        CHAIRs/i≤A vàRecall≥A−.005. Không tự scale/quét hệ số nếu fail.
        Gửi `native_detail_shared_HL_pilot_review.zip`; giữ `head.bin` **và checkpoint A** để dùng mô hình.
        ''')]
    cells.append(nb.code('''
        print('SHARED H/L PILOT: START',flush=True)
        from pathlib import Path
        HL_CACHE=Path('/kaggle/input/datasets/ducanh2403/native-detail-hl-15k-datase/native_detail_HL_15k_dataset')
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        OUTPUT=Path('/kaggle/working/native_detail_shared_HL_pilot')
        STAGE=Path('/tmp/native_detail_shared_HL_stage.h5')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,HL_CACHE/'cache_spec.json',HL_CACHE/'cache_receipt.json'):assert p.is_file(),p
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists() and not STAGE.exists(),'Preserve existing output/stage; use a fresh session or paths.'
        '''))
    cells.append(nb.bootstrap(bundle,'native-detail-shared-pilot'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow','pandas',
            'pycocotools','pycocoevalcap==1.2','nltk==3.9.2','future'])
        run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
        run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
        run(['java','-version'])
        import torch
        assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(r) for r in range(2)),'Select T4x2.'
        run([sys.executable,'-u','test_detail.py'],cwd=REPO)
        CHAIR=Path('/tmp/native-detail-pilot-chair')
        checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
        run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
        '''))
    cells.append(nb.code('''
        CONFIG=Path('/tmp/native_detail_pilot_config.json')
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),
            tile_cache=str(HL_CACHE),output=str(OUTPUT),stage=str(STAGE),inventory=str(REPO/'hl_upload_inventory.json'),chair=str(CHAIR),
            epochs=4,updates=5000,global_batch_size=8,eval_batch_size=16,seed=42,head_lr=1e-4,weight_decay=.01,clip_norm=1.,head_parameters=262656,
            wall_budget_seconds=8*3600,reported_gpu_quota_hours=30,project_commit=PROJECT_COMMIT,implementation_sha256=SOURCE_HASHES,
            target_policy='Five natural TRAIN captions per image,truncatewithEOS;BOSexcluded/PADignored/globaltokenCE includingEOS',
            evaluation='Endpoint only;rawgreedy,maxlen30,A/L/H,reusedval5000;notest/norerank/nosweep',
            planned_gate=dict(H_A_CIDEr=.01,H_L_CIDEr=.005,paired95lower_gt=0,CHAIRs_i_not_above_A=True,Recall_drop_max=.005))
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8');PROTOCOL=OUTPUT/'protocol.json'
        try:
            run([sys.executable,'-u','run_detail.py','prepare','--config',CONFIG],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
                'run_detail.py','worker','--config',PROTOCOL],cwd=REPO)
            run([sys.executable,'-u','run_detail.py','evaluate','--config',PROTOCOL],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                if not PROTOCOL.is_file():shutil.copyfile(CONFIG,OUTPUT/'requested_config.json')
                for name,source in SOURCES.items():
                    target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source,encoding='utf-8')
                run([sys.executable,'-u','run_detail.py','archive','--config',PROTOCOL if PROTOCOL.is_file() else CONFIG],cwd=REPO)
        summary=json.loads((OUTPUT/'run_summary.json').read_text(encoding='utf-8'))
        assert summary['status']=='PILOT_COMPLETE','ZIP alone is not success; send incomplete review for diagnosis.'
        print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
        print('KEEP ADAPTER:',OUTPUT/'head.bin','KEEP ORIGINAL A CHECKPOINT SEPARATELY',flush=True)
        '''))
    nb.write('NATIVE_DETAIL_SHARED_HL_2GPU.ipynb',cells)


if __name__=='__main__':build()
