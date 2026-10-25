"""Portable read-only diagnostic notebook, never executed during packaging."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('nb',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.COMMIT='973c22925826b50b1497a9502390dea96e2e621a'


def main():
    sources={n:(HERE/n).read_text(encoding='utf-8') for n in ('diagnostic_data.py','run_diagnostic.py','test_diagnostic.py')}
    for folder,names in [('coco_grounding_feasibility',['route_probe.py','audit.py','manifest_policy.py','manifest_policy_v11.py']),
                         ('upt_alignment_pilot',['pilot_data.py']),('upt_hidden_pair',['hidden_data.py'])]:
        sources.update({n:(ROOT/'experiments'/folder/n).read_text(encoding='utf-8') for n in names})
    # This existing policy normally lives two folders below captioning; embedded runtime files live at repo root.
    old="Path(__file__).resolve().parents[2]/'captioning/object_consistency.py'"
    assert sources['manifest_policy_v11.py'].count(old)==1
    sources['manifest_policy_v11.py']=sources['manifest_policy_v11.py'].replace(old,"Path(__file__).resolve().parent/'captioning/object_consistency.py'")
    for name,source in sources.items():ast.parse(source,filename=name)
    cells=[nb.md('''
        # Frozen A — attention geometry diagnostic (tối đa 200 ảnh TRAIN)
        **Internet ON · GPU T4×2 · Save & Run All.** Hai GPU chia ảnh độc lập; không huấn luyện/DDP collectives.
        Kiểm tra cụm danh từ trong **prompt legacy nguyên bản của A** có attention đúng vùng GT hay không.
        Không thay prompt bằng UPT, không tạo caption mới, không cập nhật checkpoint.
        Đầu vào: checkpoint A, legacy prompt cache, global visual cache, MSCOCO dataset JSON,
        dataset noun-box TRAIN10k v1.1 đã dùng trong COCO routing B/R. Không cần ảnh gốc/UPT hidden/CHAIR.
        Nếu cache không có văn bản prompt hoặc token-binding khác, notebook dừng và xuất review để kiểm tra.
        ''')]
    cells.append(nb.code('''
        print('FROZEN ATTENTION GEOMETRY: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        NOUN_MANIFEST=Path('/kaggle/input/datasets/ducanh2403/coco-noun-box-train10k-v11')
        OUTPUT=Path('/kaggle/working/attention_geometry_diagnostic_200')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,NOUN_MANIFEST/'manifest_receipt.json'):
            assert p.is_file(),f'Missing input: {p}; sửa đúng mount path trong cell này.'
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists(),'Giữ output cũ; dùng session/output mới.'
        '''))
    cells.append(nb.bootstrap(sources,'attention-geometry'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow'])
        import torch
        assert torch.cuda.device_count()==2,'Select GPU T4x2.'
        run([sys.executable,'-u','test_diagnostic.py'],cwd=REPO)
        # Synthetic attention observer comparison only. Actual A is checked without switching its MHA kernel.
        run([sys.executable,'-u','-c',"from route_probe import numeric_checks; print(numeric_checks('cuda:0'))"],cwd=REPO)
        '''))
    cells.append(nb.code('''
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
            dataset_json=str(COCO_JSON),noun_manifest=str(NOUN_MANIFEST),output=str(OUTPUT),
            implementation_sha256=SOURCE_HASHES,project_commit=PROJECT_COMMIT)
        CONFIG=Path('/tmp/attention_geometry_config.json')
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        PROTOCOL=OUTPUT/'protocol.json'
        try:
            run([sys.executable,'-u','run_diagnostic.py','prepare','--config',CONFIG],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc_per_node=2','--max_restarts=0','run_diagnostic.py','worker','--config',PROTOCOL],cwd=REPO)
            run([sys.executable,'-u','run_diagnostic.py','finish','--config',PROTOCOL],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                for name,source in SOURCES.items():
                    target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_text(source,encoding='utf-8')
                run([sys.executable,'-u','run_diagnostic.py','archive','--config',CONFIG],cwd=REPO)
        print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
        '''))
    cells.append(nb.md('''
        Gửi `attention_geometry_diagnostic_200_review.zip` sau khi hoàn tất, kể cả khi dừng do thiếu metadata.
        Chỉ `receipt.json` với DIAGNOSTIC_COMPLETE xác nhận chạy đủ; ZIP tồn tại chưa phải thành công.
        `PREMISE_SIGNAL_GO_DESIGN_ONLY` chỉ cho phép bàn pilot tiếp, chưa chứng minh caption tăng hay phương pháp mới.
        Thời gian chưa benchmark; không dùng con số 0 GPU. Model đọc cache CLIP, text encoder chỉ xác minh token-binding.
        Nếu FAIL, gửi log; không đổi tolerance, bỏ bước kiểm hoặc tự bật thêm module/loss.
        '''))
    nb.HERE=HERE;nb.write('ATTENTION_GEOMETRY_DIAGNOSTIC_200_2GPU.ipynb',cells)
    path=HERE/'ATTENTION_GEOMETRY_DIAGNOSTIC_200_2GPU.ipynb'
    package=dict(status='PACKAGED_NOT_EXECUTED',notebook_bytes=path.stat().st_size,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sources_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
        local_model_run=False,user_runs_kaggle=True)
    (HERE/'PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
