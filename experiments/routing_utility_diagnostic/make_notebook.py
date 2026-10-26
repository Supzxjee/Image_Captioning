"""Build an unexecuted, small notebook with sealed138 metadata and hashed runtime overlays."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.COMMIT='aadedbc'


def main():
    sources={n:(HERE/n).read_text(encoding='utf-8') for n in ('utility_data.py','utility_intervention.py','run_utility.py',
        'test_utility_data.py','test_utility_tensors.py','cohort.json')}
    for folder,names in [('attention_geometry_diagnostic',['diagnostic_data.py','run_diagnostic.py']),
        ('coco_grounding_feasibility',['route_probe.py','audit.py']),('upt_hidden_pair',['hidden_data.py','hidden_metric_smoke.py']),
        ('upt_alignment_pilot',['pilot_data.py'])]:
        sources.update({n:(ROOT/'experiments'/folder/n).read_text(encoding='utf-8') for n in names})
    for n,s in sources.items():
        if n.endswith('.py'):ast.parse(s,filename=n)
    cohort_sha=hashlib.sha256(sources['cohort.json'].encode()).hexdigest()
    cells=[nb.md('''
        # Frozen A/T/W — decoder-utility diagnostic trên đúng138 ảnh TRAIN
        **Internet ON · GPU T4×2 · Save & Run All.** Hai GPU chia69 ảnh mỗi bên; không huấn luyện.
        A: gốc. T: chỉnh visual read của cụm về Q phù hợp box đúng. W: về Q của đối tượng khác,
        khớp độ lớn thay đổi trước gate/LN. Một mức convex mixing tối đa10%, không quét tham số.
        Sinh caption A/T/W và đo lợi ích/selectivity danh từ. GT box chỉ là oracle cho phép thử,
        không phải phương pháp inference trên ảnh mới.
        **Cần4 dataset:** checkpoint A, prompt-cache .pt gốc, visual-cache, MSCOCO2014 JSON.
        Metadata138 ảnh/box/first-noun spans được nhúng từ lượt đã audit; không cần upload lại
        review ZIP, JSON prompt, noun-box dataset hay UPT hidden. Input embedding giữ nguyên.
        ''')]
    cells.append(nb.code('''
        print('FROZEN A/T/W UTILITY: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        OUTPUT=Path('/kaggle/working/routing_utility_ATW_138')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON):assert p.is_file(),f'Missing input: {p}'
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists(),'Preserve previous output; use fresh session or output path.'
        '''))
    cells.append(nb.bootstrap(sources,'routing-utility-atw'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow',
            'pycocoevalcap==1.2','nltk==3.9.2','future'])
        run([sys.executable,'-m','pip','install','-q','--no-deps',
            'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
        run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
        run(['java','-version'])
        CHAIR=Path('/tmp/routing-utility-chair')
        checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
        assert (CHAIR/'chair.pkl').is_file(),'Pinned CHAIR cache missing.'
        import torch
        assert torch.cuda.device_count()==2,'Select GPU T4x2.'
        run([sys.executable,'-u','test_utility_data.py'],cwd=REPO)
        run([sys.executable,'-u','test_utility_tensors.py'],cwd=REPO)
        run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
        '''))
    cells.append(nb.code(f'''
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
            dataset_json=str(COCO_JSON),chair=str(CHAIR),output=str(OUTPUT),
            cohort_sha256='{cohort_sha}',implementation_sha256=SOURCE_HASHES,project_commit=PROJECT_COMMIT)
        CONFIG=Path('/tmp/routing_utility_config.json')
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        PROTOCOL=OUTPUT/'protocol.json'
        try:
            run([sys.executable,'-u','run_utility.py','prepare','--config',CONFIG],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc_per_node=2','--max_restarts=0','run_utility.py','worker','--config',PROTOCOL],cwd=REPO)
            run([sys.executable,'-u','run_utility.py','evaluate','--config',PROTOCOL],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                for name,source in SOURCES.items():
                    target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_text(source,encoding='utf-8')
                run([sys.executable,'-u','run_utility.py','archive','--config',CONFIG],cwd=REPO)
        print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
        '''))
    cells.append(nb.md('''
        Gửi **routing_utility_ATW_138_review.zip**. Không có checkpoint mới.
        ZIP khi lỗi chỉ chứa log/source/partial output; receipt ATW_COMPLETE mới xác nhận hoàn tất.
        Không tăng10%, đổi pair hoặc bỏ parity nếu FAIL. Nếu khó tìm lỗi, gửi cuối LOG.
        Ảnh không can thiệp được giữ caption A trong cả A/T/W, vẫn tính đủ138.
        CIDEr dùng TRAIN10k IDF cố định, không so trực tiếp trị số với val/test hoặc dev1000 cũ.
        Lượt này chưa chứng minh đóng góp mới hay gain trên ảnh chưa dùng; chỉ kiểm tra có đáng train tiếp không.
        '''))
    nb.HERE=HERE;nb.write('ROUTING_UTILITY_ATW_138_2GPU.ipynb',cells)
    path=HERE/'ROUTING_UTILITY_ATW_138_2GPU.ipynb'
    package=dict(status='PREPARED_UNEXECUTED',notebook_bytes=path.stat().st_size,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cohort_sha256=cohort_sha,
        embedded_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
        local_model_execution=False,actual_tensor_and_metric_preflight='USER_KAGGLE_ONLY')
    (HERE/'PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
