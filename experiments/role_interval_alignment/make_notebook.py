"""Package only; no model/parser inference or Kaggle execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.COMMIT='64f0bf475d86f99e59564d29003adeb1109e7d07'
RUNTIME=['role_logic.py','role_parser.py','role_review.py','run_role.py','test_role.py','role_metadata.json']


def main():
    bundle={n:(HERE/n).read_text(encoding='utf-8') for n in RUNTIME}
    for name,path in [('diagnostic_data.py','experiments/attention_geometry_diagnostic/diagnostic_data.py'),
                      ('hidden_data.py','experiments/upt_hidden_pair/hidden_data.py'),
                      ('pilot_data.py','experiments/upt_alignment_pilot/pilot_data.py')]:
        bundle[name]=(ROOT/path).read_text(encoding='utf-8')
    for n,s in bundle.items():
        if n.endswith('.py'):ast.parse(s)
    cells=[nb.md('''
    # Role interval — feasibility 128 FIT images, frozen A, T4 ×2
    Chỉ kiểm tra bộ suy diễn ngữ nghĩa từ caption. **Không huấn luyện.** Internet ON, GPU T4 ×2, Save & Run All.
    Giữ checkpoint A và prompt gốc. Không cần UPT full cache, prompt JSON mới, CHAIR hoặc token API.
    Đầu ra: 128 cặp greedy/sample, graph/interval và 30 quyết định để kiểm tra trên HTML offline.
    Ngưỡng: ít nhất500 FIT/replay khả dụng và tách biệt;16/128 cặp có aG khác0;8/128 cặp có xử lý interval/binding ảnh hưởng quyết định.
    Qua ngưỡng số vẫn phải audit30 phép suy diễn (≥27 đúng, không lỗi thiếu nhãn→false). Không tự chuyển sang train.
    Cap30 phút cho hash/prepare/generate/score; cài môi trường/parser và xuất hình nằm ngoài cap. Đây là giới hạn dừng, không phải ETA.
    ''')]
    cells.append(nb.code('''
    print('ROLE INTERVAL FEASIBILITY: START',flush=True)
    from pathlib import Path
    CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
    LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
    VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
    COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
    COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
    STALE_ROOT=Path('/kaggle/input/notebooks/ducanh2403/stale')
    OUTPUT=Path('/kaggle/working/role_interval_feasibility_128')
    for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON):assert p.is_file(),p
    for p in (VISUAL_CACHE,COCO_IMAGES,STALE_ROOT):assert p.is_dir(),p
    assert not OUTPUT.exists(),'Preserve prior output; choose fresh OUTPUT/session.'
    '''))
    cells.append(nb.bootstrap(bundle,'role-interval-feasibility'))
    cells.append(nb.code('''
    run([sys.executable,'-m','pip','install','-q','numpy','Pillow','h5py','transformers==4.57.1','spacy==3.8.7'])
    import urllib.request
    sys.path.insert(0,str(REPO))
    from role_parser import MODEL_URL,MODEL_SHA
    WHEEL=REPO/'en_core_web_sm-3.8.0-py3-none-any.whl'
    print('Downloading pinned dependency parser (~12MB), CPU parsing on Kaggle only.',flush=True)
    urllib.request.urlretrieve(MODEL_URL,WHEEL)
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest()==MODEL_SHA,'Parser wheel SHA mismatch.'
    run([sys.executable,'-m','pip','install','-q','--no-deps',WHEEL])
    import torch
    assert torch.cuda.device_count()==2,'Select T4x2.'
    run([sys.executable,'-u','test_role.py'],cwd=REPO)
    cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
        dataset_json=str(COCO_JSON),images=str(COCO_IMAGES),stale_root=str(STALE_ROOT),output=str(OUTPUT),
        project_commit=PROJECT_COMMIT,implementation_sha256=SOURCE_HASHES)
    CONFIG=Path('/tmp/role_interval_config.json')
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    try:
        run([sys.executable,'-u','run_role.py','prepare','--config',CONFIG],cwd=REPO)
        CONFIG=OUTPUT/'protocol.json'
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
            'run_role.py','worker','--config',CONFIG],cwd=REPO)
        run([sys.executable,'-u','run_role.py','finish','--config',CONFIG],cwd=REPO)
    finally:
        if OUTPUT.is_dir():
            if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
            run([sys.executable,'-u','run_role.py','archive','--config',CONFIG],cwd=REPO)
    assert (OUTPUT/'receipt.json').is_file(),'Incomplete. Send review ZIP with logs; do not use partial output as a result.'
    print('SEND:',str(OUTPUT)+'_review.zip',flush=True)
    print('READ:',OUTPUT/'summary.json',flush=True)
    print('OFFLINE HUMAN REVIEW:',OUTPUT/'review/index.html',flush=True)
    '''))
    for i,c in enumerate(cells):c['id']=f'role-{i:02d}'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python',version='3.13')),cells=cells)
    payload=json.dumps(notebook,ensure_ascii=False,indent=1).encode('utf-8')
    if len(payload)>=1_000_000:raise ValueError('Notebook exceeds Kaggle source limit.')
    path=HERE/'ROLE_INTERVAL_FEASIBILITY_2GPU.ipynb';path.write_bytes(payload)
    report=dict(notebook_bytes=len(payload),notebook_sha256=hashlib.sha256(payload).hexdigest(),source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()},model_execution_local=False)
    (HERE/'PACKAGE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(path.name,len(payload),'bytes; unexecuted')


if __name__=='__main__':main()
