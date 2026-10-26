"""Package user-run preparation and disposable cost profiles, never launch a pilot."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('nb',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.COMMIT='bf997e1'  # Resolved to full40 SHA below; short fetches fail on Kaggle.
import subprocess
nb.COMMIT=subprocess.check_output(['git','rev-parse',nb.COMMIT],cwd=ROOT,text=True).strip()
assert len(nb.COMMIT)==40


def main():
    sources={n:(HERE/n).read_text(encoding='utf-8') for n in ('stale_data.py','run_prepare.py','profile_stale.py','test_stale_data.py','test_stale_runtime.py')}
    for folder,names in [('attention_geometry_diagnostic',['diagnostic_data.py','run_diagnostic.py']),
                         ('upt_hidden_pair',['hidden_data.py']),('upt_alignment_pilot',['pilot_data.py']),
                         ('coco_grounding_feasibility',['audit.py'])]:
        sources.update({n:(ROOT/'experiments'/folder/n).read_text(encoding='utf-8') for n in names})
    for n,s in sources.items():ast.parse(s,filename=n)
    cells=[nb.md('''
        # Stale semantic prompt — chuẩn bị4điều kiện + kiểm tra chi phí, chưa chạy pilot
        **Internet ON · T4×2 · Save & Run All.** Tối đa2000ảnh TRAIN, một target/ảnh.
        Tạo cache ảnh che segmentation union và prompt trung hòa; mã hóa lại bằng CLIP frozen.
        Đo4điều kiện trên128ảnh hash-first; xuất16panel kiểm tra.
        Sau đó benchmark **50batch forward–backward mỗi nhánh C/T**, không optimizer.step,
        kiểm tra trọng số A giữ nguyên; không tự chạy1250update hay đánh giá gain caption.
        Cần checkpoint A, prompt-cache .pt, visual-cache, MSCOCO2014 **JSON+ảnh**, model promt-cache JSON gốc.
        Tải official COCO annotations~253MB vào/tmp nếu chưa có archive; không tải ảnh thêm.
        Đầu ra cache dự kiến khoảng2,6GB vì lưu cả feature factual và che, cộng input benchmark~300MB.
        Trần30phút prep và2giờ/arm là ngân sách đề xuất; chi phí thực chưa đo.
        ''')]
    cells.append(nb.code('''
        print('STALE PROMPT PREPARATION: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
        PROMPT_TEXT=Path('/kaggle/input/models/ducanh2403/promt-cache/maxtext/default/1/prompt_cache.json')
        ANNOTATION_ARCHIVE=Path('/tmp/stale-coco-annotations/annotations_trainval2014.zip')
        OUTPUT=Path('/kaggle/working/stale_prompt_preparation_2000')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,PROMPT_TEXT):assert p.is_file(),p
        for p in (VISUAL_CACHE,COCO_IMAGES):assert p.is_dir(),p
        assert not OUTPUT.exists(),'Preserve previous output; use fresh session or output path.'
        '''))
    cells.append(nb.bootstrap(sources,'stale-prompt-preparation'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','ijson','h5py','numpy','Pillow'])
        import torch
        assert torch.cuda.device_count()==2,'Select T4x2.'
        run([sys.executable,'-u','test_stale_data.py'],cwd=REPO)
        run([sys.executable,'-u','test_stale_runtime.py'],cwd=REPO)
        '''))
    cells.append(nb.code('''
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),
            images=str(COCO_IMAGES),prompt_text_json=str(PROMPT_TEXT),annotation_archive=str(ANNOTATION_ARCHIVE),output=str(OUTPUT),
            implementation_sha256=SOURCE_HASHES,project_commit=PROJECT_COMMIT)
        CONFIG=Path('/tmp/stale_prompt_preparation.json');CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        PROTOCOL=OUTPUT/'protocol.json'
        try:
            run([sys.executable,'-u','run_prepare.py','prepare','--config',CONFIG],cwd=REPO)
            for script,mode in [('run_prepare.py','cache'),('profile_stale.py','scores'),('profile_stale.py','benchmark')]:
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                    '--max_restarts=0',script,mode,'--config',PROTOCOL],cwd=REPO)
            run([sys.executable,'-u','run_prepare.py','finish','--config',PROTOCOL],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
                for name,source in SOURCES.items():
                    target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source,encoding='utf-8')
                run([sys.executable,'-u','run_prepare.py','archive','--config',CONFIG],cwd=REPO)
        print('KEEP ENTIRE CACHE:',OUTPUT,flush=True)
        print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
        '''))
    cells.append(nb.md('''
        Giữ toàn bộ folder cache nếu muốn dùng tiếp; ZIP review không chứa HDF5/.pt.
        Gửi `stale_prompt_preparation_2000_review.zip`; xem panel.html +16ảnh để kiểm tra mask/text.
        PREPARATION_COMPLETE không có nghĩa pilot đã được duyệt. Cần kết quả chi phí, panel và phản biện Argon/Astra.
        Benchmark không cập nhật trọng số. Optimizer latency chưa đo; AdamW state memory chỉ ước tính, không cấp phát.
        Nếu model/preprocessing parity hoặc nguồn segmentation lỗi, gửi log; không thay bằng zeroing feature cache.
        '''))
    nb.HERE=HERE;nb.write('STALE_PROMPT_PREPARATION_2GPU.ipynb',cells)
    path=HERE/'STALE_PROMPT_PREPARATION_2GPU.ipynb'
    (HERE/'PACKAGE.json').write_text(json.dumps(dict(status='PREPARED_UNEXECUTED',bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),project_commit=nb.COMMIT,
        sources_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},local_model_execution=False),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
