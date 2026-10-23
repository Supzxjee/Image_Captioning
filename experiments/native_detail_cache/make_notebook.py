"""Generate a user-run full-cache notebook with hash-checked runtime overlays."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb);nb.HERE=HERE
nb.COMMIT='a2625edce0f4f554a36aec04354b158ce95b0a6f'


def main():
    bundle={n:(HERE/n).read_text(encoding='utf-8') for n in ('cache_data.py','run_full_cache.py','test_full_cache.py')}
    for n in ('tile_data.py','run_benchmark.py'):bundle[n]=(ROOT/'experiments/native_detail_benchmark'/n).read_text(encoding='utf-8')
    for n in ('captioning/data.py','captioning/visual_cache.py','captioning/scst_runtime.py'):bundle[n]=(ROOT/n).read_text(encoding='utf-8')
    cells=[nb.md('''
        # Full cache H/L — TRAIN10k + val5000, T4 ×2
        **Chỉ tạo cache, không huấn luyện, không đánh giá caption.**
        H: cắt bốn vùng ảnh gốc rồi resize224. L: resize toàn ảnh224 trước rồi cắt/phóng cùng bốn vùng.
        Dùng cùng CLIP frozen và projection frozen của A; pool14×14→7×7 trước projection/ReLU.
        Full ở đây là **15.000 ảnh đã khóa**, không phải toàn bộ COCO.

        Thêm **mscoco-2014**, **pair-source-fullframe-ce** (checkpoint A), **visual-cache**.
        Chọn **T4 ×2**, Internet ON, **Save & Run All**. Không cần API token.
        Dự kiến ~6,02GB tensor; ~55 phút là ước lượng từ benchmark, chưa tính mọi chi phí xác minh/copy và không bảo đảm ETA.

        30 shard ×500ảnh, mỗi shard có H và L. Chạy tiếp cùng session giữ nguyên OUTPUT.
        Nếu session cũ mất, phải lưu folder thành dataset rồi thêm vào input, điền RESUME_FROM bên dưới.
        Chỉ các shard hoàn tất và có hash hợp lệ được dùng lại; shard đang ghi dở làm lại tối đa500ảnh mỗi GPU.
        Việc resume không cứu được output chưa lưu khi Kaggle xóa session.

        Chỉ coi thành công khi có `cache_receipt.json` với `status=CACHE_COMPLETE` và 15.000ảnh.
        **Giữ cả folder `native_detail_HL_15k` làm dataset**, không upload ZIP review thay cache.
        Gửi `native_detail_HL_15k_review.zip` để kiểm tra. Review không chứa HDF5.
        ''')]
    cells.append(nb.code('''
        print('NATIVE DETAIL FULL CACHE: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        OUTPUT=Path('/kaggle/working/native_detail_HL_15k')
        RESUME_FROM=None  # Optional Path('/kaggle/input/datasets/OWNER/DATASET/native_detail_HL_15k')
        for p in (CHECKPOINT,COCO_JSON):assert p.is_file(),p
        for p in (COCO_IMAGES,VISUAL_CACHE):assert p.is_dir(),p
        if RESUME_FROM is not None:
            RESUME_FROM=Path(RESUME_FROM)
            assert (RESUME_FROM/'cache_spec.json').is_file(),'Point directly to folder containing cache_spec.json.'
        '''))
    cells.append(nb.bootstrap(bundle,'native-detail-full-cache'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','Pillow','numpy','pandas'])
        import torch
        assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(r) for r in range(2)),'Select T4x2.'
        run([sys.executable,'-u','test_full_cache.py'],cwd=REPO)
        CONFIG=Path('/tmp/native_detail_full_cache_config.json')
        cfg=dict(checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),images=str(COCO_IMAGES),visual_cache=str(VISUAL_CACHE),
            output=str(OUTPUT),resume_from=str(RESUME_FROM) if RESUME_FROM else None,
            groups_archive=str(REPO/'experiments/noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'),
            implementation_sha256=SOURCE_HASHES,project_commit=PROJECT_COMMIT)
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        try:
            run([sys.executable,'-u','run_full_cache.py','run','--config',CONFIG],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                # Keep each notebook attempt without overwriting previous run logs/source.
                import uuid
                source_dir=OUTPUT/'notebook_attempts'/uuid.uuid4().hex;source_dir.mkdir(parents=True)
                shutil.copyfile(LOG,source_dir/'notebook.log')
                for name,source in SOURCES.items():
                    destination=source_dir/name;destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_text(source,encoding='utf-8')
                run([sys.executable,'-u','run_full_cache.py','archive','--config',CONFIG],cwd=REPO)
        receipt=json.loads((OUTPUT/'cache_receipt.json').read_text(encoding='utf-8'))
        assert receipt['status']=='CACHE_COMPLETE' and receipt['images']==15000 and len(receipt['shards'])==30
        print('KEEP ENTIRE FOLDER AS DATASET:',OUTPUT,flush=True)
        print('SEND SMALL REVIEW ZIP:',str(OUTPUT)+'_review.zip',flush=True)
        print('No training or caption evaluation has run.',flush=True)
        '''))
    nb.write('NATIVE_DETAIL_CACHE_15K_2GPU.ipynb',cells)


if __name__=='__main__':main()
