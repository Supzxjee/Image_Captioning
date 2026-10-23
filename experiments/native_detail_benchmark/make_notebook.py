"""Build one portable notebook; never executes a model locally."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.HERE=HERE
nb.COMMIT='879d9a7dd51038a0eb804b83cc996c53fd03d2b0'


def main():
    bundle={name:(HERE/name).read_text(encoding='utf-8') for name in ('tile_data.py','test_tile_data.py','run_benchmark.py')}
    for name in ('captioning/scst_runtime.py','captioning/visual_cache.py','captioning/data.py'):
        bundle[name]=(ROOT/name).read_text(encoding='utf-8')
    cells=[nb.md('''
        # Native-detail H/L — benchmark 256 ảnh, T4 ×2
        Chỉ đo chi phí cache, **không huấn luyện và không đánh giá caption**.
        H: cắt bốn vùng ảnh gốc rồi resize224. L: resize cả ảnh224 rồi cắt cùng bốn vùng và phóng224.
        Cả hai dùng cùng CLIP frozen và projection frozen từ checkpoint A.
        Mẫu khóa: 16 ảnh warmup + 256 ảnh đo từ TRAIN10k; không đọc câu tham chiếu để chọn ảnh.

        Thêm ba dataset: **mscoco-2014**, **pair-source-fullframe-ce** (checkpoint A), **visual-cache**.
        Chọn **T4 ×2**, Internet ON, **Save & Run All**. Không cần API token.
        Sau khi hoàn tất, gửi `native_detail_benchmark_256_review.zip`.
        HDF5 nhỏ chỉ là mẫu benchmark; chưa phải full cache để huấn luyện.
        ''')]
    cells.append(nb.code('''
        print('NATIVE DETAIL BENCHMARK: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        COCO_IMAGES=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        OUTPUT=Path('/kaggle/working/native_detail_benchmark_256')
        for p in (CHECKPOINT,COCO_JSON):assert p.is_file(),p
        for p in (COCO_IMAGES,VISUAL_CACHE):assert p.is_dir(),p
        assert not OUTPUT.exists(),'Preserve prior output; choose a fresh OUTPUT or session.'
        '''))
    cells.append(nb.bootstrap(bundle,'native-detail-benchmark'))
    cells.append(nb.code('''
        run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','Pillow','numpy','pandas'])
        import torch
        assert torch.cuda.device_count()==2,'Select GPU T4x2.'
        assert all('T4' in torch.cuda.get_device_name(r) for r in range(2)),'Benchmark protocol requires two T4 GPUs.'
        run([sys.executable,'-u','test_tile_data.py'],cwd=REPO)
        CONFIG=Path('/tmp/native_detail_benchmark_config.json')
        cfg=dict(checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),images=str(COCO_IMAGES),visual_cache=str(VISUAL_CACHE),
            output=str(OUTPUT),groups_archive=str(REPO/'experiments/noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'),
            project_commit=PROJECT_COMMIT,implementation_sha256=SOURCE_HASHES,
            protocol='native_detail_cache_benchmark_v1',benchmark_only=True)
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        try:
            run([sys.executable,'-u','run_benchmark.py','run','--config',CONFIG],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                shutil.copyfile(LOG,OUTPUT/'notebook.log')
                source_dir=OUTPUT/'source';source_dir.mkdir(exist_ok=True)
                for name,source in SOURCES.items():
                    destination=source_dir/name;destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_text(source,encoding='utf-8')
                run([sys.executable,'-u','run_benchmark.py','archive','--config',CONFIG],cwd=REPO)
        result=json.loads((OUTPUT/'benchmark_summary.json').read_text(encoding='utf-8'))
        assert result['status']=='COMPLETE','Benchmark failed; return review ZIP with logs.'
        print('BENCHMARK COMPLETE. No training or caption evaluation was run.',flush=True)
        print('SEND:',str(OUTPUT)+'_review.zip',flush=True)
        print('Full-cache estimate excludes training/evaluation; review quota before proceeding.',flush=True)
        '''))
    nb.write('NATIVE_DETAIL_BENCHMARK_256_2GPU.ipynb',cells)


if __name__=='__main__':main()
