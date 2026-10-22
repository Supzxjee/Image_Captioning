"""Package an unexecuted Kaggle smoke, with only four embedded TRAIN masks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from audit import geometry

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('hidden_notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb)
nb.COMMIT='6acd3c9f0553d4d12d46684bbe7dcb8c38033d6b'


def main():
    audit=json.loads((HERE/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    samples=[]
    for row in audit['manual_sample'][:4]:
        row=dict(row);region=geometry(row['bbox_xywh'],*row['image_size'],include_mask=True)
        mask=np.asarray(region['patch_area_fraction'])
        assert mask.shape==(14,14) and abs(mask.sum()-row['geometry']['patch_area_equivalents'])<1e-8
        row['patch_area_fraction']=region['patch_area_fraction'];samples.append(row)
    assert [r['coco_id'] for r in samples]==[64627,529017,468966,580507]
    sample_text=json.dumps(samples,ensure_ascii=False,indent=2)+'\n'
    bundle={name:(HERE/name).read_text(encoding='utf-8') for name in ['route_probe.py','run_route_smoke.py']}
    bundle['train4_masks.json']=sample_text
    for name,text in bundle.items():
        if name.endswith('.py'):ast.parse(text,filename=name)
    models_sha=hashlib.sha256((ROOT/'captioning/models.py').read_text(encoding='utf-8').encode()).hexdigest()
    cells=[nb.md('''
        # COCO noun–patch routing — interface smoke, T4×2
        **Internet ON · T4×2 · Save & Run All.** Người dùng tự chạy.
        Chỉ kiểm interface/gradient trên **4 ảnh TRAIN**; mỗi GPU hai ảnh, không DDP collectives.
        **0 optimizer updates, không huấn luyện, không CIDEr/CHAIR, không đánh giá dev/test.**
        Dùng checkpoint original SCST-object đã khóa, không dùng checkpoint A-end/CSA/GV.
        Giữ nguyên forward baseline. Quan sát attention bằng phép tính Q/K dùng chung trọng số;
        đối chiếu với PyTorch MHA, kiểm logits/greedy parity, mask, causal offset và gradients.
        Không chuẩn hóa lại Q slice hoặc patch slice sau khi bỏ legacy/CLS.
        Box chỉ là mask cho kiểm số học; chưa phải nhãn grounding sạch hay bằng chứng gain.
        '''),nb.code('''
        print('COCO ROUTING INTERFACE SMOKE: START',flush=True)
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        OUTPUT=Path('/kaggle/working/coco_routing_interface_smoke')
        for p in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON):
            assert p.is_file(),p
        assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
        assert not OUTPUT.exists(),'Preserve existing output; use fresh session or OUTPUT.'
        OUTPUT.mkdir(parents=True)
        '''),nb.bootstrap(bundle,'coco-routing-interface')]
    cells.append(nb.code(f'''
        import zipfile
        CONFIG=OUTPUT/'protocol.json'
        cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
            dataset_json=str(COCO_JSON),output=str(OUTPUT),project_commit=PROJECT_COMMIT,
            source_sha256=SOURCE_HASHES,models_source_sha256='{models_sha}',
            samples=json.loads((REPO/'train4_masks.json').read_text()),
            rtol=1e-5,atol=1e-6,route_epsilon=1e-8,optimizer_updates=0,
            policy='FP32 eval, strict math SDP, original calls unchanged; two independent T4 workers, no updates',
            route='Last-layer head means; raw decoder-Q slice times raw Q-spatial slice; no renormalization',
            test_used=False,dev_used=False,caption_quality_evaluated=False,
            data_audit_sha256='{hashlib.sha256((HERE/'DATA_AUDIT.json').read_bytes()).hexdigest()}')
        CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        for name,source in SOURCES.items():
            target=OUTPUT/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source,encoding='utf-8')
        try:
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','h5py','numpy','Pillow','tqdm'])
            run([sys.executable,'-u','run_route_smoke.py','prepare','--config',CONFIG],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc_per_node=2','--max_restarts=0','run_route_smoke.py','worker','--config',CONFIG],cwd=REPO)
            receipts=[json.loads((OUTPUT/f'complete_rank{{rank}}.json').read_text()) for rank in range(2)]
            assert all(r['status']=='INTERFACE_SMOKE_PASS' and r['optimizer_updates']==0 for r in receipts)
            assert receipts[0]['before_state_sha256']==receipts[1]['before_state_sha256']
            print('INTERFACE SMOKE COMPLETE: 0 updates; caption gain NOT evaluated.',flush=True)
        finally:
            if LOG.is_file():shutil.copyfile(LOG,OUTPUT/'notebook.log')
            review=Path(str(OUTPUT)+'_review.zip')
            with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
                for p in sorted(OUTPUT.rglob('*')):
                    if p.is_file() and p.suffix in ('.json','.log','.py'):
                        archive.write(p,p.relative_to(OUTPUT))
            print('SEND REVIEW ZIP:',review,flush=True)
        '''))
    cells.append(nb.md('''
        Gửi **coco_routing_interface_smoke_review.zip**, kể cả khi test lỗi.
        Nếu không thấy traceback gốc: `print(LOG.read_text(encoding='utf-8')[-16000:])`.
        Không đổi tolerance hoặc tiếp tục sang training nếu FAIL. PASS chỉ xác nhận kỹ thuật trên batch nhỏ.
        Không cần ảnh gốc, detector cache, UPT, VG hoặc full annotations. Không lưu checkpoint mới.
        Đường dẫn khác giữa hai tài khoản: chỉ sửa bốn INPUT paths ở cell đầu, giữ nguyên nội dung dataset.
        '''))
    for i,c in enumerate(cells):c['id']=f'coco-routing-smoke-{i:02d}'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    raw=json.dumps(notebook,ensure_ascii=False,indent=1).encode('utf-8')
    assert len(raw)<1_000_000
    for c in cells:
        if c['cell_type']=='code':assert c['execution_count'] is None and not c['outputs'];ast.parse(''.join(c['source']))
    (HERE/'COCO_ROUTING_INTERFACE_SMOKE_2GPU.ipynb').write_bytes(raw)
    print('Unexecuted notebook:',len(raw),'bytes; sources:',list(bundle),'TRAIN IDs:',[r['coco_id'] for r in samples])


if __name__=='__main__':main()
