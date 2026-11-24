"""Build bank-only TEST from the successful matched-baseline TEST scaffold."""
import ast
import hashlib
import json
from pathlib import Path
from make_notebook import once, literals

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'experiments/matched_visual_baseline/BASELINE_MATCHED_PARALLEL3_V188_TEST5000_API_2GPU.ipynb'
NAME = 'M2_ONLY_BANK_ALIGNMENT_V245_TEST5000_API_2GPU.ipynb'
STATUS = 'M2_ONLY_BANK_ALIGNMENT_TEST5000_COMPLETE'


def literal_assignments(source):
    return literals(source)


def build():
    manifest = json.loads((PARENT.parent/'TEST_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != manifest['notebook_sha256']:
        raise ValueError('Changed successful TEST parent notebook')
    nb = json.loads(PARENT.read_text(encoding='utf-8'))
    setup = ''.join(nb['cells'][1]['source'])
    run = ''.join(nb['cells'][3]['source'])
    values = literals(setup)
    payloads = values['SOURCE_PAYLOADS']
    origins = dict(manifest['origins'])
    for name, code in payloads.items():
        if hashlib.sha256(code.encode('utf-8')).hexdigest() != values['SOURCE_SHA256'][name]:
            raise ValueError('Changed parent TEST source: '+name)
    for name in ('visual_only_model.py', 'baseline_test_data.py', 'BASELINE_TRAINING_LOCK.json',
                 'MATCHED_BASELINE_TEST_PROTOCOL.md'):
        del payloads[name]
        origins.pop(name, None)
    training_nb = json.loads((HERE/'M2_ONLY_BANK_ALIGNMENT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
    training = literals(''.join(training_nb['cells'][1]['source']))['SOURCE_PAYLOADS']
    training_manifest = json.loads((HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name in ('csa_patch_model.py', 'bank_layers.py', 'semantic_patch_model.py', 'memory_fusion.py', 'bank_only_model.py'):
        payloads[name] = training[name]
        origins[name] = dict(training_notebook=True, training_source_sha256=training_manifest['sha256'][name])
    for name in ('bank_only_test_data.py', 'bank_only_inference_guard.py'):
        payloads[name] = (HERE/name).read_text(encoding='utf-8')
        origins[name] = dict(bank_only_TEST_override=True)
    result = json.loads((HERE/'FULL_XE10_RESULT.json').read_text(encoding='utf-8'))
    if result['source_version'] != 'ducanh2403/2-vi-cai-kia-qua-nhieu/245' or result['best_epoch'] != 9:
        raise ValueError('Need audited M2-only v245 best9')
    payloads['M2_ONLY_TRAINING_LOCK.json'] = json.dumps(dict(source_version=result['source_version'],
        best_checkpoint_sha256=result['receipts']['best_receipt.json']['sha256'],
        best_VALID_CIDEr=result['metrics_by_epoch']['9']['CIDEr'], source_sha256=training_manifest['sha256']), indent=2)+'\n'
    payloads['M2_ONLY_TEST_PROTOCOL.md'] = (HERE/'TEST_PROTOCOL.md').read_text(encoding='utf-8')
    worker = payloads['run_csa_per_caption_test.py']
    worker = worker.replace('matched_baseline', 'bank_only').replace('baseline_test_data', 'bank_only_test_data')
    worker = worker.replace('matched visual-only baseline TEST', 'M2-only TEST')
    worker = once(worker,
        "        from visual_only_model import VisualOnlyCaptioner,VisualOnlyContext\n"
        "        model=VisualOnlyCaptioner(len(words));context=VisualOnlyContext(cfg['stage'])",
        "        from bank_only_model import BankOnlyCaptioner,BankOnlyContext\n"
        "        model=BankOnlyCaptioner(len(words),ck['model']['core.nouns'],ck['model']['core.verbs']);context=BankOnlyContext(cfg['stage'])")
    worker = once(worker,
        "                memory=model.encode(x,context.images(index,device)) if context else model.encode(x);seq,scores=beam_search(model,memory,width=locked['beam_width'],maximum=locked['max_new_tokens'])",
        "                memory=model.encode(x,context.images(index,device))\n"
        "                if offset==0:\n"
        "                    from bank_only_inference_guard import inference_guard\n"
        "                    inference_guard(model,x,memory,rank,out)\n"
        "                seq,scores=beam_search(model,memory,width=locked['beam_width'],maximum=locked['max_new_tokens'])")
    worker = once(worker, "runtime_policy=policy,first_GPU_beam_path_check_passed=True)",
                  "runtime_policy=policy,bank_guard_sha256=sha(out/f'bank_only_inference_rank{rank}.json'),first_GPU_beam_path_check_passed=True)")
    worker = worker.replace('BASELINE_MATCHED_PARALLEL3_TEST5000_COMPLETE', STATUS)
    worker = once(worker, "images=5000,checkpoint_sha256=contracts(cfg).CHECKPOINT_SHA,source_version=",
                  "images=5000,seed=42,method='bank_only_parallel3_xe10_v1',checkpoint_sha256=contracts(cfg).CHECKPOINT_SHA,source_version=")
    payloads['run_csa_per_caption_test.py'] = worker
    origins['run_csa_per_caption_test.py'] = dict(origins['run_csa_per_caption_test.py'], bank_only_TEST_override=True)
    hashes = {name:hashlib.sha256(code.encode('utf-8')).hexdigest() for name,code in payloads.items()}
    for name, code in payloads.items():
        if name.endswith('.py'):
            ast.parse(code, filename=name)
    setup = setup.split('\nSOURCE_PAYLOADS = ', 1)[0]
    setup = setup.replace('matched_baseline=True', 'bank_only=True')
    setup = setup.replace('baseline_matched_parallel3_v188_test5000', 'm2_only_bank_alignment_v245_test5000')
    setup = setup.replace('ducanh3/notebook7bc83c4c76/188', result['source_version'])
    setup += '\nSOURCE_PAYLOADS = '+repr(payloads)+'\nSOURCE_SHA256 = '+repr(hashes)+'\n'
    run = run.replace('from baseline_test_data import', 'from bank_only_test_data import')
    run = run.replace('BASELINE_MATCHED_PARALLEL3_TEST5000_COMPLETE', STATUS)
    intro = '''# M2-only — TEST5000, exact v245 best9

**GPU T4 ×2, Internet ON, Save & Run All.** Chỉ gắn hai Dataset: `ducanh2403/visual-cache` và `vuthetam/mscoco-2014`. Bank noun/verb đã nằm trong checkpoint; không cần Datasetbank130/prompt175/TEST181 hoặc Datasetcheckpoint mới.

Bật Secret **codex** hoặc **KAGGLE_API_TOKEN** có quyền đọc `ducanh2403/2-vi-cai-kia-qua-nhieu/245`. CHECKPOINT_ROOT trống: API chỉ tải best.bin exactv245 và kiểm SHA3fe31a92…; có thể gắn đúngfile và điền CHECKPOINT_ROOT nếu đã có. Không dùng final10/v243/baseline/zero-text/full checkpoint, không fallbacklatest. Không gửi token vào chat.

Khóa bestepoch9/seed42/159399updates đã chọn bằng maxVALIDCIDEr. Giữ nguyên BankOnlyCaptioner/core/M2/decoder3lớp từtrain,49.239.288tham số; hai bank lấy từbuffer checkpoint. BỏM3, không ROI/prompt/detector. Đầu mỗiGPU kiểm trên8ảnhthật: bankfinite/L2, memoryđúngcoreM2, promptgiảkhôngảnhhưởng, khôngadapterM3; strictload và beamrescore. Chỉ inference FP32/T4x2/beam5/max20/rawlogprob, không training/SCST/rerank/tuning.

Cùng cohort5000TEST/vocabulary/patch/scorers với baseline/zero/full đã chấm. Stage chỉTEST≈1.4GiB, không copy fullTRAINcache hoặc chạyCLIP lại. Lưu top1 và5beam/ảnh trướcCPUmetrics; đủ BLEU1/4,METEOR,ROUGE-L,CIDEr,SPICE,CHAIRs/i (+Recall,NLL) mới complete. ReferenceTEST dùngchấmsau decode, khôngchọncâu. Từngphase lưu riêng để CPUrecovery nếu lỗi; không tự sinh lại caption.

Source đãnhúng/hash; dependencies/Java/CHAIR vẫn cầnInternet. Cap2h gồmsetup/scorers là giới hạn, khôngETA/billing. ThiếuAPI/CUDA/metric thì giữlog/output, không tự đổiTorch/checkpoint. Gửi `m2_only_bank_alignment_v245_test5000_review.zip`, giữ toàn bộOutput. Host chỉstatic/mock; actualAPI/binary/CUDA/Java/scorers pendinguser-run. Mộtseed, tổngtham sốkhác nhau vàpriorTESTexposure của dự án đượcgiữtrongbáocáo.
'''
    for i, source in ((0,intro),(1,setup),(3,run)):
        nb['cells'][i]['source'] = source.splitlines(keepends=True)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'cell{i}','exec')
            cell['execution_count'],cell['outputs'] = None,[]
    target=HERE/NAME
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (HERE/'TEST_SOURCE_MANIFEST.json').write_text(json.dumps(dict(parent_notebook=str(PARENT.relative_to(ROOT)),
        parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),source_version=result['source_version'],
        checkpoint_sha256=result['receipts']['best_receipt.json']['sha256'],sha256=hashes,origins=origins,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
    print(NAME,len(payloads),'embedded files; no model/API/scorer executed')
    return nb,payloads


if __name__=='__main__':
    build()
