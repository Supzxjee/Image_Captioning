"""Build M2-only from the successful, source-contained matched baseline runner."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'experiments/matched_visual_baseline/BASELINE_MATCHED_PARALLEL3_XE10_2GPU.ipynb'
METHOD = 'bank_only_parallel3_xe10_v1'
STATUS = 'BANK_ONLY_PARALLEL3_XE10_COMPLETE'
NAME = 'M2_ONLY_BANK_ALIGNMENT_SEED42_XE10_2GPU.ipynb'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Changed source anchor: '+old[:120])
    return source.replace(old, new, 1)


def literals(source):
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def build():
    parent_manifest = json.loads((PARENT.parent/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    parent_sha = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    if parent_sha != parent_manifest['notebook_sha256']:
        raise ValueError('Changed matched baseline notebook')
    nb = json.loads(PARENT.read_text(encoding='utf-8'))
    setup = ''.join(nb['cells'][1]['source'])
    values = literals(setup)
    payloads = values['SOURCE_PAYLOADS']
    origins = values['SOURCE_ORIGINS']
    for name, code in payloads.items():
        if hashlib.sha256(code.encode('utf-8')).hexdigest() != values['SOURCE_SHA256'][name]:
            raise ValueError('Changed embedded parent source: '+name)
    for name in ('visual_only_model.py', 'MATCHED_BASELINE_PROTOCOL.md'):
        del payloads[name]
        origins.pop(name, None)
    runner = payloads['run_csa_full.py'].replace('matched_visual_parallel3_xe10_v1', METHOD)
    runner = once(runner, 'from visual_only_model import VisualOnlyContext as SemanticContext',
                  'from bank_only_model import BankOnlyContext as SemanticContext')
    runner = once(runner,
        "        if cfg.get('semantic') or stage.get('semantic'):raise ValueError('Visual-only baseline cannot use semantic staging')",
        "        if (cfg.get('semantic') or stage.get('semantic') or not cfg.get('bank_alignment')\n"
        "            or stage.get('bank_alignment',{}).get('method')!='bank_only_v130'):\n"
        "            raise ValueError('M2-only requires sealed bank130 and no image-specific semantics')")
    runner = once(runner, 'decoder_layers=3,semantic_conditioning=False,software=',
                  'decoder_layers=3,semantic_conditioning=True,image_specific_prompts=False,software=')
    runner = once(runner, '        prepare(cfg)\n',
                  "        prepare(cfg)\n        if cfg.get('bank_alignment'):\n"
                  '            from bank_only_data import stage_banks\n            stage_banks(cfg,deadline)\n')
    payloads['run_csa_full.py'] = runner

    runtime = payloads['csa_full_runtime.py']
    a = runtime.index('def make_model(')
    b = runtime.index('\n\ndef ', a+1)
    runtime = runtime[:a]+'''def make_model(vocab,cfg):
    import torch
    from bank_only_model import BankOnlyCaptioner
    if (cfg.get('method')!='bank_only_parallel3_xe10_v1' or cfg.get('semantic')
        or not cfg.get('bank_alignment') or cfg.get('hybrid_sharing')
        or cfg.get('semantic_strength_gate') or cfg.get('seed')!=42):
        raise ValueError('Expected fresh seed42 M2-only recipe')
    banks=[torch.from_numpy(np.load(Path(cfg['stage'])/('sem_'+k+'_bank.npy'),allow_pickle=False)) for k in ('noun','verb')]
    return BankOnlyCaptioner(vocab,*banks)
''' + runtime[b:]
    runtime = once(runtime,
        'from visual_only_model import VisualOnlyContext as SemanticContext, contract_guard',
        'from bank_only_model import BankOnlyContext as SemanticContext, contract_guard')
    payloads['csa_full_runtime.py'] = runtime
    score = payloads['score_csa_full.py'].replace('matched_visual_parallel3_xe10_v1', METHOD)
    score = score.replace('MATCHED_VISUAL_PARALLEL3_XE10_COMPLETE', STATUS)
    a, b = score.index("    r['limits'] ="), score.index('    save(path,r)')
    score = score[:a]+'''    r['limits'] = ['One seed and reused VALID; TEST not evaluated.',
        'M1/M2 retained; only the two image-specific M3 adapters removed.',
        '49,239,288 expected parameters versus full53,183,736; not capacity matched.',
        'Historical fullseed42 patch staging differs; direct H5 matches the new baseline/zero-text arms.']
    r['bank_only_contract'] = read(Path(cfg['output'])/'bank_only_contract_rank0.json')
    r['bank_stage'] = read(Path(cfg['output'])/'bank_stage.json')
    r['image_specific_prompts'] = False
''' + score[b:]
    payloads['score_csa_full.py'] = score
    for name in ('bank_only_model.py', 'bank_only_data.py'):
        payloads[name] = (HERE/name).read_text(encoding='utf-8')
    payloads['M2_ONLY_PROTOCOL.md'] = (HERE/'PROTOCOL.md').read_text(encoding='utf-8')
    for name in ('run_csa_full.py', 'csa_full_runtime.py', 'score_csa_full.py',
                 'bank_only_model.py', 'bank_only_data.py', 'M2_ONLY_PROTOCOL.md'):
        origins[name] = dict(origins.get(name, {}), M2_only_override=True)
    hashes = {name: hashlib.sha256(code.encode('utf-8')).hexdigest() for name, code in payloads.items()}
    for name, code in payloads.items():
        if name.endswith('.py'):
            ast.parse(code, filename=name)

    setup = setup.split('\nSOURCE_PAYLOADS = ', 1)[0]
    setup = setup.replace('matched_visual_parallel3_xe10_v1', METHOD)
    setup = setup.replace('baseline_matched_parallel3_seed', 'm2_only_bank_alignment_seed')
    setup = once(setup, 'SEED=42',
                  "BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')\nSEED=42")
    setup = once(setup, 'for p in (VISUAL_CACHE,COCO_JSON):', 'for p in (VISUAL_CACHE,COCO_JSON,BANK_ROOT):')
    setup = once(setup, 'semantic=False,prepare_seconds=',
                  'semantic=False,bank_alignment=True,bank_root=str(BANK_ROOT),prepare_seconds=')
    setup = setup.replace('Resume must be from this matched baseline and seed; full-model/old-baseline checkpoints are incompatible',
                          'Resume must be this M2-only arm and seed; full/zero-text/baseline checkpoints are incompatible')
    setup += '\nSOURCE_PAYLOADS = '+repr(payloads)+'\nSOURCE_SHA256 = '+repr(hashes)+'\nSOURCE_ORIGINS = '+repr(origins)+'\n'
    run = ''.join(nb['cells'][3]['source'])
    run = once(run, 'from visual_only_model import VisualOnlyCaptioner;',
               'from bank_only_model import BankOnlyCaptioner;')
    run = run.replace('Matched baseline imports PASS', 'M2-only imports PASS')
    run = run.replace('MATCHED_VISUAL_PARALLEL3_XE10_COMPLETE', STATUS)
    run = run.replace('MATCHED BASELINE FULL10 COMPLETE;', 'M2-ONLY FULL10 COMPLETE;')
    intro = '''# M2-only: bank + căn chỉnh bank–ảnh — seed42 / CE10

**GPU T4 ×2, Internet ON, Save & Run All.** Tài khoản báo còn30hGPU; cap11.5h/phiên là giới hạn, không ETA hoặc quota thực tính.

Gắn đúng ba Dataset:
1. `ducanh2403/visual-cache`
2. `vuthetam/mscoco-2014` (có `dataset_coco.json`)
3. `ducanh2403/csa-semantic-banks-v130`

CLIP patch đã cache → projection → M2 dùng noun/verb bank → hai memory đối tượng/quan hệ → decoder parallel3lớp/dropout0.2. **Bỏ hoàn toàn M3, không gắn prompt175 hoặc ROI/UPT/YOLO.** Constructor full rồi xóa riêng hai adapter để giữ khởi tạo các phần còn lại; không dùng trọng số đã học. Dự kiến49.239.288tham số, Kaggle guard phải xác nhận.

Freshseed42,10epochCE,113287TRAIN/5000VALID,566747caption events/epoch,global32,177110updates. Cùng sampler/optimizer/lịchLR/decoder/beam5/max20/chọnbest=maxVALIDCIDEr với baseline3lớp và full. FP32/Accelerate1.15.0/DDP/per-caption, không sharing/AMP/compile. Đọc H5 trực tiếp, chỉ stage hai bank nhỏ; không copy fullpatch hoặc chạy model trích đặc trưng lại. Source đã nhúng/hash; không cần Secret. Dependencies/evaluator vẫn tải qua Internet.

Kaggle preflight: giữ mọi tensor/buffer M1/M2/decoder như full cùng seed; chỉ thiếu M3, output đúng coreM2 và không phụ thuộc prompt/ROI. Positivecontrol chỉ zero bank trên copy, phải đổi memory; originalstate giữ nguyên. Kiểm causal/beam, DDPfirst32/tail27 và64timingupdates bị bỏ; worker chính khởi tạo lại. Local chỉ static/mock, chưa có GPU PASS.

Nếu chạm cap, giữ **toàn bộ OUTPUT** gồm checkpoint.bin/best.bin/receipts/identity/evaluations/source. Resume chính arm bằng RESUME_ROOT và RUN_TAG='run2'; không nạp full/zero-text/baseline checkpoint, không tự đổi lịch/batch/precision. Best chọn bằng VALIDCIDEr; chấm caption best/end bằng cùng bộ đo VALID. Chưa SCST/rerank/TEST/METEOR/SPICE trong lượt này.

Gửi `m2_only_bank_alignment_seed42_run1_review.zip`, giữ trọng số đầy đủ. Mộtseed/khác tổng tham số/historicalI/O khác nhau phải được ghi nhận khi diễn giải. Không bảo đảm điểm tăng; đây là đối chứng thành phần còn thiếu.
'''
    for index, source in ((0, intro), (1, setup), (3, run)):
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell{i}', 'exec')
            cell['execution_count'], cell['outputs'] = None, []
    target = HERE/NAME
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(method=METHOD,
        parent_notebook=str(PARENT.relative_to(ROOT)), parent_sha256=parent_sha,
        revision=values['REVISION'], sha256=hashes, origins=origins,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(NAME, len(payloads), 'embedded files; no models executed')
    return nb, payloads


if __name__ == '__main__':
    build()
